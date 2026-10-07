"""Real CPU import/conversion, source confinement and saved annotation coverage."""

import asyncio
import io
import json
import subprocess
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from vla_platform.api import create_app
from vla_platform.datasets.library import DatasetLibrary, LibraryError
from vla_platform.datasets.local_preview import reader_python
from vla_platform.datasets.snapshots import create_snapshot, resolve_snapshot
from vla_platform.settings import Settings


async def terminal(library, ident):
    for _ in range(200):
        value = library.get(ident)
        if value["status"] != "converting":
            return value
        await asyncio.sleep(0.05)
    raise AssertionError("Conversion did not finish")


def run_worker(root, operation, **kwargs):
    from vla_platform.datasets import import_worker

    response = subprocess.run(
        [str(reader_python()), "-I", str(Path(import_worker.__file__))],
        input=json.dumps({"root": str(root), "operation": operation, **kwargs}).encode(),
        capture_output=True,
        timeout=30,
    )
    result = json.loads(response.stdout)
    assert response.returncode == 0, result
    return result


def test_example_labels_export_and_training_snapshot(tmp_path):
    async def check():
        library = DatasetLibrary(tmp_path.resolve())
        entry = await library.example("project-one")
        value = await terminal(library, entry["id"])
        assert value["status"] == "ready", value
        assert value["converted"]["frames"] == 24
        assert value["converted"]["episodes"] == 2
        assert len(value["converted"]["cameras"]) == 2
        sample = library.samples(value["id"])[0]
        camera = sample["camera"]
        key = f"{sample['episode_index']}:{sample['frame_index']}:{camera}"
        library.annotate(
            value["id"],
            {"revision": 0, "views": {camera: "Front camera"}, "frames": {key: "Red block"}},
        )
        with pytest.raises(LibraryError, match="another window"):
            library.annotate(value["id"], {"revision": 0, "views": {}, "frames": {}})
        restored = DatasetLibrary(tmp_path.resolve(), reconcile=False)
        assert restored.get(value["id"])["annotations"]["frames"][key] == "Red block"
        exported = library.export(value["id"])
        with zipfile.ZipFile(exported) as archive:
            assert (
                json.loads(archive.read("openjensen/annotations.json"))["views"][camera]
                == "Front camera"
            )
            assert json.loads(archive.read("meta/info.json"))["total_frames"] == 24
        exported.unlink()
        store = tmp_path / "snapshots"
        store.mkdir()
        snapshot = await asyncio.to_thread(
            create_snapshot,
            library.root,
            str(library.resolve(value["id"], "project-one")),
            store,
            python=reader_python(),
        )
        assert resolve_snapshot(store, snapshot).is_dir()
        assert snapshot["total_frames"] == 24
        assert (await library.example("project-one"))["id"] == value["id"]
        await library.close()

    asyncio.run(check())


@pytest.mark.parametrize("family", ["robomimic", "aloha"])
def test_hdf5_adapters_preserve_actions_and_cameras(tmp_path, family):
    root = tmp_path / "source"
    root.mkdir()
    script = """
import h5py, numpy as np, sys
from pathlib import Path
root=Path(sys.argv[1]); family=sys.argv[2]
for episode in range(2):
    with h5py.File(root/f"episode_{episode}.hdf5", "w") as f:
        g=f.create_group("data/demo_0") if family=="robomimic" else f
        g.create_dataset("actions" if family=="robomimic" else "action",
                         data=np.full((3,2),episode+.25,dtype=np.float32))
        g.create_dataset("obs/joints" if family=="robomimic" else "observations/qpos",
                         data=np.full((3,2),episode+.5,dtype=np.float32))
        g.create_dataset("obs/front_image" if family=="robomimic" else "observations/images/front",
                         data=np.full((3,16,16,3),50+episode,dtype=np.uint8))
"""
    subprocess.run([str(reader_python()), "-I", "-c", script, str(root), family], check=True)
    detection = run_worker(root, "detect")
    assert detection["format"] == ("robomimic_hdf5" if family == "robomimic" else "aloha_hdf5")
    output = tmp_path / "converted"
    result = run_worker(
        root,
        "convert",
        output=str(output),
        sha256=detection["sha256"],
        settings={"fps": 10, "task": "Pick up block"},
    )
    assert result["episodes"] == 2 and result["frames"] == 6
    verify = """import pyarrow.parquet as p,sys
r=p.read_table(sys.argv[1]).to_pylist()
assert r[0]["action"]==[.25,.25]
assert r[0]["observation.state"]==[.5,.5]
"""
    subprocess.run(
        [str(reader_python()), "-I", "-c", verify, str(output / "data/chunk-000/file-000.parquet")],
        check=True,
    )


def test_detection_ambiguous_formats_and_changed_source(tmp_path):
    detected = run_worker(tmp_path, "demo")
    (tmp_path / "frames.jsonl").write_text((tmp_path / "frames.jsonl").read_text() + "\n")
    from vla_platform.datasets import import_worker

    result = subprocess.run(
        [str(reader_python()), "-I", str(Path(import_worker.__file__))],
        input=json.dumps(
            {
                "operation": "convert",
                "root": str(tmp_path),
                "output": str(tmp_path.parent / "never-published"),
                "sha256": detected["sha256"],
                "settings": {},
            }
        ).encode(),
        capture_output=True,
    )
    assert result.returncode and "Source bytes changed" in result.stdout.decode()


def test_zip_traversal_rejected_and_uploaded_bytes_preserved(tmp_path):
    async def check():
        library = DatasetLibrary(tmp_path.resolve())
        entry = library.begin("one", "unsafe archive")
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr("../escape.txt", "untrusted")

        async def chunks():
            yield payload.getvalue()

        await library.upload(entry["id"], "dataset.zip", chunks())
        with pytest.raises(LibraryError, match="relative"):
            await library.finalize(entry["id"])
        assert (library.directory(entry["id"]) / "source/dataset.zip").is_file()
        assert not (tmp_path / "escape.txt").exists()
        await library.close()

    asyncio.run(check())


def test_managed_intake_without_host_root_and_project_boundary(tmp_path):
    settings = Settings(data_dir=tmp_path.resolve())
    with TestClient(create_app(settings)) as client:
        first = client.post("/api/v1/projects", json={"name": "First"}).json()
        second = client.post("/api/v1/projects", json={"name": "Second"}).json()
        entry = client.post(f"/api/v1/projects/{first['id']}/datasets/example").json()
        for _ in range(200):
            value = client.get(f"/api/v1/datasets/{entry['id']}").json()
            if value["status"] != "converting":
                break
            time.sleep(0.05)
        assert value["status"] == "ready", value
        payload = {"source": "local", "library_id": entry["id"]}
        wrong = client.post(f"/api/v1/projects/{second['id']}/intakes", json=payload)
        assert wrong.status_code in (404, 422), wrong.text
        response = client.post(
            f"/api/v1/projects/{first['id']}/intakes",
            json=payload,
            headers={"Idempotency-Key": "library-intake-one"},
        )
        assert response.status_code == 202, response.text
        job = response.json()
        for _ in range(100):
            job = client.get(f"/api/v1/jobs/{job['id']}").json()
            if job["status"] not in ("queued", "running"):
                break
            time.sleep(0.05)
        assert job["status"] == "succeeded", job
        assert job["request"]["library_id"] == entry["id"]
        assert job["result"]["total_frames"] == 24
        history = client.get(f"/api/v1/datasets?project_id={first['id']}").json()
        assert len(history) == 1 and history[0]["job_id"] == job["id"]
        assert client.get(f"/api/v1/datasets?project_id={second['id']}").json() == []
        download = client.get(f"/api/v1/datasets/{entry['id']}/download")
        assert download.status_code == 200
        assert b"PK" == download.content[:2]


def test_hdf5_external_link_is_not_followed(tmp_path):
    script = """import h5py,sys
f=h5py.File(sys.argv[1],"w")
f["secret"]=h5py.ExternalLink("/etc/passwd","/")
f.close()
"""
    subprocess.run(
        [str(reader_python()), "-I", "-c", script, str(tmp_path / "dataset.h5")], check=True
    )
    from vla_platform.datasets import import_worker

    result = subprocess.run(
        [str(reader_python()), "-I", str(Path(import_worker.__file__))],
        input=json.dumps({"operation": "detect", "root": str(tmp_path)}).encode(),
        capture_output=True,
    )
    assert result.returncode and "external and soft links" in result.stdout.decode()
