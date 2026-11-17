"""The isolated compact.py worker preserves robot samples and creates usable camera videos."""

import asyncio
import hashlib
import json
import subprocess

import httpx
from vla_platform.datasets.library import DatasetLibrary
from vla_platform.datasets.local_preview import reader_python
from vla_platform.robot_lab_bridge import RobotLabBridge


def embedded_recording(root):
    script = r"""
import io, json, sys
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
root=Path(sys.argv[1]); (root/"data/chunk-000").mkdir(parents=True)
(root/"meta/episodes/chunk-000").mkdir(parents=True)
episodes=[]
for ep, length in enumerate((1, 3)):
    rows=[]
    for frame in range(length):
        image=io.BytesIO()
        Image.new("RGB",(32,24),(40+ep*30,frame*20,100)).save(image,format="PNG")
        rows.append({"episode_index":ep,"frame_index":frame,"index":ep+frame,
                     "timestamp":frame/6,"task_index":0,"action":[.125,frame+.25],
                     "observation.state":[ep+.5,frame+.75],
                     "observation.images.front":{"bytes":image.getvalue(),"path":None}})
    pq.write_table(pa.Table.from_pylist(rows),root/f"data/chunk-000/file-{ep:03}.parquet")
    episodes.append({"episode_index":ep,"length":length,"dataset_from_index":ep,
                     "dataset_to_index":ep+length,"data/chunk_index":0,"data/file_index":ep,
                     "meta/episodes/chunk_index":0,"meta/episodes/file_index":0})
pq.write_table(pa.Table.from_pylist(episodes),root/"meta/episodes/chunk-000/file-000.parquet")
pq.write_table(pa.Table.from_pylist([{"task_index":0,"task":"pick up"}]),root/"meta/tasks.parquet")
info={"codebase_version":"v3.0","fps":6,"total_frames":4,"total_episodes":2,
      "data_path":"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
      "video_path":None,"features":{"action":{"dtype":"float32","shape":[2]},
      "observation.state":{"dtype":"float32","shape":[2]},
      "observation.images.front":{"dtype":"image","shape":[24,32,3]}}}
(root/"meta/info.json").write_text(json.dumps(info))
(root/"openjensen").mkdir(); (root/"openjensen/calibration.json").write_text('{"profile":"saved"}')
"""
    subprocess.run([str(reader_python()), "-I", "-c", script, str(root)], check=True)


def test_large_recording_manifest_has_no_two_gib_ceiling(tmp_path):
    size = 3 * 1024**3
    manifest = {
        "id": "large",
        "total_bytes": size,
        "files": [{"path": "meta/info.json", "size": size, "sha256": "a" * 64}],
    }
    bridge = RobotLabBridge(tmp_path, client=httpx.AsyncClient())

    async def request(*args):
        return manifest

    bridge.request = request

    async def check():
        _, inventory, _ = await bridge.manifest("large")
        assert inventory[0]["size"] == size
        await bridge.close()

    asyncio.run(check())


def test_compact_embedded_recording_preserves_source_and_numeric_samples(tmp_path):
    source = tmp_path / "original"
    embedded_recording(source)
    original = {
        p.relative_to(source): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source.rglob("*")
        if p.is_file()
    }

    async def check():
        library = DatasetLibrary(tmp_path / "workspace")
        entry = library.begin("project-one", "duck")
        for path in sorted(source.rglob("*")):
            if path.is_file():

                async def stream(path=path):
                    yield path.read_bytes()

                await library.upload(entry["id"], path.relative_to(source).as_posix(), stream())
        entry = await library.finalize(entry["id"])
        compact = await library.start_compaction(entry["id"])
        assert compact["status"] == "converting"
        assert (await library.start_compaction(entry["id"]))["id"] == compact["id"]
        await library.tasks[compact["id"]]
        result = library.get(compact["id"])
        assert result["status"] == "ready", result
        output = library.resolve(result["id"], "project-one")
        assert (output / "openjensen/calibration.json").read_bytes() == (
            source / "openjensen/calibration.json"
        ).read_bytes()
        assert result["detection"]["frames"] == 4 and result["detection"]["episodes"] == 2
        assert result["compaction"]["source_bytes"] == sum(
            p.stat().st_size for p in source.rglob("*") if p.is_file()
        )
        verify = r"""
import json,sys
from pathlib import Path
import pyarrow.parquet as pq
source,output=map(Path,sys.argv[1:])
for p in (source/"data").rglob("*.parquet"):
    a=pq.read_table(p)
    columns=[k for k in a.column_names if not k.startswith("observation.images.")]
    b=pq.read_table(output/p.relative_to(source))
    assert a.select(columns).equals(b)
info=json.loads((output/"meta/info.json").read_text())
assert info["features"]["observation.images.front"]["dtype"]=="video" and info["fps"]==6
"""
        subprocess.run(
            [str(reader_python()), "-I", "-c", verify, str(source), str(output)], check=True
        )
        for video in output.rglob("*.mp4"):
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_read_frames,width,height",
                    "-of",
                    "json",
                    str(video),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            stream = json.loads(probe.stdout)["streams"][0]
            assert int(stream["nb_read_frames"]) in (1, 3)
            assert (stream["width"], stream["height"]) == (32, 24)
        assert (await library.start_compaction(entry["id"]))["id"] == compact["id"]
        await library.close()

    asyncio.run(check())
    assert {
        p.relative_to(source): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source.rglob("*")
        if p.is_file()
    } == original


def test_browser_can_compact_and_import_a_recording_in_one_request(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from test_robot_lab_api import bridge, connection, manifest
    from vla_platform.dataset_library_api import router as library_router
    from vla_platform.robot_lab_api import router as robot_router

    source = tmp_path / "recording"
    embedded_recording(source)
    files = {
        p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }
    connection(tmp_path)

    def handler(request):
        if request.url.path == "/v1/calibration/bundle":
            return httpx.Response(
                200,
                json={
                    "format": "openjensen.calibration.bundle",
                    "version": 1,
                    "profiles": [{"saved": "calibration"}],
                },
            )
        if request.url.path.endswith("/manifest"):
            return httpx.Response(200, json=manifest(files))
        return httpx.Response(200, content=files[request.url.params["path"]])

    class Projects:
        async def get(self, ident):
            return {"id": ident} if ident == "project-one" else None

    companion = bridge(tmp_path, handler)
    app = FastAPI()
    app.include_router(robot_router)
    app.include_router(library_router)
    app.state.robot_lab = companion
    app.state.dataset_library = DatasetLibrary(tmp_path)
    app.state.projects = Projects()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/projects/project-one/robot-lab/datasets/pick-block/import?compact=true"
        )
        assert response.status_code == 201, response.text
        compact = response.json()
        assert compact["status"] == "converting" and compact["compacted_from"]

        async def finish():
            await app.state.dataset_library.tasks[compact["id"]]

        client.portal.call(finish)
        result = client.get("/api/v1/datasets/" + compact["id"]).json()
        assert result["status"] == "ready" and result["compaction"]["frames"] == 4
        folder = app.state.dataset_library.resolve(result["id"], "project-one")
        bundle = json.loads((folder / "openjensen/calibration-profiles.json").read_bytes())
        assert bundle["profiles"] == [{"saved": "calibration"}]
        assert client.get("/api/v1/datasets/" + compact["compacted_from"]).json()["can_compact"]
        client.portal.call(app.state.dataset_library.close)
        client.portal.call(companion.close)
