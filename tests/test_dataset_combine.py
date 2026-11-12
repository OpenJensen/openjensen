"""Fifty recordings of one episode each must add up to one trainable dataset.

An agent-supervised collection run writes exactly one episode into its own dataset, and
a single episode cannot be trained on. These tests record real datasets with the Robot
Lab writer, combine them, and hand the result to the complete training validator.
"""

import json
import os
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_dataset_snapshots import build_dataset
from vla_platform.api import create_app
from vla_platform.datasets import import_worker
from vla_platform.datasets.local_preview import reader_python
from vla_platform.datasets.snapshots import create_snapshot
from vla_platform.settings import Settings

pytestmark = pytest.mark.skipif(os.name == "nt", reason="Secure snapshots require POSIX")
ROBOT_LAB = Path(__file__).resolve().parents[1] / "workers/robot_lab"
RECORD = r"""
import sys
sys.path.insert(0, sys.argv[1])
from pathlib import Path
import numpy as np
from dataset import DatasetWriter
from hardware import JOINTS

root, count, fps = Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
tasks, cameras = sys.argv[5].split("|"), sys.argv[6].split(",")
shapes = {"observation.images." + name: (24, 32, 3) for name in cameras}
for run in range(count):
    writer = DatasetWriter(root, f"agent-run{run:02d}", tasks[run % len(tasks)], fps, shapes)
    writer.start_episode()
    for index in range(3 + run % 3):  # Episodes of different lengths, like real ones.
        positions = {joint: float(run * 100 + index + n) for n, joint in enumerate(JOINTS)}
        images = {
            key: np.full(shape, [index * 9 % 255, run * 5 % 255, 120], dtype=np.uint8)
            for key, shape in shapes.items()
        }
        writer.add_frame(positions, positions, images, captured_at=100 + index / fps)
    writer.save_episode()
    writer.finalize()
"""


def record(root, count, *, fps=10, tasks="Pick up the pen", cameras="wrist,side"):
    """One finalized dataset per run, each holding a single episode."""
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(reader_python()), "-I", "-c", RECORD, str(ROBOT_LAB), str(root)]
        + [str(count), str(fps), tasks, cameras],
        check=True,
        timeout=60,
    )
    return sorted(path for path in root.iterdir() if path.is_dir())


def worker(payload):
    response = subprocess.run(
        [str(reader_python()), "-I", "-B", str(Path(import_worker.__file__))],
        input=json.dumps(payload).encode(),
        capture_output=True,
        timeout=60,
    )
    return response.returncode, json.loads(response.stdout)


def table(path):
    script = "import json,sys,pyarrow.parquet as pq\n" + (
        "rows = pq.read_table(sys.argv[1]).to_pylist()\n"
        "print(json.dumps([{k: v for k, v in r.items() if 'images' not in k} for r in rows]))"
    )
    output = subprocess.check_output([str(reader_python()), "-I", "-c", script, str(path)])
    return json.loads(output)


def test_one_episode_recordings_combine_into_a_dataset_the_trainer_accepts(tmp_path):
    recordings = record(tmp_path / "recorded", 5)
    for recording in recordings:
        assert json.loads((recording / "meta/info.json").read_text())["total_episodes"] == 1
    output = tmp_path / "combined"
    code, result = worker(
        {"operation": "merge", "roots": [str(r) for r in recordings], "output": str(output)}
    )
    assert code == 0, result
    assert result == {
        "episodes": 5,
        "frames": 3 + 4 + 5 + 3 + 4,
        "fps": 10,
        "tasks": ["Pick up the pen"],
        "sources": 5,
    }
    # The complete training validator: every frame, index, timestamp and statistic.
    snapshot = create_snapshot(tmp_path, output, tmp_path / "snapshots")
    assert snapshot["total_episodes"] == 5 and snapshot["total_frames"] == 19
    # Episodes keep their order and their own frames; only the numbering changed.
    episodes = table(output / "meta/episodes/chunk-000/file-000.parquet")
    assert [(e["dataset_from_index"], e["dataset_to_index"]) for e in episodes] == [
        (0, 3),
        (3, 7),
        (7, 12),
        (12, 15),
        (15, 19),
    ]
    third = table(output / "data/chunk-000/file-002.parquet")
    assert [row["index"] for row in third] == [7, 8, 9, 10, 11]
    assert {row["episode_index"] for row in third} == {2}
    assert [row["frame_index"] for row in third] == [0, 1, 2, 3, 4]
    assert third[0]["observation.state"] == [200.0, 201.0, 202.0, 203.0, 204.0, 205.0]
    # The originals are untouched.
    assert json.loads((recordings[2] / "meta/info.json").read_text())["total_episodes"] == 1
    stats = json.loads((output / "meta/stats.json").read_text())
    assert stats["action"]["count"] == [19]
    assert stats["action"]["min"][0] == 0.0 and stats["action"]["max"][0] == 403.0


def test_different_instructions_stay_attached_to_their_own_episodes(tmp_path):
    recordings = record(tmp_path / "recorded", 4, tasks="Pick up the pen|Put the pen down")
    output = tmp_path / "combined"
    code, result = worker(
        {"operation": "merge", "roots": [str(r) for r in recordings], "output": str(output)}
    )
    assert code == 0, result
    assert result["tasks"] == ["Pick up the pen", "Put the pen down"]
    create_snapshot(tmp_path, output, tmp_path / "snapshots")
    episodes = table(output / "meta/episodes/chunk-000/file-000.parquet")
    assert [e["tasks"] for e in episodes] == [
        ["Pick up the pen"],
        ["Put the pen down"],
        ["Pick up the pen"],
        ["Put the pen down"],
    ]
    second = table(output / "data/chunk-000/file-001.parquet")
    assert {row["task_index"] for row in second} == {1}


def test_datasets_that_do_not_match_are_refused_with_the_reason(tmp_path):
    base = record(tmp_path / "base", 1)
    slower = record(tmp_path / "slower", 1, fps=5)
    other_camera = record(tmp_path / "front", 1, cameras="front,side")
    for extra, reason in ((slower, "5 fps"), (other_camera, "different cameras")):
        output = tmp_path / ("out-" + extra[0].parent.name)
        code, result = worker(
            {"operation": "merge", "roots": [str(base[0]), str(extra[0])], "output": str(output)}
        )
        assert code == 1 and reason in result["error"], result
        assert not (output / "meta/info.json").exists()
    code, result = worker(
        {"operation": "merge", "roots": [str(base[0])], "output": str(tmp_path / "alone")}
    )
    assert code == 1 and "between 2 and 1,000" in result["error"]


def test_video_datasets_combine_and_keep_each_camera_file(tmp_path):
    first = build_dataset(tmp_path / "first", episodes=2)
    second = build_dataset(tmp_path / "second", episodes=3)
    output = tmp_path / "combined"
    code, result = worker(
        {"operation": "merge", "roots": [str(first), str(second)], "output": str(output)}
    )
    assert code == 0, result
    assert result["episodes"] == 5 and result["frames"] == 15
    snapshot = create_snapshot(tmp_path, output, tmp_path / "snapshots")
    assert snapshot["total_episodes"] == 5
    videos = sorted(path.name for path in (output / "videos").rglob("*.mp4"))
    assert videos == ["file-000.mp4", "file-001.mp4"]


def wait(client, ident):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        job = client.get("/api/v1/jobs/" + ident).json()
        if job["status"] not in {"queued", "running"}:
            return job
        time.sleep(0.05)
    raise AssertionError("Job did not finish")


def save(client, project, name, source):
    entry = client.post(f"/api/v1/projects/{project}/dataset-uploads", json={"name": name}).json()
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        uploaded = client.put(
            f"/api/v1/dataset-uploads/{entry['id']}/file",
            params={"path": path.relative_to(source).as_posix()},
            content=path.read_bytes(),
        )
        assert uploaded.status_code == 200, uploaded.text
    saved = client.post(f"/api/v1/dataset-uploads/{entry['id']}/finish").json()
    assert saved["status"] == "ready", saved
    return saved


def test_saved_recordings_are_untrainable_alone_and_trainable_once_combined(tmp_path):
    """The whole journey a person takes, through the same requests the page sends."""
    recordings = record(tmp_path / "recorded", 4)
    settings = Settings(data_dir=tmp_path / "workspace", local_root=tmp_path)
    with TestClient(create_app(settings)) as client:
        pid = client.post("/api/v1/projects", json={"name": "Pen"}).json()["id"]
        saved = [save(client, pid, r.name, r) for r in recordings]
        intake = f"/api/v1/projects/{pid}/intakes"

        def prepare(dataset):
            body = {
                "source": "local",
                "library_id": dataset["id"],
                "revision": "main",
                "snapshot_for_training": True,
            }
            return wait(client, client.post(intake, json=body).json()["id"])

        alone = prepare(saved[0])
        assert alone["status"] == "succeeded" and "has 1" in alone["result"]["training_blocker"]

        combined = client.post(
            f"/api/v1/projects/{pid}/datasets/combine",
            json={"name": "Pen pickup, all runs", "dataset_ids": [item["id"] for item in saved]},
        )
        assert combined.status_code == 201, combined.text
        dataset = combined.json()
        assert dataset["status"] == "ready" and dataset["name"] == "Pen pickup, all runs"
        assert dataset["detection"]["episodes"] == 4
        assert [item["name"] for item in dataset["combined_from"]] == [r.name for r in recordings]
        ready = prepare(dataset)
        assert ready["status"] == "succeeded", ready
        assert "training_blocker" not in ready["result"]
        assert ready["result"]["snapshot"]["total_episodes"] == 4
        # The four originals are still there, unchanged, beside the new dataset.
        listed = client.get(f"/api/v1/datasets?project_id={pid}").json()
        names = sorted(row["name"] for row in listed if not row["id"].startswith("inspection:"))
        assert names == sorted(["Pen pickup, all runs", *[r.name for r in recordings]])
        # Combining the same datasets again does not pile up a duplicate.
        again = client.post(
            f"/api/v1/projects/{pid}/datasets/combine",
            json={"name": "Again", "dataset_ids": [item["id"] for item in saved]},
        )
        assert again.status_code == 201 and again.json()["id"] == dataset["id"]

        refused = client.post(
            f"/api/v1/projects/{pid}/datasets/combine",
            json={"name": "One", "dataset_ids": [saved[0]["id"]]},
        )
        assert refused.status_code == 422


def test_a_refused_combination_leaves_no_dataset_and_no_unfinished_job(tmp_path):
    fast = record(tmp_path / "fast", 1)
    slow = record(tmp_path / "slow", 1, fps=5)
    settings = Settings(data_dir=tmp_path / "workspace", local_root=tmp_path)
    with TestClient(create_app(settings)) as client:
        pid = client.post("/api/v1/projects", json={"name": "Mixed"}).json()["id"]
        saved = [save(client, pid, "ten fps", fast[0]), save(client, pid, "five fps", slow[0])]
        refused = client.post(
            f"/api/v1/projects/{pid}/datasets/combine",
            json={"name": "Mixed rates", "dataset_ids": [item["id"] for item in saved]},
        )
        assert refused.status_code == 422
        assert "5 fps" in refused.json()["detail"] and "10 fps" in refused.json()["detail"]
        listed = client.get(f"/api/v1/datasets?project_id={pid}").json()
        assert sorted(row["name"] for row in listed) == ["five fps", "ten fps"]
        jobs = settings.data_dir / "dataset-jobs"
        states = [json.loads(path.read_text())["status"] for path in jobs.glob("*.json")]
        assert sorted(states) == ["failed", "succeeded", "succeeded"]
        other = client.post("/api/v1/projects", json={"name": "Other"}).json()["id"]
        foreign = client.post(
            f"/api/v1/projects/{other}/datasets/combine",
            json={"name": "Not mine", "dataset_ids": [item["id"] for item in saved[:2]]},
        )
        assert foreign.status_code in {404, 422}
