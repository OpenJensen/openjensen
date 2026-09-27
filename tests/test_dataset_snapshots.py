"""Real Parquet and MP4 coverage; no Torch or model download in the application."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from vla_platform.datasets import snapshots
from vla_platform.datasets.local_preview import reader_python
from vla_platform.datasets.snapshots import (
    SnapshotError,
    SnapshotLimits,
    create_snapshot,
    resolve_snapshot,
    stage_snapshot,
    verify_snapshot,
)

pytestmark = pytest.mark.skipif(os.name == "nt", reason="Secure local snapshot opens require POSIX")
CAMERA = "observation.images.front"


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    script = r"""
import json, sys
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

root = Path(sys.argv[1])
camera = "observation.images.front"
for name in (
    "meta/episodes/chunk-000",
    "data/chunk-000",
    "videos/" + camera + "/chunk-000",
):
    (root / name).mkdir(parents=True)
features = {
    key: {"dtype": "int64", "shape": [1], "names": None}
    for key in ("index", "episode_index", "frame_index", "task_index")
}
features["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
for key in ("action", "observation.state"):
    features[key] = {"dtype": "float32", "shape": [2], "names": ["x", "y"]}
features[camera] = {
    "dtype": "video",
    "shape": [16, 16, 3],
    "names": ["height", "width", "channels"],
    "info": {
        "video.fps": 10,
        "video.height": 16,
        "video.width": 16,
        "video.channels": 3,
    },
}
info = {
    "codebase_version": "v3.0",
    "robot_type": "synthetic_fixture",
    "total_episodes": 4,
    "total_frames": 12,
    "total_tasks": 1,
    "fps": 10,
    "chunks_size": 1000,
    "data_files_size_in_mb": 100,
    "video_files_size_in_mb": 200,
    "splits": {"train": "0:4"},
    "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    "features": features,
}
(root / "meta/info.json").write_text(json.dumps(info))
stats = {}
for key in ("action", "observation.state", camera):
    vector = [0.0, 0.0] if key != camera else [[[0.0]], [[0.0]], [[0.0]]]
    one = [1.0, 1.0] if key != camera else [[[1.0]], [[1.0]], [[1.0]]]
    stats[key] = {"min": vector, "max": one, "mean": vector, "std": one, "count": [12]}
(root / "meta/stats.json").write_text(json.dumps(stats))
pq.write_table(
    pa.Table.from_pylist([{"task_index": 0, "task": "Synthetic fixture only"}]),
    root / "meta/tasks.parquet",
)
episodes = []
rows = []
for ep in range(4):
    episodes.append(
        {
            "episode_index": ep,
            "tasks": ["Synthetic fixture only"],
            "length": 3,
            "dataset_from_index": ep * 3,
            "dataset_to_index": (ep + 1) * 3,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "videos/" + camera + "/chunk_index": 0,
            "videos/" + camera + "/file_index": 0,
            "videos/" + camera + "/from_timestamp": ep * 0.3,
            "videos/" + camera + "/to_timestamp": (ep + 1) * 0.3,
        }
    )
    for frame in range(3):
        rows.append(
            {
                "index": ep * 3 + frame,
                "episode_index": ep,
                "frame_index": frame,
                "timestamp": frame / 10,
                "task_index": 0,
                "action": [0.0, 1.0],
                "observation.state": [1.0, 0.0],
            }
        )
pq.write_table(
    pa.Table.from_pylist(episodes), root / "meta/episodes/chunk-000/file-000.parquet"
)
schema = pa.schema([
    pa.field(key, pa.list_(pa.float32()) if key in ("action", "observation.state") else
             pa.float32() if key == "timestamp" else pa.int64()) for key in rows[0]
])
pq.write_table(pa.Table.from_pylist(rows, schema=schema), root / "data/chunk-000/file-000.parquet")
"""
    subprocess.run([str(reader_python()), "-I", "-c", script, str(root)], check=True, timeout=20)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=16x16:r=10",
            "-frames:v",
            "12",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(root / f"videos/{CAMERA}/chunk-000/file-000.mp4"),
        ],
        check=True,
        timeout=20,
        capture_output=True,
    )
    return root


def hashes(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def make(dataset, tmp_path, **kwargs):
    return create_snapshot(tmp_path, dataset, tmp_path / "snapshots", **kwargs)


def mutate_table(dataset, relative, expression):
    script = (
        "import pyarrow as pa, pyarrow.parquet as pq, sys\n"
        "from pathlib import Path\np=Path(sys.argv[1])\n"
        "table=pq.read_table(p); rows=table.to_pylist()\n"
        + expression
        + "\npq.write_table(pa.Table.from_pylist(rows, schema=table.schema),p)"
    )
    subprocess.run(
        [str(reader_python()), "-I", "-c", script, str(dataset / relative)], check=True, timeout=20
    )


def test_complete_snapshot_stages_exact_immutable_bytes(dataset, tmp_path):
    before = hashes(dataset)
    value = make(dataset, tmp_path)
    assert value["total_frames"] == 12
    assert value["lineage_validated"] is False
    assert "ancestry is unknown" in " ".join(value["warnings"])
    root = resolve_snapshot(tmp_path / "snapshots", value)
    assert make(dataset, tmp_path) == value
    staged = stage_snapshot(root, tmp_path / "job-dataset", value["manifest_sha256"])
    assert hashes(root) == hashes(staged)
    assert hashes(dataset) == before
    (dataset / "meta/stats.json").write_text("{}")
    assert verify_snapshot(staged, value["manifest_sha256"])["total_frames"] == 12
    with pytest.raises(FileExistsError):
        stage_snapshot(root, staged, value["manifest_sha256"])


@pytest.mark.parametrize(
    "change,match",
    [
        ("missing_video", "camera media"),
        ("truncated_video", "validation failed"),
        ("extra_video", "unreferenced camera"),
        ("unfinished", "unfinished"),
        ("extra_rows", "unfinalized"),
        ("missing_rows", "Missing finalized"),
        ("nan_action", "Invalid action"),
        ("bad_time", "timestamps"),
        ("overlap", "overlap"),
        ("video_gap", "video range"),
        ("missing_stats", "normalization statistics"),
        ("bad_tasks", "Task indices"),
        ("incomplete_lineage", "every episode"),
        ("symlink", "symlink"),
        ("fifo", "special file"),
    ],
)
def test_invalid_dataset_never_publishes(dataset, tmp_path, change, match):
    video = dataset / f"videos/{CAMERA}/chunk-000/file-000.mp4"
    table = "data/chunk-000/file-000.parquet"
    episode_table = "meta/episodes/chunk-000/file-000.parquet"
    if change == "missing_video":
        video.unlink()
    elif change == "truncated_video":
        video.write_bytes(video.read_bytes()[:100])
    elif change == "extra_video":
        shutil.copyfile(video, video.with_name("file-001.mp4"))
    elif change == "unfinished":
        (dataset / "recording.tmp").write_text("unfinished")
    elif change == "extra_rows":
        mutate_table(dataset, table, "rows.append(dict(rows[-1]))")
    elif change == "missing_rows":
        mutate_table(dataset, table, "rows.pop()")
    elif change == "nan_action":
        mutate_table(dataset, table, "rows[0]['action'][0]=float('nan')")
    elif change == "bad_time":
        mutate_table(dataset, table, "rows[1]['timestamp']=0.4")
    elif change == "overlap":
        mutate_table(dataset, episode_table, "rows[1]['dataset_from_index']=2")
    elif change == "video_gap":
        mutate_table(dataset, episode_table, f"rows[1]['videos/{CAMERA}/from_timestamp']=0.4")
    elif change == "missing_stats":
        (dataset / "meta/stats.json").write_text("{}")
    elif change == "bad_tasks":
        mutate_table(dataset, "meta/tasks.parquet", "rows[0]['task_index']=1")
    elif change == "incomplete_lineage":
        (dataset / "meta/firebird-lineage.json").write_text(
            json.dumps({"schema_version": 1, "episodes": []})
        )
    elif change == "symlink":
        (dataset / "link").symlink_to(video)
    elif change == "fifo":
        os.mkfifo(dataset / "pipe")
    with pytest.raises((SnapshotError, ValueError), match=match):
        make(dataset, tmp_path)
    assert not list((tmp_path / "snapshots").iterdir())


def test_declared_lineage_is_bound_to_snapshot(dataset, tmp_path):
    episodes = [
        {
            "episode_index": i,
            "origin": "recorded" if i % 2 == 0 else "augmented",
            "lineage_group": f"root:{i // 2}",
        }
        for i in range(4)
    ]
    (dataset / "meta/firebird-lineage.json").write_text(
        json.dumps({"schema_version": 1, "episodes": episodes})
    )
    value = make(dataset, tmp_path)
    assert value["lineage_validated"] is True
    assert (
        verify_snapshot(resolve_snapshot(tmp_path / "snapshots", value), value["manifest_sha256"])[
            "lineage"
        ]
        == episodes
    )


def test_mutation_during_copy_rejected(dataset, tmp_path, monkeypatch):
    original = snapshots._copy

    def mutate(*args):
        result = original(*args)
        (dataset / "meta/stats.json").write_text("{}")
        return result

    monkeypatch.setattr(snapshots, "_copy", mutate)
    with pytest.raises(SnapshotError, match="changed during snapshot"):
        make(dataset, tmp_path)
    assert not list((tmp_path / "snapshots").iterdir())


def test_escape_and_symlink_ancestors_rejected(dataset, tmp_path):
    with pytest.raises(SnapshotError, match="outside"):
        create_snapshot(tmp_path / "allowed", dataset, tmp_path / "snapshots")
    alias = tmp_path / "alias"
    alias.symlink_to(dataset, target_is_directory=True)
    with pytest.raises(SnapshotError, match="symlink"):
        make(alias, tmp_path)


def test_staging_rejects_mutated_snapshot(dataset, tmp_path):
    value = make(dataset, tmp_path)
    root = resolve_snapshot(tmp_path / "snapshots", value)
    path = root / "meta/stats.json"
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b"1.0", b"2.0", 1))
    with pytest.raises(SnapshotError, match="hash mismatch"):
        stage_snapshot(root, tmp_path / "stage", value["manifest_sha256"])
    assert not (tmp_path / "stage").exists()


def test_budget_rejection_before_reader(dataset, tmp_path):
    with pytest.raises(SnapshotError, match="budget"):
        make(dataset, tmp_path, limits=SnapshotLimits(max_bytes=1))


def test_cloud_bundle_contains_only_verified_selected_snapshot(dataset, tmp_path, monkeypatch):
    from vla_platform.lifecycle import sky_runner
    from vla_platform.lifecycle.native_profiles import NATIVE_PROFILES

    value = make(dataset, tmp_path)
    root = resolve_snapshot(tmp_path / "snapshots", value)
    profile = NATIVE_PROFILES["act"]
    payload = {
        "schema_version": 1,
        "operation": "policy.finetune",
        "job_id": "local-training",
        "parameters": {
            "timeout_seconds": 300,
            "training_method": "full",
            "training": {
                "model_id": profile["model_id"],
                "model_revision": profile["model_revision"],
                "steps": 2,
            },
        },
        "runtime": {},
        "dataset": {"source": "local"},
        "dataset_snapshot": {
            "path": str(root),
            "manifest_sha256": value["manifest_sha256"],
            "id": value["id"],
        },
        "artifact": None,
        "source": None,
    }
    target = {
        "project_id": "robotics-fixture",
        "region": "us-central1",
        "accelerator": "L4",
        "gpu_count": 1,
        "disk_size_gb": 100,
        "idle_minutes": 10,
        "sky_api_endpoint": "http://127.0.0.1:46580",
        "workspace": "fixture",
    }
    monkeypatch.setattr(sky_runner, "executable", lambda: "/fixture/sky")
    stage = tmp_path / "operation"
    sky_runner.prepare(payload, stage, target)
    bundle = stage / "sky-bundle"
    submitted = json.loads((bundle / "request.json").read_text())
    assert submitted["dataset_snapshot"]["path"] == "inputs/dataset"
    assert hashes(root) == hashes(bundle / "inputs/dataset")
    assert str(root) not in (bundle / "request.json").read_text()
    assert not (bundle / "inputs/dataset/source").exists()
    (root / "meta/info.json").write_text("tampered")
    with pytest.raises(SnapshotError, match="size mismatch"):
        sky_runner.prepare(payload, tmp_path / "second-operation", target)


def test_atomic_publication_cannot_replace_an_existing_directory(tmp_path):
    source, destination = tmp_path / "staging", tmp_path / "existing"
    source.mkdir()
    (source / "verified").write_text("new")
    destination.mkdir()
    with pytest.raises(FileExistsError):
        snapshots._publish(source, destination)
    assert source.is_dir()
    assert list(destination.iterdir()) == []


def test_validation_deadline_reaps_owned_decoder(tmp_path, monkeypatch):
    import sys
    import time

    fake = tmp_path / "snapshot_reader.py"
    child_pid = tmp_path / "child.pid"
    fake.write_text(
        "import subprocess,sys,time,pathlib\n"
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])\n"
        f"pathlib.Path({str(child_pid)!r}).write_text(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    monkeypatch.setattr(snapshots, "__file__", str(tmp_path / "snapshots.py"))
    started = time.monotonic()
    with pytest.raises(SnapshotError, match="budget"):
        snapshots._validate(tmp_path, SnapshotLimits(timeout_seconds=1), sys.executable)
    assert time.monotonic() - started < 5
    pid = int(child_pid.read_text())
    result = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    assert result.returncode != 0 or result.stdout.strip().startswith("Z")


def test_original_mutation_during_validation_rejected(dataset, tmp_path, monkeypatch):
    original = snapshots._validate

    def mutate(*args):
        value = original(*args)
        (dataset / "meta/stats.json").write_text("{}")
        return value

    monkeypatch.setattr(snapshots, "_validate", mutate)
    with pytest.raises(SnapshotError, match="changed during snapshot"):
        make(dataset, tmp_path)
    assert list((tmp_path / "snapshots").iterdir()) == []


def test_cancellation_during_reader_creation_reaps_process(tmp_path, monkeypatch):
    import signal
    import sys

    fake = tmp_path / "snapshot_reader.py"
    fake.write_text("import time; time.sleep(60)\n")
    monkeypatch.setattr(snapshots, "__file__", str(tmp_path / "snapshots.py"))
    original = snapshots.subprocess.Popen
    created = []
    previous = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    def create_then_interrupt(*args, **kwargs):
        process = original(*args, **kwargs)
        created.append(process)
        os.kill(os.getpid(), signal.SIGTERM)
        return process

    signal.signal(signal.SIGTERM, terminate)
    monkeypatch.setattr(snapshots.subprocess, "Popen", create_then_interrupt)
    try:
        with pytest.raises(SystemExit):
            snapshots._validate(tmp_path, SnapshotLimits(timeout_seconds=1), sys.executable)
        assert signal.getsignal(signal.SIGTERM) is terminate
        assert created[0].poll() is not None
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_demonstration_provenance_cannot_bypass_lineage(dataset, tmp_path):
    (dataset / "meta/firebird-demonstrations.json").write_text('{"schema_version":1}')
    with pytest.raises(SnapshotError, match="requires complete episode lineage"):
        make(dataset, tmp_path)


def test_core_and_native_group_split_implementations_remain_identical():
    import ast
    import inspect

    root = Path(__file__).parents[1]
    worker = ast.parse(
        (root / "workers/smolvla_qlora/src/firebird_vla/local_dataset.py").read_text()
    )
    definition = next(
        node
        for node in worker.body
        if isinstance(node, ast.FunctionDef) and node.name == "split_lineage"
    )
    assert ast.dump(definition) == ast.dump(
        ast.parse(inspect.getsource(snapshots.split_lineage)).body[0]
    )


def test_physical_parquet_dtype_must_match_metadata(dataset, tmp_path):
    path = dataset / "data/chunk-000/file-000.parquet"
    script = (
        "import pyarrow as pa, pyarrow.parquet as pq,sys\n"
        "table=pq.read_table(sys.argv[1]); rows=table.to_pylist()\n"
        "pq.write_table(pa.Table.from_pylist(rows),sys.argv[1])\n"
    )
    subprocess.run([str(reader_python()), "-I", "-c", script, str(path)], check=True, timeout=20)
    with pytest.raises(SnapshotError, match="storage dtype differs"):
        make(dataset, tmp_path)


def test_variable_camera_timing_rejected_even_with_correct_frame_count(dataset, tmp_path):
    video = dataset / f"videos/{CAMERA}/chunk-000/file-000.mp4"
    changed = tmp_path / "bad-timing.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
            r"setpts=PTS+gte(N\,6)/(10*TB)",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            str(changed),
        ],
        check=True,
        timeout=20,
        capture_output=True,
    )
    video.write_bytes(changed.read_bytes())
    with pytest.raises(SnapshotError, match="frame timestamps differ"):
        make(dataset, tmp_path)


def test_video_probe_output_is_bounded():
    import sys

    from vla_platform.datasets.snapshot_reader import bounded_output

    with pytest.raises(ValueError, match="byte budget"):
        bounded_output([sys.executable, "-c", "print('x' * 1000000)"], 1024)


@pytest.mark.parametrize(
    "feature,bad_shape", [(CAMERA, [[0.0, 0.0, 0.0]]), ("observation.state", [[0.0, 0.0]])]
)
def test_normalization_statistics_need_native_broadcast_shape(
    dataset, tmp_path, feature, bad_shape
):
    path = dataset / "meta/stats.json"
    stats = json.loads(path.read_text())
    stats[feature]["mean"] = bad_shape
    path.write_text(json.dumps(stats))
    with pytest.raises(SnapshotError, match="broadcast shape mismatch"):
        make(dataset, tmp_path)


def test_rehash_reads_no_more_than_the_declared_file_size(dataset, tmp_path, monkeypatch):
    value = make(dataset, tmp_path)
    root = resolve_snapshot(tmp_path / "snapshots", value)
    original = snapshots._open_beneath
    reads = []

    class GrowOnRead:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def fileno(self):
            return self.handle.fileno()

        def read(self, size):
            reads.append(size)
            if len(reads) == 1:
                with (root / "meta/stats.json").open("ab") as writer:
                    writer.write(b"x" * 10000)
            return self.handle.read(size)

    def growing(root_arg, name):
        handle = original(root_arg, name)
        # Inventory opens do not read. Wrap only this file; the first read grows it.
        return GrowOnRead(handle) if str(name) == "meta/stats.json" else handle

    monkeypatch.setattr(snapshots, "_open_beneath", growing)
    declared_size = (root / "meta/stats.json").stat().st_size
    with pytest.raises(SnapshotError, match="changed while reading"):
        verify_snapshot(root, value["manifest_sha256"])
    assert sum(reads) <= declared_size + 1
