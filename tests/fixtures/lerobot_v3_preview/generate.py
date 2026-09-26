"""Regenerate synthetic, known-value LeRobot-v3-style Parquet fixture in place.

Run with workers/_cpu_readers/.venv/bin/python; requires only pyarrow==25.0.1.
This is format/reader evidence, not a robot recording or LeRobot loader test.
"""

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent


def generate():
    assert pa.__version__ == "25.0.1"
    frames = pa.table(
        {
            "index": pa.array(range(6), type=pa.int64()),
            "episode_index": pa.array([0, 0, 0, 1, 1, 1], type=pa.int64()),
            "frame_index": pa.array([0, 1, 2, 0, 1, 2], type=pa.int64()),
            "timestamp": pa.array([0, 0.1, 0.2, 0, 0.1, 0.2], type=pa.float32()),
            "task_index": pa.array([0, 0, 0, 1, 1, 1], type=pa.int64()),
            "action": pa.array([[i, -i] for i in range(6)], type=pa.list_(pa.float32(), 2)),
            "observation.state": pa.array(
                [[i + 0.25, i + 0.5] for i in range(6)], type=pa.list_(pa.float32(), 2)
            ),
        }
    )
    episodes = pa.table(
        {
            "episode_index": pa.array([0, 1], type=pa.int64()),
            "tasks": pa.array([["synthetic reach"], ["synthetic reset"]]),
            "length": pa.array([3, 3], type=pa.int64()),
            "dataset_from_index": pa.array([0, 3], type=pa.int64()),
            "dataset_to_index": pa.array([3, 6], type=pa.int64()),
            "data/chunk_index": pa.array([0, 0], type=pa.int64()),
            "data/file_index": pa.array([0, 0], type=pa.int64()),
        }
    )
    features = {
        name: {"dtype": "int64", "shape": [1], "names": None}
        for name in ["index", "episode_index", "frame_index", "task_index"]
    }
    features.update(
        {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "action": {"dtype": "float32", "shape": [2], "names": ["synthetic_a", "synthetic_b"]},
            "observation.state": {
                "dtype": "float32",
                "shape": [2],
                "names": ["synthetic_x", "synthetic_y"],
            },
        }
    )
    info = {
        "codebase_version": "v3.0",
        "robot_type": "synthetic_fixture",
        "total_episodes": 2,
        "total_frames": 6,
        "total_tasks": 2,
        "fps": 10,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "splits": {"train": "0:2"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": None,
        "features": features,
    }
    manifest = {
        "provenance": "Locally generated synthetic values; no external dataset",
        "files": {},
    }
    for relative, table, row_group_size in [
        ("data/chunk-000/file-000.parquet", frames, 3),
        ("meta/episodes/chunk-000/file-000.parquet", episodes, 1),
    ]:
        target = ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            table,
            target,
            row_group_size=row_group_size,
            compression="NONE",
            version="2.6",
            use_dictionary=False,
            write_statistics=True,
        )
        raw = target.read_bytes()
        manifest["files"][relative] = {
            "rows": table.num_rows,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    target = ROOT / "meta/info.json"
    target.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8", newline="\n")
    raw = target.read_bytes()
    manifest["files"]["meta/info.json"] = {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    generate()
