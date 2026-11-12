"""Fixed CPU adapters. Dataset contents are data, never Python/configuration to execute."""

import csv
import hashlib
import io
import json
import math
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

MAX_ROWS = 100_000
# Training reads single frames in random order. A decoder can only start at a keyframe,
# so with the encoder's default of one keyframe per 250 frames each read decodes over a
# hundred frames first and the GPU waits. A keyframe every second frame, with no
# backward-predicted frames, is the LeRobot convention and makes every read cheap.
RANDOM_ACCESS = ("-g", "2", "-keyint_min", "2", "-sc_threshold", "0", "-bf", "0")
MAX_FILES = 4096
MAX_BYTES = 2 * 1024**3


def safe(root, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or "\\" in str(name) or not path.parts:
        raise ValueError("Expected a relative dataset file")
    target = root / path
    if any(part.is_symlink() for part in [target, *target.parents] if part != root.parent):
        raise ValueError("Dataset symlinks are unsupported")
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Dataset file escapes its folder")
    return target


def inventory(root):
    files, total = [], 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Dataset symlinks are unsupported")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("Only regular dataset files are supported")
        total += path.stat().st_size
        if total > MAX_BYTES or len(files) >= MAX_FILES:
            raise ValueError("Dataset exceeds 2 GiB or 4096 files")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        files.append([path.relative_to(root).as_posix(), path.stat().st_size, digest.hexdigest()])
    return hashlib.sha256(json.dumps(files, separators=(",", ":")).encode()).hexdigest(), files


def tabular(path, *, batch_size=256):
    if path.suffix == ".csv":
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                for key, value in row.items():
                    if value and value[:1] in "[{":
                        row[key] = json.loads(value)
                yield row
    elif path.suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.ParquetFile(path, arrow_extensions_enabled=False)
        decoded = sum(
            table.metadata.row_group(i).total_byte_size
            for i in range(table.metadata.num_row_groups)
        )
        if decoded > 512 * 1024**2:
            raise ValueError("Parquet table exceeds the 512 MiB decoded limit")
        if table.metadata.num_rows > MAX_ROWS:
            raise ValueError("Table exceeds 100,000 rows")
        for batch in table.iter_batches(batch_size=batch_size, use_threads=False):
            yield from batch.to_pylist()
    else:
        with path.open() as handle:
            for index, line in enumerate(handle):
                if index >= MAX_ROWS or len(line) > 1024**2:
                    raise ValueError("Record table exceeds its row or line limit")
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("Each JSONL record must be an object")
                    yield row


def h5_safe(handle):
    import h5py

    def check(group, depth=0):
        if depth > 16 or len(group) > MAX_FILES:
            raise ValueError("HDF5 hierarchy exceeds its limit")
        for key in group:
            if not isinstance(group.get(key, getlink=True), h5py.HardLink):
                raise ValueError("HDF5 external and soft links are unsupported")
            value = group[key]
            if isinstance(value, h5py.Group):
                check(value, depth + 1)
            elif value.is_virtual or value.external:
                raise ValueError("HDF5 external/virtual arrays are unsupported")
            elif math.prod(value.shape) * value.dtype.itemsize > MAX_BYTES:
                raise ValueError("HDF5 array exceeds the 2 GiB decoded limit")
            elif value.ndim == 4 and (value.shape[1] > 2160 or value.shape[2] > 3840):
                raise ValueError("HDF5 camera resolution exceeds 3840×2160")

    check(handle)


def detect(root):
    digest, files = inventory(root)
    names = [row[0] for row in files]
    candidates = []
    if "meta/info.json" in names:
        info = json.loads(safe(root, "meta/info.json").read_bytes())
        version = str(info.get("codebase_version", ""))
        if re.fullmatch(r"v[23]\.\d+(\.\d+)?", version) and "action" in info.get("features", {}):
            candidates.append(
                {
                    "format": f"lerobot_v{version[1]}",
                    "mapping": {},
                    "episodes": info.get("total_episodes", 0),
                    "frames": info.get("total_frames", 0),
                    "fps": info.get("fps"),
                    "cameras": [
                        k
                        for k, v in info["features"].items()
                        if v.get("dtype") in ("image", "video")
                    ],
                }
            )
    h5_files = [n for n in names if Path(n).suffix.lower() in (".h5", ".hdf5")]
    if h5_files:
        import h5py

        families, frames, episodes, mapping = set(), 0, 0, None
        for name in h5_files:
            with h5py.File(safe(root, name), "r") as handle:
                h5_safe(handle)
                if "data" in handle and isinstance(handle["data"], h5py.Group):
                    demos = sorted(handle["data"])
                    if demos and all(
                        "actions" in handle["data"][d] and "obs" in handle["data"][d] for d in demos
                    ):
                        families.add("robomimic_hdf5")
                        for demo in demos:
                            group = handle["data"][demo]
                            frames += len(group["actions"])
                            episodes += 1
                            observations = group["obs"]
                            state = [
                                "obs/" + k
                                for k in sorted(observations)
                                if observations[k].ndim == 2 and observations[k].dtype.kind in "fiu"
                            ]
                            cameras = [
                                "obs/" + k
                                for k in sorted(observations)
                                if observations[k].ndim == 4 and observations[k].shape[-1] == 3
                            ]
                            current = {"action": "actions", "state": state, "cameras": cameras}
                            if mapping is not None and current != mapping:
                                raise ValueError(
                                    "HDF5 episodes have inconsistent observation schemas"
                                )
                            mapping = current
                elif all(
                    k in handle for k in ("action", "observations/qpos", "observations/images")
                ):
                    families.add("aloha_hdf5")
                    frames += len(handle["action"])
                    episodes += 1
                    current = {
                        "action": "action",
                        "state": ["observations/qpos"],
                        "cameras": [
                            "observations/images/" + k
                            for k in sorted(handle["observations/images"])
                        ],
                    }
                    if mapping is not None and current != mapping:
                        raise ValueError("ALOHA episodes have inconsistent schemas")
                    mapping = current
        if len(families) == 1:
            candidates.append(
                {
                    "format": families.pop(),
                    "episodes": episodes,
                    "frames": frames,
                    "fps": None,
                    "mapping": mapping,
                    "cameras": mapping["cameras"],
                }
            )
        elif families:
            raise ValueError("Mixed HDF5 dataset families; select one dataset folder")
    tables = [
        n
        for n in names
        if Path(n).name
        in ("frames.csv", "frames.jsonl", "frames.parquet", "records.csv", "records.jsonl")
    ]
    if len(tables) == 1:
        first = next(tabular(safe(root, tables[0])), {})
        action = next((k for k in ("action", "actions") if k in first), "")
        state = next((k for k in ("observation.state", "state", "qpos") if k in first), "")
        cameras = [
            k for k in first if k.startswith("observation.images.") or k.startswith("image.")
        ]
        if action and state and cameras:
            candidates.append(
                {
                    "format": "image_records",
                    "episodes": None,
                    "frames": None,
                    "fps": None,
                    "mapping": {
                        "table": tables[0],
                        "action": action,
                        "state": [state],
                        "cameras": cameras,
                    },
                    "cameras": cameras,
                }
            )
    if len(candidates) > 1:
        raise ValueError(
            "Ambiguous folder contains multiple dataset formats; choose a single dataset"
        )
    value = (
        candidates[0]
        if candidates
        else {
            "format": "unknown",
            "mapping": {},
            "cameras": [],
            "episodes": None,
            "frames": None,
            "fps": None,
        }
    )
    value.update(
        {
            "sha256": digest,
            "file_count": len(files),
            "total_bytes": sum(row[1] for row in files),
            "convertible": bool(candidates),
        }
    )
    if not candidates:
        if any("tfrecord" in n for n in names) and any(
            Path(n).name == "dataset_info.json" for n in names
        ):
            value["format"] = "rlds_tfrecord"
            value["reason"] = (
                "RLDS/TFRecord detected. Export synchronized frames to CSV/JSONL or LeRobot first; "
                "uploaded TensorFlow builders are not executed."
            )
        elif any(Path(n).suffix in (".bag", ".mcap") for n in names):
            value["format"] = "ros_recording"
            value["reason"] = (
                "ROS recording detected. Export synchronized camera, state and action "
                "topics to CSV/JSONL first."
            )
        else:
            value["reason"] = (
                "No supported robotics schema found. Use LeRobot, robomimic/ALOHA HDF5, "
                "or frames.csv / frames.jsonl with action, observation.state and "
                "observation.images.<view> columns."
            )
    return value


def vector(value):
    import numpy as np

    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if not 1 <= array.size <= 256 or not np.isfinite(array).all():
        raise ValueError("Actions and state need finite vectors of 1–256 values")
    return array


def image(value, root):
    import numpy as np
    from PIL import Image

    if isinstance(value, str):
        with Image.open(safe(root, value)) as opened:
            if opened.width > 3840 or opened.height > 2160:
                raise ValueError("Image resolution exceeds 3840×2160")
            opened.load()
            result = np.asarray(opened.convert("RGB"))
    elif isinstance(value, dict):
        if value.get("bytes"):
            with Image.open(io.BytesIO(value["bytes"])) as opened:
                if opened.width > 3840 or opened.height > 2160:
                    raise ValueError("Image resolution exceeds 3840×2160")
                result = np.asarray(opened.convert("RGB"))
        else:
            return image(value.get("path", ""), root)
    else:
        result = np.asarray(value)
        if result.ndim == 1:
            with Image.open(io.BytesIO(result.tobytes())) as opened:
                if opened.width > 3840 or opened.height > 2160:
                    raise ValueError("Image resolution exceeds 3840×2160")
                result = np.asarray(opened.convert("RGB"))
    if (
        result.dtype != np.uint8
        or result.ndim != 3
        or result.shape[-1] != 3
        or result.shape[0] > 2160
        or result.shape[1] > 3840
    ):
        raise ValueError("Camera images must be uint8 RGB, at most 3840×2160")
    return np.ascontiguousarray(result)


def episodes(root, detected, mapping, fps, task):
    import numpy as np

    kind = detected["format"]
    if kind in ("robomimic_hdf5", "aloha_hdf5"):
        import h5py

        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in (".h5", ".hdf5"):
                continue
            with h5py.File(path, "r") as handle:
                h5_safe(handle)
                groups = (
                    [handle["data"][d] for d in sorted(handle["data"], key=lambda n: (len(n), n))]
                    if kind == "robomimic_hdf5"
                    else [handle]
                )
                for group in groups:
                    count = len(group[mapping["action"]])
                    keys = [mapping["action"], *mapping["state"], *mapping["cameras"]]
                    if count < 1 or count > MAX_ROWS or any(len(group[k]) != count for k in keys):
                        raise ValueError("Episode arrays must have equal nonzero frame counts")

                    def rows(group=group, count=count):
                        for index in range(count):
                            yield {
                                "action": vector(group[mapping["action"]][index]),
                                "state": vector(
                                    np.concatenate(
                                        [vector(group[k][index]) for k in mapping["state"]]
                                    )
                                ),
                                "images": {
                                    k: image(group[k][index], root) for k in mapping["cameras"]
                                },
                                "task": task,
                            }

                    yield count, rows()
    elif kind == "image_records":
        groups = {}
        frame_count = 0
        for row in tabular(safe(root, mapping["table"])):
            frame_count += 1
            if frame_count > MAX_ROWS:
                raise ValueError("Dataset exceeds 100,000 frames")
            index = int(row.get("episode_index", 0))
            groups.setdefault(index, []).append(row)
        for index in sorted(groups):
            records = groups[index]
            if all("frame_index" in row for row in records):
                records.sort(key=lambda row: int(row["frame_index"]))
                if [int(r["frame_index"]) for r in records] != list(range(len(records))):
                    raise ValueError(
                        "Frame indices must be unique and contiguous within each episode"
                    )

            def rows(records=records):
                for row in records:
                    yield {
                        "action": vector(row[mapping["action"]]),
                        "state": vector(np.concatenate([vector(row[k]) for k in mapping["state"]])),
                        "images": {k: image(row[k], root) for k in mapping["cameras"]},
                        "task": str(row.get("task") or task),
                    }

            yield len(records), rows()
    elif kind == "lerobot_v2":
        info = json.loads((root / "meta/info.json").read_text())
        task_file = root / "meta/tasks.jsonl"
        task_names = (
            {row["task_index"]: row["task"] for row in tabular(task_file)}
            if task_file.is_file()
            else {}
        )
        for path in sorted((root / "data").rglob("*.parquet")):
            records = list(tabular(path))
            episode_id = records[0]["episode_index"]
            cameras = [k for k, v in info["features"].items() if v["dtype"] in ("image", "video")]

            def rows(records=records, episode_id=episode_id):
                for row in records:
                    images = {}
                    for key in cameras:
                        if info["features"][key]["dtype"] == "image":
                            images[key] = image(row[key], root)
                        else:
                            name = info["video_path"].format(
                                episode_chunk=episode_id // info.get("chunks_size", 1000),
                                episode_index=episode_id,
                                video_key=key,
                            )
                            images[key] = video_frame(safe(root, name), row["timestamp"])
                    yield {
                        "action": vector(row["action"]),
                        "state": vector(row["observation.state"]),
                        "images": images,
                        "task": task_names.get(row.get("task_index"), task),
                    }

            yield len(records), rows()
    else:
        raise ValueError("This source does not have a conversion adapter")


def ffmpeg():
    binary = shutil.which("ffmpeg")
    if not binary:
        raise ValueError("Install FFmpeg on the app host to convert or preview videos")
    return binary


def video_frame(path, timestamp):
    import numpy as np
    from PIL import Image

    raw = subprocess.check_output(
        [
            ffmpeg(),
            "-v",
            "error",
            "-ss",
            str(timestamp),
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-threads",
            "1",
            "-",
        ],
        timeout=20,
    )
    with Image.open(io.BytesIO(raw)) as opened:
        return np.asarray(opened.convert("RGB"))


def convert(root, output, detected, settings):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    fps = int(settings.get("fps") or detected.get("fps") or 30)
    if not 1 <= fps <= 240:
        raise ValueError("Frame rate must be an integer from 1 to 240")
    task = settings.get("task", "Recorded robot task").strip()
    if not task:
        raise ValueError("Provide a task description")
    mapping = settings.get("mapping") or detected["mapping"]
    output.mkdir(parents=True)
    previews = output.parent / "previews"
    previews.mkdir(exist_ok=True)
    episode_metadata, tasks, numeric, camera_stats, features, samples = [], [], {}, {}, {}, []
    total = 0
    for episode_index, (length, records) in enumerate(episodes(root, detected, mapping, fps, task)):
        if episode_index >= 2000 or total + length > MAX_ROWS:
            raise ValueError("Conversion supports at most 2,000 episodes / 100,000 frames")
        pipes, rows, camera_keys = {}, [], None
        try:
            for frame_index, row in enumerate(records):
                raw_keys = sorted(row["images"])
                if not raw_keys or len(raw_keys) > 8:
                    raise ValueError("Provide 1–8 synchronized camera views")
                normalized = {
                    k: "observation.images."
                    + re.sub(
                        r"[^a-zA-Z0-9_.-]",
                        "_",
                        k.split("/")[-1].removeprefix("observation.images.").removeprefix("image."),
                    )
                    for k in raw_keys
                }
                if len(set(normalized.values())) != len(normalized):
                    raise ValueError("Camera keys collide after normalization")
                if camera_keys is not None and camera_keys != normalized:
                    raise ValueError("Camera views must be identical in every frame")
                camera_keys = normalized
                for key in ("action", "state"):
                    value = row[key]
                    name = "observation.state" if key == "state" else key
                    definition = {
                        "dtype": "float32",
                        "shape": [len(value)],
                        "names": [f"{key}_{i}" for i in range(len(value))],
                    }
                    if name in features and features[name] != definition:
                        raise ValueError("Action/state shapes change across episodes")
                    features[name] = definition
                    numeric.setdefault(name, []).append(value)
                task_text = row["task"]
                if task_text not in tasks:
                    tasks.append(task_text)
                rows.append(
                    {
                        "index": total + frame_index,
                        "episode_index": episode_index,
                        "frame_index": frame_index,
                        "timestamp": frame_index / fps,
                        "task_index": tasks.index(task_text),
                        "action": row["action"].tolist(),
                        "observation.state": row["state"].tolist(),
                    }
                )
                for raw_key, key in normalized.items():
                    array = row["images"][raw_key]
                    definition = {
                        "dtype": "video",
                        "shape": list(array.shape),
                        "names": ["height", "width", "channels"],
                        "info": {
                            "video.fps": fps,
                            "video.codec": "h264",
                            "video.pix_fmt": "yuv420p"
                            if array.shape[0] % 2 == 0 and array.shape[1] % 2 == 0
                            else "yuv444p",
                            "video.is_depth_map": False,
                            "has_audio": False,
                        },
                    }
                    if key in features and features[key] != definition:
                        raise ValueError("Camera resolution changes across frames")
                    features[key] = definition
                    if key not in pipes:
                        path = output / f"videos/{key}/chunk-000/file-{episode_index:03d}.mp4"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        pipes[key] = subprocess.Popen(
                            [
                                ffmpeg(),
                                "-y",
                                "-v",
                                "error",
                                "-f",
                                "rawvideo",
                                "-pix_fmt",
                                "rgb24",
                                "-s",
                                f"{array.shape[1]}x{array.shape[0]}",
                                "-r",
                                str(fps),
                                "-i",
                                "-",
                                "-an",
                                "-c:v",
                                "libx264",
                                *RANDOM_ACCESS,
                                "-threads",
                                "1",
                                "-pix_fmt",
                                definition["info"]["video.pix_fmt"],
                                str(path),
                            ],
                            stdin=subprocess.PIPE,
                            stderr=subprocess.DEVNULL,
                        )
                    pipes[key].stdin.write(array.tobytes())
                    stat = array.reshape(-1, 3).astype(np.float64) / 255
                    previous = camera_stats.setdefault(
                        key,
                        {
                            "sum": np.zeros(3),
                            "sq": np.zeros(3),
                            "min": np.ones(3),
                            "max": np.zeros(3),
                            "count": 0,
                        },
                    )
                    previous["sum"] += stat.sum(axis=0)
                    previous["sq"] += (stat * stat).sum(axis=0)
                    previous["min"] = np.minimum(previous["min"], stat.min(axis=0))
                    previous["max"] = np.maximum(previous["max"], stat.max(axis=0))
                    previous["count"] += len(stat)
                    if frame_index in sorted({0, length // 2, length - 1}):
                        preview_name = f"{episode_index}-{frame_index}-{key}.jpg"
                        rendered = Image.fromarray(array)
                        rendered.thumbnail((640, 480))
                        rendered.save(previews / preview_name)
                        samples.append(
                            {
                                "episode_index": episode_index,
                                "frame_index": frame_index,
                                "camera": key,
                                "path": preview_name,
                                "task": task_text,
                            }
                        )
            for pipe in pipes.values():
                pipe.stdin.close()
                if pipe.wait(timeout=30):
                    raise ValueError("Video conversion failed")
        finally:
            for pipe in pipes.values():
                if pipe.poll() is None:
                    pipe.kill()
                pipe.wait()
        if len(rows) != length:
            raise ValueError("Episode length changed during conversion")
        path = output / f"data/chunk-000/file-{episode_index:03d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        schema = pa.schema(
            [
                (
                    k,
                    pa.list_(pa.float32())
                    if k in ("action", "observation.state")
                    else pa.float32()
                    if k == "timestamp"
                    else pa.int64(),
                )
                for k in rows[0]
            ]
        )
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
        meta = {
            "episode_index": episode_index,
            "tasks": list(dict.fromkeys(tasks[r["task_index"]] for r in rows)),
            "length": length,
            "dataset_from_index": total,
            "dataset_to_index": total + length,
            "data/chunk_index": 0,
            "data/file_index": episode_index,
        }
        for key in camera_keys.values():
            meta.update(
                {
                    f"videos/{key}/chunk_index": 0,
                    f"videos/{key}/file_index": episode_index,
                    f"videos/{key}/from_timestamp": 0.0,
                    f"videos/{key}/to_timestamp": length / fps,
                }
            )
        episode_metadata.append(meta)
        total += length
    if not total:
        raise ValueError("Dataset contains no frames")
    features.update(
        {
            key: {"dtype": dtype, "shape": [1], "names": None}
            for key, dtype in [
                ("index", "int64"),
                ("episode_index", "int64"),
                ("frame_index", "int64"),
                ("timestamp", "float32"),
                ("task_index", "int64"),
            ]
        }
    )
    info = {
        "codebase_version": "v3.0",
        "robot_type": settings.get("robot_type") or "unspecified",
        "total_episodes": len(episode_metadata),
        "total_frames": total,
        "total_tasks": len(tasks),
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": fps,
        "splits": {"train": f"0:{len(episode_metadata)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    meta_root = output / "meta"
    (meta_root / "episodes/chunk-000").mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(episode_metadata), meta_root / "episodes/chunk-000/file-000.parquet"
    )
    task_table = pa.Table.from_pylist([{"task_index": i, "task": t} for i, t in enumerate(tasks)])
    pandas_metadata = {
        "index_columns": ["task"],
        "column_indexes": [],
        "columns": [
            {
                "name": "task_index",
                "field_name": "task_index",
                "pandas_type": "int64",
                "numpy_type": "int64",
                "metadata": None,
            },
            {
                "name": "task",
                "field_name": "task",
                "pandas_type": "unicode",
                "numpy_type": "object",
                "metadata": None,
            },
        ],
        "creator": {"library": "pyarrow", "version": pa.__version__},
        "pandas_version": "2.2.3",
    }
    task_table = task_table.replace_schema_metadata(
        {b"pandas": json.dumps(pandas_metadata).encode()}
    )
    pq.write_table(task_table, meta_root / "tasks.parquet")
    stats = {}
    for key, values in numeric.items():
        array = np.stack(values)
        stats[key] = {
            "mean": array.mean(0).tolist(),
            "std": array.std(0).tolist(),
            "min": array.min(0).tolist(),
            "max": array.max(0).tolist(),
            "count": [len(array)],
        }
    for key, value in camera_stats.items():
        mean = value["sum"] / value["count"]
        stats[key] = {
            "mean": mean.tolist(),
            "std": np.sqrt(np.maximum(0, value["sq"] / value["count"] - mean**2)).tolist(),
            "min": value["min"].tolist(),
            "max": value["max"].tolist(),
            "count": [total],
        }
    (meta_root / "stats.json").write_text(json.dumps(stats, allow_nan=False))
    (meta_root / "info.json").write_text(json.dumps(info, allow_nan=False))
    (output.parent / "samples.json").write_text(json.dumps(samples))
    return {
        "episodes": len(episode_metadata),
        "frames": total,
        "fps": fps,
        "cameras": list(camera_stats),
        "format": "lerobot_v3",
        "samples": len(samples),
    }


def _dataset(root):
    """One finalized LeRobot v3 dataset: its metadata, tasks and episode rows."""
    info = json.loads(safe(root, "meta/info.json").read_bytes())
    if info.get("codebase_version") != "v3.0":
        raise ValueError("Only finalized LeRobot v3 datasets can be combined")
    if info.get("data_path") != "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet":
        raise ValueError("Dataset uses a nonstandard data layout")
    tasks = {}
    for row in tabular(safe(root, "meta/tasks.parquet")):
        name = row.get("task", row.get("__index_level_0__"))
        if not isinstance(name, str) or not name.strip() or type(row.get("task_index")) is not int:
            raise ValueError("Dataset has an invalid task table")
        tasks[row["task_index"]] = name
    rows = []
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        rows.extend(tabular(safe(root, path.relative_to(root).as_posix())))
    rows.sort(key=lambda row: row["episode_index"])
    if len(rows) != info.get("total_episodes") or not rows:
        raise ValueError("Dataset episode metadata is incomplete")
    return info, tasks, rows


def _combined_statistics(parts):
    """Pool per-dataset statistics, weighted by how many frames each one describes."""
    import numpy as np

    output = {}
    for key in parts[0]:
        entries = [part[key] for part in parts if key in part]
        if len(entries) != len(parts):
            continue  # Normalization needs every source; the validator names what is missing.
        counts = np.array([float(entry["count"][0]) for entry in entries])
        total = counts.sum()
        means = np.stack([np.asarray(entry["mean"], dtype=float) for entry in entries])
        weights = counts.reshape((-1,) + (1,) * (means.ndim - 1))
        mean = (means * weights).sum(0) / total
        spread = np.stack([np.asarray(entry["std"], dtype=float) ** 2 for entry in entries])
        variance = ((spread + (means - mean) ** 2) * weights).sum(0) / total
        output[key] = {
            "mean": mean.tolist(),
            "std": np.sqrt(np.maximum(variance, 0)).tolist(),
            "min": np.stack([np.asarray(e["min"], dtype=float) for e in entries]).min(0).tolist(),
            "max": np.stack([np.asarray(e["max"], dtype=float) for e in entries]).max(0).tolist(),
            "count": [int(total)],
        }
        for name in ("q01", "q10", "q50", "q90", "q99"):
            if all(name in entry for entry in entries):
                # A frame-weighted estimate; exact quantiles would need every frame again.
                values = np.stack([np.asarray(entry[name], dtype=float) for entry in entries])
                output[key][name] = ((values * weights).sum(0) / total).tolist()
    return output


def _recorder_statement(roots):
    """What the Robot Lab recorder said about every source, when all of them agree.

    The recorder states the units of the arm positions. A policy trained on the combined
    dataset speaks those units, so the statement carries over, but only when every source
    is a finished Robot Lab recording and they all say the same thing.
    """
    shared = None
    for root in roots:
        try:
            recording = json.loads(safe(root, "openjensen.json").read_bytes())
        except OSError, ValueError:
            return None
        if (
            not isinstance(recording, dict)
            or recording.get("created_by") != "OpenJensen Robot Lab"
            or recording.get("finalized") is not True
            or recording.get("abandoned", False) is not False
        ):
            return None
        statement = {
            key: recording.get(key) for key in ("created_by", "format_version", "units", "format")
        }
        if shared is not None and statement != shared:
            return None
        shared = statement
    return shared


def merge(roots, output):
    """Join datasets that share features and frame rate into one LeRobot v3 dataset.

    Recording one episode per dataset is common (an agent-supervised run writes exactly
    that), and one episode cannot be trained on. Episodes keep their order: every episode
    of the first dataset, then the second, and so on. Frames are copied as they are; only
    the episode, frame and task numbering changes.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not 2 <= len(roots) <= 1000:
        raise ValueError("Choose between 2 and 1,000 datasets to combine")
    if output.exists() and any(output.iterdir()):
        raise ValueError("The combined dataset folder must be empty")
    sources = [(root, *_dataset(root)) for root in roots]
    first = sources[0][1]
    for index, (_, info, _, _) in enumerate(sources[1:], start=2):
        if info.get("fps") != first.get("fps"):
            raise ValueError(
                f"Dataset {index} was recorded at {info.get('fps')} fps and the first at "
                f"{first.get('fps')} fps; only datasets with one frame rate can be combined"
            )
        if info.get("features") != first.get("features"):
            ours, theirs = set(first.get("features", {})), set(info.get("features", {}))
            detail = (
                "it has different cameras or values: " + ", ".join(sorted(ours ^ theirs))
                if ours != theirs
                else "its cameras or values have a different size or type"
            )
            raise ValueError(f"Dataset {index} does not match the first: {detail}")
    videos = [key for key, value in first["features"].items() if value.get("dtype") == "video"]
    task_ids, episode_rows, statistics = {}, [], []
    episode_index = frame_index = 0
    video_files = {key: 0 for key in videos}
    for root, info, tasks, rows in sources:
        try:
            statistics.append(json.loads(safe(root, "meta/stats.json").read_bytes()))
        except OSError as exc:
            raise ValueError("A dataset has no normalization statistics") from exc
        renamed = {key: {} for key in videos}  # (chunk, file) in the source -> new file index
        for row in rows:
            length = row["length"]
            names = row.get("tasks") or []
            for name in names:
                task_ids.setdefault(name, len(task_ids))
            source_file = safe(
                root,
                info["data_path"].format(
                    chunk_index=row["data/chunk_index"], file_index=row["data/file_index"]
                ),
            )
            target = output / (
                f"data/chunk-{episode_index // 1000:03d}/file-{episode_index % 1000:03d}.parquet"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            table = pq.ParquetFile(source_file, arrow_extensions_enabled=False)
            written = 0
            with pq.ParquetWriter(target, table.schema_arrow) as writer:
                for batch in table.iter_batches(batch_size=32, use_threads=False):
                    keep = [
                        position
                        for position, value in enumerate(batch.column("episode_index").to_pylist())
                        if value == row["episode_index"]
                    ]
                    if not keep:
                        continue
                    batch = batch.take(pa.array(keep, type=pa.int64()))
                    columns = dict(zip(batch.schema.names, batch.columns, strict=True))
                    count = batch.num_rows
                    columns["episode_index"] = pa.array([episode_index] * count, type=pa.int64())
                    columns["index"] = pa.array(
                        range(frame_index + written, frame_index + written + count),
                        type=pa.int64(),
                    )
                    columns["task_index"] = pa.array(
                        [
                            task_ids[tasks[value]]
                            for value in batch.column("task_index").to_pylist()
                        ],
                        type=pa.int64(),
                    )
                    writer.write_batch(
                        pa.RecordBatch.from_arrays(
                            [columns[name] for name in batch.schema.names],
                            schema=table.schema_arrow,
                        )
                    )
                    written += count
            if written != length:
                raise ValueError("A dataset's frames do not match its episode metadata")
            entry = {
                key: value
                for key, value in row.items()
                if key.startswith("stats/") or key.endswith(("/from_timestamp", "/to_timestamp"))
            }
            entry.update(
                {
                    "episode_index": episode_index,
                    "tasks": names,
                    "length": length,
                    "data/chunk_index": episode_index // 1000,
                    "data/file_index": episode_index % 1000,
                    "dataset_from_index": frame_index,
                    "dataset_to_index": frame_index + length,
                    "meta/episodes/chunk_index": 0,
                    "meta/episodes/file_index": 0,
                }
            )
            for key in videos:
                prefix = "videos/" + key
                origin = (row[prefix + "/chunk_index"], row[prefix + "/file_index"])
                if origin not in renamed[key]:
                    renamed[key][origin] = video_files[key]
                    video_files[key] += 1
                    number = renamed[key][origin]
                    copied = output / (
                        f"videos/{key}/chunk-{number // 1000:03d}/file-{number % 1000:03d}.mp4"
                    )
                    copied.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(
                        safe(
                            root,
                            info["video_path"].format(
                                video_key=key, chunk_index=origin[0], file_index=origin[1]
                            ),
                        ),
                        copied,
                    )
                number = renamed[key][origin]
                entry[prefix + "/chunk_index"] = number // 1000
                entry[prefix + "/file_index"] = number % 1000
            episode_rows.append(entry)
            episode_index += 1
            frame_index += length
    meta = output / "meta"
    (meta / "episodes/chunk-000").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(episode_rows), meta / "episodes/chunk-000/file-000.parquet")
    task_table = pa.Table.from_pydict(
        {"task_index": list(task_ids.values()), "task": list(task_ids)},
        schema=pa.schema([("task_index", pa.int64()), ("task", pa.string())]),
    )
    pq.write_table(task_table, meta / "tasks.parquet")
    info = {
        **first,
        "total_episodes": episode_index,
        "total_frames": frame_index,
        "total_tasks": len(task_ids),
        "splits": {"train": f"0:{episode_index}"},
    }
    (meta / "info.json").write_text(json.dumps(info, allow_nan=False))
    (meta / "stats.json").write_text(json.dumps(_combined_statistics(statistics), allow_nan=False))
    statement = _recorder_statement([root for root, *_ in sources])
    if statement is not None:
        (output / "openjensen.json").write_text(
            json.dumps(
                {
                    **statement,
                    "episodes": episode_index,
                    "frames": frame_index,
                    "fps": first["fps"],
                    "task": next(iter(task_ids), ""),
                    "finalized": True,
                    "abandoned": False,
                    "pending_frames": 0,
                    "camera_features": [
                        key
                        for key, value in first["features"].items()
                        if value.get("dtype") in {"video", "image"}
                    ],
                },
                allow_nan=False,
            )
        )
    return {
        "episodes": episode_index,
        "frames": frame_index,
        "fps": first["fps"],
        "tasks": list(task_ids),
        "sources": len(sources),
    }


def preview_lerobot(root, output):
    """Bounded recorded-image sequences; video sources keep their thumbnail path."""
    import io

    from PIL import Image

    metadata = (root / "meta/info.json").read_bytes()
    info = json.loads(metadata)
    cameras = {k: v for k, v in info["features"].items() if v["dtype"] in ("image", "video")}
    if not cameras or any(v["dtype"] != "image" for v in cameras.values()):
        return _preview_lerobot_samples(root, output)
    if any(not re.fullmatch(r"observation\.[A-Za-z0-9_.-]{1,100}", k) for k in cameras):
        raise ValueError("Camera names are not safe preview filenames")
    if output.is_symlink():
        raise ValueError("Preview output must not be a symlink")
    output.mkdir(exist_ok=True)
    metadata_sha256 = hashlib.sha256(metadata).hexdigest()
    samples, total, truncated = [], 0, False
    paths = sorted((root / "data").rglob("*.parquet"))
    for path in paths[:64]:
        safe(root, path.relative_to(root).as_posix())
        for row in tabular(path, batch_size=1):
            for key in cameras:
                if len(samples) >= 2400:
                    truncated = True
                    break
                episode, frame = row["episode_index"], row["frame_index"]
                if type(episode) is not int or type(frame) is not int or min(episode, frame) < 0:
                    raise ValueError("Preview frame indices must be nonnegative integers")
                rendered = Image.fromarray(image(row[key], root)).convert("RGB")
                rendered.thumbnail((640, 480))
                buffer = io.BytesIO()
                rendered.save(buffer, format="JPEG", quality=80)
                data = buffer.getvalue()
                if total + len(data) > 64 * 1024**2:
                    truncated = True
                    break
                name = f"{episode}-{frame}-{key}.jpg"
                temporary = output / (".preview-" + uuid.uuid4().hex)
                try:
                    temporary.write_bytes(data)
                    temporary.replace(output / name)
                finally:
                    temporary.unlink(missing_ok=True)
                total += len(data)
                samples.append(
                    {
                        "episode_index": episode,
                        "frame_index": frame,
                        "camera": key,
                        "path": name,
                        "task": str(row.get("task", "")),
                        "timestamp": float(row.get("timestamp", frame / info["fps"])),
                        "state": row.get("observation.state"),
                        "action": row.get("action"),
                        "metadata_sha256": metadata_sha256,
                    }
                )
            if truncated:
                break
        if truncated:
            break
    if samples and (truncated or len(paths) > 64):
        samples[-1]["preview_truncated"] = True
    temporary = output.parent / (".samples-" + uuid.uuid4().hex)
    try:
        temporary.write_text(json.dumps(samples, allow_nan=False))
        temporary.replace(output.parent / "samples.json")
    finally:
        temporary.unlink(missing_ok=True)
    return {"samples": len(samples), "bytes": total, "truncated": truncated or len(paths) > 64}


def _preview_lerobot_samples(root, output):
    from PIL import Image

    info = json.loads((root / "meta/info.json").read_text())
    cameras = {k: v for k, v in info["features"].items() if v["dtype"] in ("image", "video")}
    output.mkdir(exist_ok=True)
    samples = []
    paths = sorted((root / "data").rglob("*.parquet"))[:2]
    for path in paths:
        for row in list(tabular(path))[:1]:
            for key, feature in cameras.items():
                if feature["dtype"] == "image":
                    array = image(row[key], root)
                else:
                    if info["codebase_version"].startswith("v2"):
                        name = info["video_path"].format(
                            episode_index=row["episode_index"],
                            episode_chunk=row["episode_index"] // info.get("chunks_size", 1000),
                            video_key=key,
                        )
                        timestamp = row["timestamp"]
                    else:
                        episode = next(
                            (
                                e
                                for p in sorted((root / "meta/episodes").rglob("*.parquet"))
                                for e in tabular(p)
                                if e["episode_index"] == row["episode_index"]
                            ),
                            None,
                        )
                        if not episode:
                            continue
                        name = info["video_path"].format(
                            chunk_index=episode[f"videos/{key}/chunk_index"],
                            file_index=episode[f"videos/{key}/file_index"],
                            video_key=key,
                        )
                        timestamp = episode[f"videos/{key}/from_timestamp"]
                    array = video_frame(safe(root, name), timestamp)
                name = f"{row['episode_index']}-{row['frame_index']}-{key}.jpg"
                rendered = Image.fromarray(array)
                rendered.thumbnail((640, 480))
                rendered.save(output / name)
                samples.append(
                    {
                        "episode_index": row["episode_index"],
                        "frame_index": row["frame_index"],
                        "camera": key,
                        "path": name,
                        "task": "",
                    }
                )
    (output.parent / "samples.json").write_text(json.dumps(samples))
    return {"samples": len(samples)}


def demo(root, episodes=2):
    from PIL import Image, ImageDraw

    root.mkdir(exist_ok=True)
    records = []
    for episode in range(episodes):
        for frame in range(12):
            row = {
                "episode_index": episode,
                "frame_index": frame,
                "action": [frame / 12, episode, 0.5],
                "observation.state": [frame / 12, episode, 0.4],
                "task": "Move the red block into the outlined target (synthetic example)",
            }
            for camera in ("front", "wrist"):
                image = Image.new("RGB", (320, 240), "#e8eef4")
                draw = ImageDraw.Draw(image)
                draw.rectangle((220, 80, 285, 145), outline="#245780", width=3)
                x = 40 + frame * 14
                draw.rectangle((x, 85, x + 35, 120), fill="#e15b4b")
                draw.line((160, 235, x + 17, 115), fill="#4b6579", width=10)
                if camera == "wrist":
                    draw.rectangle((0, 35, 320, 240), fill="#d3e0e9")
                    draw.rounded_rectangle(
                        (210, 60, 298, 148), radius=8, outline="#245780", width=4
                    )
                    draw.ellipse((x, 80, x + 46, 126), fill="#e15b4b")
                    draw.line((0, 195, x + 23, 117), fill="#617d90", width=15)
                draw.text(
                    (12, 12),
                    f"{camera} view / episode {episode + 1} / frame {frame}",
                    fill="#153149",
                )
                name = f"images/{camera}/{episode}-{frame}.png"
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path)
                row[f"observation.images.{camera}"] = name
            records.append(row)
    (root / "frames.jsonl").write_text("\n".join(json.dumps(row) for row in records) + "\n")
    return detect(root)


def main():
    if sys.platform != "win32":
        import resource

        resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
        resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_BYTES, MAX_BYTES))
        if sys.platform == "linux":
            resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
    request = json.loads(sys.stdin.buffer.read(1024**2))
    op = request["operation"]
    if op == "merge":
        print(
            json.dumps(
                merge([Path(root) for root in request["roots"]], Path(request["output"])),
                allow_nan=False,
            )
        )
        return
    root = Path(request["root"])
    if op == "detect":
        result = detect(root)
    elif op == "demo":
        result = demo(root)
    elif op == "preview":
        result = preview_lerobot(root, Path(request["output"]))
    else:
        detected = detect(root)
        if detected["sha256"] != request["sha256"]:
            raise ValueError("Source bytes changed since format detection")
        result = convert(root, Path(request["output"]), detected, request["settings"])
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {str(exc)[:500]}"}))
        sys.exit(1)
