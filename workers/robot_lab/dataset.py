"""LeRobot v3.0 Parquet/image recorder without Torch on the hardware computer.

The wire format is verified against the official LeRobot 0.4.4 loader. Images are
lossless RGB PNG bytes in Arrow Image structs; task/episode metadata and stats
follow LeRobot v3.0, allowing direct training after importing the folder.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import time
from collections import deque
from pathlib import Path, PurePosixPath

from hardware import JOINTS
from image_writer import AsyncImageWriter

SLUG = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$")


def safe_slug(value):
    if not isinstance(value, str) or not SLUG.fullmatch(value):
        raise ValueError("Use a name of 1–80 letters, digits, underscores or hyphens")
    return value


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def statistics(values, image=False):
    import numpy as np

    frame_count = len(values)
    values = np.asarray(values, dtype=np.float64)
    if image:
        # Image samples are [frames, sample pixels, channels]; LeRobot uses C11.
        values = values.reshape(-1, 3)
    if values.ndim == 1:
        values = values[:, None]
    result = {
        "min": values.min(0),
        "max": values.max(0),
        "mean": values.mean(0),
        "std": values.std(0),
    }
    for quantile, key in ((0.01, "q01"), (0.10, "q10"), (0.5, "q50"), (0.9, "q90"), (0.99, "q99")):
        result[key] = np.quantile(values, quantile, axis=0)
    result = {
        key: (value.reshape(3, 1, 1).tolist() if image else value.tolist())
        for key, value in result.items()
    }
    result["count"] = [frame_count]
    return result


def aggregate_stats(episodes):
    import numpy as np

    if not episodes:
        return {}
    output = {}
    for feature in episodes[0]:
        entries = [episode[feature] for episode in episodes]
        counts = np.array([entry["count"][0] for entry in entries], dtype=float)
        means = np.stack([entry["mean"] for entry in entries])
        weights = counts.reshape((-1,) + (1,) * (means.ndim - 1))
        mean = (means * weights).sum(0) / counts.sum()
        variances = np.stack([np.array(entry["std"]) ** 2 for entry in entries])
        variance = ((variances + (means - mean) ** 2) * weights).sum(0) / counts.sum()
        output[feature] = {
            "min": np.min(np.stack([entry["min"] for entry in entries]), 0).tolist(),
            "max": np.max(np.stack([entry["max"] for entry in entries]), 0).tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "count": [int(counts.sum())],
        }
        # Conservative weighted estimate; exact per-episode quantiles are kept.
        for key in ("q01", "q10", "q50", "q90", "q99"):
            output[feature][key] = (
                (np.stack([entry[key] for entry in entries]) * weights).sum(0).tolist()
            )
            output[feature][key] = (np.asarray(output[feature][key]) / counts.sum()).tolist()
    return output


class DatasetWriter:
    def __init__(self, root: Path, dataset_id: str, task: str, fps: int, image_shape=None):
        import numpy  # noqa: F401
        import pyarrow  # noqa: F401
        from PIL import Image  # noqa: F401

        self.id = safe_slug(dataset_id)
        self.root = root / self.id
        if self.root.exists():
            raise ValueError(
                "Dataset name already exists; choose a new name to preserve existing episodes"
            )
        self.root.mkdir(parents=True)
        self.task = task
        self.fps = fps
        self.image_shape = image_shape
        self.image_writer = None
        self.features = {
            "observation.state": {
                "dtype": "float32",
                "shape": [6],
                "names": [joint + ".pos" for joint in JOINTS],
            },
            "action": {
                "dtype": "float32",
                "shape": [6],
                "names": [joint + ".pos" for joint in JOINTS],
            },
        }
        if image_shape is not None:
            self.features["observation.images.front"] = {
                "dtype": "image",
                "shape": list(image_shape),
                "names": ["height", "width", "channels"],
            }
        for name in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            self.features[name] = {
                "dtype": "float32" if name == "timestamp" else "int64",
                "shape": [1],
                "names": None,
            }
        self.episodes = []
        self.all_stats = []
        self.total_frames = 0
        self.frames = 0
        self.episode_active = False
        self.finalized = False
        self.abandoned = False
        self.capture_times = deque(maxlen=max(60, fps * 3))
        self.actual_fps = None
        self.episode_capture_fps = []
        self.episode_capture_timing = []
        self.pending = self.root / ".pending"
        self.pending.mkdir()
        self.write_metadata()
        if image_shape is not None:
            self.image_writer = AsyncImageWriter()

    def summary(self):
        return {
            "id": self.id,
            "path": str(self.root),
            "episodes": len(self.episodes),
            "frames": self.total_frames,
            "fps": self.fps,
            "task": self.task,
            "format": "lerobot",
            "finalized": self.finalized,
            "abandoned": self.abandoned,
            "pending_frames": self.frames if self.abandoned else 0,
            "measured_fps": self.episode_capture_fps,
            "capture_timing": self.episode_capture_timing,
            "image_write_error": self.image_writer.error if self.image_writer else None,
        }

    def write_metadata(self):
        atomic_json(
            self.root / "openjensen.json",
            {
                **self.summary(),
                "created_by": "OpenJensen Robot Lab",
                "format_version": "v3.0",
                "units": "degrees; gripper percent",
            },
        )
        info = {
            "codebase_version": "v3.0",
            "robot_type": "so101_follower",
            "total_episodes": len(self.episodes),
            "total_frames": self.total_frames,
            "total_tasks": 1,
            "chunks_size": 1000,
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 200,
            "fps": self.fps,
            "splits": {"train": f"0:{len(self.episodes)}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": None,
            "features": self.features,
        }
        atomic_json(self.root / "meta/info.json", info)
        atomic_json(self.root / "meta/stats.json", aggregate_stats(self.all_stats))
        import pyarrow as pa
        import pyarrow.parquet as pq

        tasks = pa.Table.from_pydict({"task_index": [0], "__index_level_0__": [self.task]})
        pandas_meta = {
            "index_columns": ["__index_level_0__"],
            "column_indexes": [
                {
                    "name": None,
                    "field_name": None,
                    "pandas_type": "unicode",
                    "numpy_type": "object",
                    "metadata": {"encoding": "UTF-8"},
                }
            ],
            "columns": [
                {
                    "name": "task_index",
                    "field_name": "task_index",
                    "pandas_type": "int64",
                    "numpy_type": "int64",
                    "metadata": None,
                },
                {
                    "name": None,
                    "field_name": "__index_level_0__",
                    "pandas_type": "unicode",
                    "numpy_type": "object",
                    "metadata": None,
                },
            ],
            "attributes": {},
            "creator": {"library": "pyarrow", "version": pa.__version__},
            "pandas_version": "2.2.0",
        }
        tasks = tasks.replace_schema_metadata({b"pandas": json.dumps(pandas_meta).encode()})
        pq.write_table(tasks, self.root / "meta/tasks.parquet")
        if self.episodes:
            target = self.root / "meta/episodes/chunk-000/file-000.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(self.episodes), target)

    def start_episode(self):
        if self.abandoned:
            raise ValueError("This unsaved recording was closed; create a new dataset")
        if self.finalized or self.episode_active or self.frames:
            raise ValueError("Save or discard the pending episode before starting another")
        self.episode_active = True
        self.started_at = time.monotonic()
        self.last_frame_at = None
        self.first_frame_at = None
        self.largest_gap = 0.0
        self.capture_times.clear()
        self.actual_fps = None

    def add_frame(self, positions: dict, action: dict, image=None, captured_at=None):
        if not self.episode_active:
            return
        if self.frames >= self.fps * 300:
            raise ValueError("Episode reached the 5-minute limit; save it before continuing")
        if shutil.disk_usage(self.root).free < 512 * 1024**2:
            raise ValueError(
                "Recording stopped with less than 512 MiB disk space remaining; saved episodes "
                "are preserved"
            )
        now = time.monotonic() if captured_at is None else captured_at
        if self.last_frame_at is not None and now - self.last_frame_at >= max(0.5, 3.0 / self.fps):
            raise ValueError(
                "Capture fell below requested FPS after an interruption; "
                "pending frames are preserved. "
                "Check camera/USB and "
                "computer load. For a new dataset, Automatic frame rate measures this setup."
            )
        if self.first_frame_at is None:
            self.first_frame_at = now
        if self.last_frame_at is not None:
            self.largest_gap = max(self.largest_gap, now - self.last_frame_at)
        self.capture_times.append(now)
        if len(self.capture_times) > 1:
            elapsed = self.capture_times[-1] - self.capture_times[0]
            self.actual_fps = (len(self.capture_times) - 1) / elapsed if elapsed > 0 else None
            if elapsed >= 3 and self.actual_fps is not None and self.actual_fps < self.fps * 0.9:
                raise ValueError(
                    "Sustained capture cannot maintain this dataset frame rate; "
                    "pending frames are preserved. "
                    "Create a new dataset with Automatic frame rate to measure current conditions."
                )
        state = [float(positions[joint]) for joint in JOINTS]
        target = [float(action.get(joint, positions[joint])) for joint in JOINTS]
        if not all(math.isfinite(value) for value in state + target):
            raise ValueError("Non-finite motor reading; episode stopped")
        if self.image_shape is not None:
            if image is None or tuple(image.shape) != tuple(self.image_shape):
                raise ValueError("Camera image missing or changed size; episode stopped")
            self.image_writer.save_image(image, self.pending / f"{self.frames:06d}.png")
        row = {
            "observation.state": state,
            "action": target,
            "timestamp": self.frames / self.fps,
            "frame_index": self.frames,
            "episode_index": len(self.episodes),
            "index": self.total_frames + self.frames,
            "task_index": 0,
        }
        with (self.pending / "frames.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.frames += 1
        self.last_frame_at = now

    def _schema(self):
        import pyarrow as pa

        fields, hf = [], {}
        for key, feature in self.features.items():
            if feature["dtype"] == "image":
                fields.append(
                    pa.field(key, pa.struct([("bytes", pa.binary()), ("path", pa.string())]))
                )
                hf[key] = {"_type": "Image"}
            elif key in ("observation.state", "action"):
                fields.append(pa.field(key, pa.list_(pa.float32(), 6)))
                hf[key] = {
                    "feature": {"dtype": "float32", "_type": "Value"},
                    "length": 6,
                    "_type": "Sequence",
                }
            else:
                dtype = "float32" if key == "timestamp" else "int64"
                fields.append(pa.field(key, pa.float32() if key == "timestamp" else pa.int64()))
                hf[key] = {"dtype": dtype, "_type": "Value"}
        return pa.schema(
            fields, metadata={b"huggingface": json.dumps({"info": {"features": hf}}).encode()}
        )

    def save_episode(self):
        if self.abandoned:
            raise ValueError("This unsaved recording was closed; its files are preserved")
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        from PIL import Image

        self.episode_active = False
        if self.image_writer:
            self.image_writer.retry_failed()
        if self.frames < 2:
            raise ValueError("Record at least two frames before saving an episode")
        if not len(self.episodes) < 1000:
            raise ValueError("Dataset reached 1000 episodes; finalize and create another dataset")
        index = len(self.episodes)
        target = self.root / f"data/chunk-000/file-{index:03d}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".tmp")
        collected = {key: [] for key in self.features}
        schema = self._schema()
        with pq.ParquetWriter(partial, schema) as writer:
            batch = []
            with (self.pending / "frames.jsonl").open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if self.image_shape is not None:
                        image_path = self.pending / f"{row['frame_index']:06d}.png"
                        row["observation.images.front"] = {
                            "bytes": image_path.read_bytes(),
                            "path": None,
                        }
                        with Image.open(image_path) as image:
                            collected["observation.images.front"].append(
                                np.asarray(image.resize((8, 8)), dtype=float).reshape(-1, 3) / 255
                            )
                    for key in collected:
                        if key != "observation.images.front":
                            collected[key].append(row[key])
                    batch.append(row)
                    if len(batch) == 20:
                        writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                        batch = []
                if batch:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
        partial.replace(target)
        stats = {
            key: statistics(values, key == "observation.images.front")
            for key, values in collected.items()
        }
        entry = {
            "episode_index": index,
            "tasks": [self.task],
            "length": self.frames,
            "data/chunk_index": 0,
            "data/file_index": index,
            "dataset_from_index": self.total_frames,
            "dataset_to_index": self.total_frames + self.frames,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for feature, values in stats.items():
            for key, value in values.items():
                entry[f"stats/{feature}/{key}"] = value
        self.episodes.append(entry)
        self.episode_capture_fps.append(self.actual_fps)
        self.episode_capture_timing.append(
            {
                "elapsed_s": self.last_frame_at - self.first_frame_at,
                "largest_gap_s": self.largest_gap,
                "samples": self.frames,
            }
        )
        self.all_stats.append(stats)
        self.total_frames += self.frames
        try:
            self.write_metadata()
        except Exception:
            self.episodes.pop()
            self.episode_capture_fps.pop()
            self.episode_capture_timing.pop()
            self.all_stats.pop()
            self.total_frames -= self.frames
            raise
        self.discard_episode()

    def discard_episode(self):
        if self.abandoned:
            raise ValueError("This unsaved recording was closed; its files are preserved")
        self.episode_active = False
        if self.image_writer:
            self.image_writer.wait_until_done(ignore_error=True)
        # Only this writer's own private pending directory is retired.
        for path in self.pending.iterdir():
            if path.is_file() and (
                path.name == "frames.jsonl" or re.fullmatch(r"\d{6}\.png(?:\.tmp)?", path.name)
            ):
                path.unlink()
        self.frames = 0
        if self.image_writer:
            self.image_writer.clear_error()

    def abandon_empty(self):
        """Retire an explicitly closed unsaved recording without deleting files.

        A caller must separately verify explicit user confirmation. Saved
        episodes can only be finalized through the normal dataset workflow.
        Metadata is committed before the caller detaches this writer. If that
        write fails, collection stays paused and the action can be retried.
        """
        if self.finalized or self.abandoned or self.episodes or self.total_frames:
            raise ValueError("Only a recording with no saved episodes can be closed this way")
        self.episode_active = False
        if self.image_writer:
            self.image_writer.wait_until_done(ignore_error=True)
        metadata_path = self.root / "openjensen.json"
        metadata = json.loads(metadata_path.read_text())
        metadata.update(abandoned=True, pending_frames=self.frames)
        atomic_json(metadata_path, metadata)
        self.abandoned = True
        if self.image_writer:
            self.image_writer.stop(ignore_error=True)

    def finalize(self):
        if self.abandoned:
            raise ValueError("A closed unsaved recording cannot be finalized or imported")
        if self.episode_active or self.frames:
            raise ValueError("Save or discard the current episode before finalizing")
        if not self.episodes:
            raise ValueError("Save at least one episode before finalizing")
        if self.image_writer:
            self.image_writer.wait_until_done()
        self.finalized = True
        try:
            self.write_metadata()
        except Exception:
            self.finalized = False
            raise
        self.pending.rmdir()
        self.close()

    def close(self):
        if self.image_writer:
            self.image_writer.stop()


def list_datasets(root):
    output = []
    for path in sorted(root.glob("*/openjensen.json")):
        if path.parent.is_symlink():
            continue
        try:
            data = json.loads(path.read_text())
            data.setdefault("abandoned", False)
            data.setdefault("pending_frames", 0)
            output.append(
                {
                    key: data.get(key)
                    for key in (
                        "id",
                        "path",
                        "episodes",
                        "frames",
                        "fps",
                        "task",
                        "format",
                        "finalized",
                        "abandoned",
                        "pending_frames",
                    )
                }
            )
        except (OSError, ValueError):  # fmt: skip
            continue
    return output


def dataset_file(root: Path, dataset_id: str, relative: str):
    directory = root / safe_slug(dataset_id)
    metadata = json.loads((directory / "openjensen.json").read_text())
    if not metadata.get("finalized"):
        raise ValueError("Finalize the dataset before importing it into a project")
    posix = PurePosixPath(relative)
    if (
        not relative
        or posix.is_absolute()
        or "\\" in relative
        or any(part in ("..", ".") or part.startswith(".") for part in posix.parts)
    ):
        raise ValueError("Invalid dataset file path")
    target = directory / Path(*posix.parts)
    if (
        directory.is_symlink()
        or any(part.is_symlink() for part in [target, *target.parents] if part != root.parent)
        or not target.is_file()
        or not target.resolve().is_relative_to(directory.resolve())
    ):
        raise ValueError("Dataset file missing or unsafe")
    return target


def manifest(root, dataset_id):
    directory = root / safe_slug(dataset_id)
    files = []
    for candidate in sorted(directory.rglob("*")):
        if candidate.is_file():
            relative = candidate.relative_to(directory).as_posix()
            if any(part.startswith(".") for part in candidate.relative_to(directory).parts):
                continue
            target = dataset_file(root, dataset_id, relative)
            hasher = hashlib.sha256()
            with target.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
            files.append(
                {"path": relative, "size": target.stat().st_size, "sha256": hasher.hexdigest()}
            )
    if not files:
        raise ValueError("Dataset not found or empty")
    return {
        "id": dataset_id,
        "name": dataset_id,
        "dataset_id": dataset_id,
        "files": files,
        "total_bytes": sum(entry["size"] for entry in files),
    }
