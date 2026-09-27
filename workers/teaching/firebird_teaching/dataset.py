"""Finalize selected lossless captures using the real pinned LeRobot v3 writer."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import stat
import tempfile
from pathlib import Path
from unittest.mock import patch

from .contracts import IDENTITY, LINEAGE, decode, finite, integer, text
from .journal import atomic_new

FLOAT32_MAX = 3.4028234663852886e38

UPSTREAM = "e595b7902714ba51f91e47523f66f89c5181b649"
WRITER_SOURCES = {
    "datasets/lerobot_dataset.py": "8f6a83a46d82b4401f58e859844c9b265695368ee50af7772143d042e01a84ec",  # noqa: E501
    "datasets/dataset_writer.py": "b1ce93c7e668adfea081075a0fd02d44d50aae81c83096e105f6ce0346e35a78",  # noqa: E501
    "datasets/video_utils.py": "b7f0e94dc547a269198ef19f1fd364374c32f4be8eb5cf5cc8505243c9bb79ed",
    "configs/video.py": "8575fd0baabbc1c35acb7491871e4b146b9981c92686f4bc95721079f038f079",
}


def verify_writer():
    if importlib.metadata.version("lerobot") != "0.6.2":
        raise ValueError(
            "Recording conversion requires the pinned native LeRobot 0.6.2 environment"
        )
    package = Path(importlib.util.find_spec("lerobot").origin).parent
    for name, expected in WRITER_SOURCES.items():
        if hashlib.sha256(_read(package / name, 2 * 1024**2)).hexdigest() != expected:
            raise ValueError("LeRobot writer source differs from the approved native pin: " + name)


def _no_download(*args, **kwargs):
    raise ValueError("Local teaching dataset readback must never download missing files")


def _open_regular(path: Path):
    """No-follow traversal keeps ancestor replacement from redirecting reads."""
    if os.open not in os.supports_dir_fd or not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("Secure capture conversion currently requires POSIX directory descriptors")
    absolute = path.absolute()
    if ".." in absolute.parts:
        raise ValueError("Capture paths must not contain parent traversal")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    directory = os.open(absolute.anchor, flags | os.O_DIRECTORY)
    try:
        for component in absolute.parts[1:-1]:
            next_directory = os.open(component, flags | os.O_DIRECTORY, dir_fd=directory)
            os.close(directory)
            directory = next_directory
        fd = os.open(absolute.name, flags, dir_fd=directory)
    except OSError:
        raise ValueError("Capture path is missing, linked or unreadable") from None
    finally:
        os.close(directory)
    stream = os.fdopen(fd, "rb")
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        stream.close()
        raise ValueError("Capture file must be regular")
    return stream


def _stamp(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _read(path: Path, limit: int) -> bytes:
    with _open_regular(path) as stream:
        before = os.fstat(stream.fileno())
        if before.st_size > limit:
            raise ValueError("Capture file exceeds its bound")
        raw = stream.read(min(limit, before.st_size) + 1)
        if len(raw) != before.st_size or _stamp(before) != _stamp(os.fstat(stream.fileno())):
            raise ValueError("Capture file changed during read")
        return raw


def _inventory(root: Path) -> dict:
    result = {}
    total = 0
    entries = 0
    for path in root.rglob("*"):
        entries += 1
        if entries > 50000:
            raise ValueError("Capture inventory exceeds entry limit")
        if path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise ValueError("Capture contains a symlink or special file")
        if path.is_file():
            with _open_regular(path) as stream:
                before = os.fstat(stream.fileno())
                total += before.st_size
                if before.st_size > 2 * 1024**3 or total > 8 * 1024**3:
                    raise ValueError("Capture exceeds file or total byte bound")
                digest = hashlib.sha256()
                remaining = before.st_size
                while remaining:
                    part = stream.read(min(1024**2, remaining))
                    if not part:
                        raise ValueError("Capture shrank during inventory")
                    digest.update(part)
                    remaining -= len(part)
                if stream.read(1) or _stamp(before) != _stamp(os.fstat(stream.fileno())):
                    raise ValueError("Capture changed during inventory")
                result[path.relative_to(root).as_posix()] = digest.hexdigest()
    return result


def inspect_capture(root: Path, episodes: list[str]) -> tuple[dict, list[dict], dict]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Capture must be a real directory")
    if not episodes or len(set(episodes)) != len(episodes) or len(episodes) > 100:
        raise ValueError("Select 1..100 distinct finalized episodes explicitly")
    before = _inventory(root)

    def read(path, limit):
        raw = _read(path, limit)
        if hashlib.sha256(raw).hexdigest() != before.get(path.relative_to(root).as_posix()):
            raise ValueError("Capture content differs from the initial inventory")
        return raw

    def read_json(path):
        return decode(read(path, 2 * 1024**2), limit=2 * 1024**2)

    meta = read_json(root / "session.json")
    if (
        meta.get("schema_version") != 1
        or meta.get("controller") != "joint_position_targets"
        or meta.get("state_units") != "radians"
        or meta.get("action_units") != "radians"
        or meta.get("timebase") != "simulation_seconds"
    ):
        raise ValueError("Unsupported teaching coordinate/time contract")
    names = meta.get("joint_names")
    if (
        not isinstance(names, list)
        or not 1 <= len(names) <= 32
        or len(set(names)) != len(names)
        or any(not isinstance(n, str) or not n.isidentifier() for n in names)
    ):
        raise ValueError("Invalid native joint order")
    width, height = (integer(meta.get(k), k, 2, 1920) for k in ("width", "height"))
    fps = integer(meta.get("fps"), "fps", 1, 60)
    lineage = text(meta.get("lineage_group"), "lineage_group", 128)
    if not LINEAGE.fullmatch(lineage):
        raise ValueError("Invalid lineage group identifier")
    if (
        not isinstance(meta.get("scene_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", meta["scene_sha256"]) is None
    ):
        raise ValueError("Missing scene identity")
    if meta.get("origin") not in {"recorded", "synthetic"}:
        raise ValueError("Capture origin must distinguish real recordings from generated fixtures")
    if width % 2 or height % 2:
        raise ValueError("Video dimensions must be even")
    selected = []
    for ident in episodes:
        if not isinstance(ident, str) or not IDENTITY.fullmatch(ident):
            raise ValueError("Invalid selected episode identity")
        folder = root / ident
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError("Selected episode is missing")
        receipt = read_json(folder / "episode.json")
        count = integer(receipt.get("frames"), "frames", 1, 3600)
        if (
            receipt.get("schema_version") != 1
            or receipt.get("episode_id") != ident
            or receipt.get("finalized") is not True
            or receipt.get("outcome") not in {"unknown", "operator_reported_failure"}
        ):
            raise ValueError("Episode is unfinalized or makes an unsupported outcome claim")
        if receipt.get("termination") not in {"finish", "reset", "step_limit", "shutdown"}:
            raise ValueError("Unknown episode termination")
        events_raw = read(folder / "events.jsonl", 2 * 1024**2)
        if not events_raw.endswith(b"\n") or hashlib.sha256(events_raw).hexdigest() != receipt.get(
            "events_sha256"
        ):
            raise ValueError("Intervention journal is torn or changed")
        events = [decode(line) for line in events_raw.splitlines()]
        if events != receipt.get("events") or not 1 <= len(events) <= 512:
            raise ValueError("Intervention journal differs from final receipt")
        command_ids = set()
        command_events = {}
        for event in events:
            from .contracts import Command

            command = Command.parse(
                {
                    "command_id": event.get("command_id"),
                    "session_id": meta.get("session_id"),
                    "episode_id": ident,
                    "expected_revision": 0,
                    "operation": event.get("operation"),
                    "arguments": event.get("arguments"),
                }
            )
            if command.command_id in command_ids or event.get("source") != "operator_command":
                raise ValueError("Invalid intervention identity/source")
            command_ids.add(command.command_id)
            command_events[command.command_id] = event
            step = integer(event.get("step"), "intervention step", 0, count)
            if abs(finite(event.get("sim_time"), "intervention sim_time") - step / fps) > 1e-6:
                raise ValueError("Invalid intervention simulation timestamp")
            integer(event.get("accepted_monotonic_ns"), "intervention acceptance", 1, 2**63 - 1)
        if {p.name for p in (folder / "frames").iterdir()} != {
            f"{n:06d}.rgb" for n in range(count)
        }:
            raise ValueError("Unrecorded or missing source image")
        raw = read(folder / "trajectory.jsonl", 16 * 1024**2)
        if hashlib.sha256(raw).hexdigest() != receipt.get("trajectory_sha256") or not raw.endswith(
            b"\n"
        ):
            raise ValueError("Trajectory is torn or changed")
        rows = [decode(line) for line in raw.splitlines()]
        if len(rows) != count:
            raise ValueError("Trajectory count differs from final receipt")
        if receipt.get("bytes") != len(raw) + count * width * height * 3:
            raise ValueError("Capture byte count differs from final receipt")
        for step, row in enumerate(rows):
            if (
                row.get("episode_id") != ident
                or type(row.get("step")) is not int
                or row["step"] != step
            ):
                raise ValueError("Noncontiguous episode/control frame order")
            integer(row.get("applied_monotonic_ns"), "action application time", 1, 2**63 - 1)
            for key, expected in [("sim_time", step / fps), ("next_sim_time", (step + 1) / fps)]:
                if abs(finite(row.get(key), key) - expected) > 1e-6:
                    raise ValueError("Capture simulation timestamp differs from fps")
            for key in [
                "state_rad",
                "requested_target_rad",
                "applied_target_rad",
                "next_state_rad",
            ]:
                values = row.get(key)
                if not isinstance(values, list) or len(values) != len(names):
                    raise ValueError("Native state/action dimension mismatch")
                for value in values:
                    if abs(finite(value, key)) > FLOAT32_MAX:
                        raise ValueError(
                            "State/action value is not finite-representable in float32"
                        )
            for key in ["joint_limit_clipped", "speed_limit_clipped"]:
                if (
                    not isinstance(row.get(key), list)
                    or len(row[key]) != len(names)
                    or any(type(v) is not bool for v in row[key])
                ):
                    raise ValueError("Invalid motion guard provenance")
            text(row.get("task"), "task")
            event = command_events.get(row.get("command_id"))
            if event is None or event["operation"] not in {"start", "correct"}:
                raise ValueError("Frame has no motion-target command provenance")
            if event["step"] > step or event["accepted_monotonic_ns"] > row["applied_monotonic_ns"]:
                raise ValueError("Frame references a future motion-target command")
            if step and rows[step - 1]["next_state_rad"] != row["state_rad"]:
                raise ValueError("Adjacent simulator observations are discontinuous")
            if row.get("action_source") != "operator_joint_position":
                raise ValueError("Unknown action supervision source")
            expected_name = f"frames/{step:06d}.rgb"
            if row.get("frame") != expected_name:
                raise ValueError("Unexpected frame path")
            pixels = read(folder / expected_name, width * height * 3)
            if len(pixels) != width * height * 3 or hashlib.sha256(pixels).hexdigest() != row.get(
                "rgb_sha256"
            ):
                raise ValueError("RGB frame truncated or changed")
        selected.append({"id": ident, "rows": rows, "receipt": receipt, "lineage_group": lineage})
    if _inventory(root) != before:
        raise ValueError("Capture changed during admission")
    return meta, selected, before


def verify_statistics(root: Path):
    value = decode(_read(root / "meta/stats.json", 16 * 1024**2), limit=16 * 1024**2)

    def visit(node):
        if isinstance(node, dict):
            for item in node.values():
                visit(item)
        elif isinstance(node, list):
            for item in node:
                visit(item)
        elif type(node) not in {int, float} or not math.isfinite(node):
            raise ValueError("LeRobot statistics must contain finite numeric values")

    visit(value)


def read_local_dataset(repo_id: str, root: Path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Upstream reader and metadata reader each have independent Hub fallbacks.
    # This isolated converter is single-threaded; none may repair local output remotely.
    with (
        patch("lerobot.datasets.lerobot_dataset.snapshot_download", _no_download),
        patch("lerobot.datasets.lerobot_dataset.get_safe_version", _no_download),
        patch("lerobot.datasets.dataset_metadata.snapshot_download", _no_download),
        patch("lerobot.datasets.dataset_metadata.get_safe_version", _no_download),
    ):
        return LeRobotDataset(repo_id=repo_id, root=root, video_backend="pyav", token=False)


def convert(
    root: Path, output: Path, episodes: list[str], *, repo_id: str = "local/teaching"
) -> dict:
    if os.path.lexists(output):
        raise FileExistsError("Dataset output already exists")
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Dataset output must be outside capture storage")
    verify_writer()
    meta, selected, before = inspect_capture(root, episodes)
    # Lazy heavy imports: the control/Isaac process never imports the writer stack.
    import numpy as np
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    n = len(meta["joint_names"])
    features = {
        key: {"dtype": "float32", "shape": (n,), "names": meta["joint_names"]}
        for key in [
            "observation.state",
            "action",
            "teaching.requested_action",
            "teaching.next_state",
        ]
    }
    features["observation.images.front"] = {
        "dtype": "video",
        "shape": (meta["height"], meta["width"], 3),
        "names": ["height", "width", "channels"],
    }
    features["teaching.sim_time"] = {
        "dtype": "float64",
        "shape": (1,),
        "names": ["simulation_seconds"],
    }
    features["teaching.intervention"] = {
        "dtype": "bool",
        "shape": (1,),
        "names": ["operator_correction"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".teaching-convert-", dir=output.parent) as scratch:
        staged = Path(scratch) / "dataset"
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=staged,
            fps=meta["fps"],
            robot_type="isaac_joint_position",
            features=features,
            use_videos=True,
            video_backend="pyav",
            rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=18),
            image_writer_processes=0,
            image_writer_threads=0,
            encoder_threads=1,
        )
        counts = []
        lineage = []
        details = []
        try:
            for index, episode in enumerate(selected):
                receipt = episode["receipt"]
                corrections = {
                    e["command_id"]
                    for e in receipt.get("events", [])
                    if e.get("operation") == "correct"
                }
                for row in episode["rows"]:
                    pixels = _read(
                        root / episode["id"] / row["frame"], meta["width"] * meta["height"] * 3
                    )
                    if hashlib.sha256(pixels).hexdigest() != row["rgb_sha256"]:
                        raise ValueError("Capture frame changed during conversion")
                    dataset.add_frame(
                        {
                            "observation.images.front": np.frombuffer(
                                pixels, dtype=np.uint8
                            ).reshape(meta["height"], meta["width"], 3),
                            "observation.state": np.array(row["state_rad"], dtype=np.float32),
                            "action": np.array(row["applied_target_rad"], dtype=np.float32),
                            "teaching.requested_action": np.array(
                                row["requested_target_rad"], dtype=np.float32
                            ),
                            "teaching.next_state": np.array(
                                row["next_state_rad"], dtype=np.float32
                            ),
                            "teaching.sim_time": np.array([row["sim_time"]], dtype=np.float64),
                            "teaching.intervention": np.array(
                                [row.get("command_id") in corrections], dtype=np.bool_
                            ),
                            "task": row["task"],
                        }
                    )
                dataset.save_episode(parallel_encoding=False)
                counts.append(len(episode["rows"]))
                lineage.append(
                    {
                        "episode_index": index,
                        "origin": meta["origin"],
                        "lineage_group": episode["lineage_group"],
                    }
                )
                details.append(
                    {
                        "episode_index": index,
                        "capture_episode_id": episode["id"],
                        "reset_id": episode["id"],
                        "outcome": receipt["outcome"],
                        "termination": receipt["termination"],
                        "events": receipt.get("events", []),
                        "source_trajectory_sha256": receipt["trajectory_sha256"],
                    }
                )
        finally:
            dataset.finalize()
        verify_statistics(staged)
        # Read genuine writer output using upstream reader, including video decoding.
        loaded = read_local_dataset(repo_id, staged)
        if len(loaded) != sum(counts) or loaded.num_episodes != len(counts):
            raise ValueError("LeRobot readback frame/episode counts differ")
        offset = 0
        for count, episode in zip(counts, selected, strict=True):
            for local, original in enumerate(episode["rows"]):
                row = loaded[offset + local]
                for key in (
                    "action",
                    "observation.state",
                    "teaching.requested_action",
                    "teaching.next_state",
                    "teaching.sim_time",
                ):
                    if not np.isfinite(row[key].numpy()).all():
                        raise ValueError("LeRobot readback contains nonfinite state/action data")
                if abs(float(row["timestamp"]) - local / meta["fps"]) > 1e-5:
                    raise ValueError("LeRobot readback time mismatch")
                for key, source in [
                    ("action", "applied_target_rad"),
                    ("observation.state", "state_rad"),
                    ("teaching.requested_action", "requested_target_rad"),
                ]:
                    if not np.array_equal(
                        row[key].numpy(), np.array(original[source], dtype=np.float32)
                    ):
                        raise ValueError("LeRobot readback changed state/action values")
                if tuple(row["observation.images.front"].shape) != (
                    3,
                    meta["height"],
                    meta["width"],
                ):
                    raise ValueError("LeRobot readback video shape mismatch")
            offset += count
        atomic_new(
            staged / "meta/firebird-lineage.json", {"schema_version": 1, "episodes": lineage}
        )
        atomic_new(
            staged / "meta/firebird-demonstrations.json",
            {
                "schema_version": 1,
                "writer_upstream_revision": UPSTREAM,
                "lerobot_version": "0.6.2",
                "source_session_id": meta["session_id"],
                "controller": meta["controller"],
                "state_units": meta["state_units"],
                "action_units": meta["action_units"],
                "joint_order": meta["joint_names"],
                "timebase": meta["timebase"],
                "scene_sha256": meta["scene_sha256"],
                "scene_hash_scope": "root USD bytes; referenced assets not inventoried",
                "camera_prim": meta["camera_prim"],
                "action_column": "action",
                "requested_action_column": "teaching.requested_action",
                "task_success_verified": False,
                "episodes": details,
                "source_files": before,
                "video_encoding": {"codec": "h264", "crf": 18, "lossless_source_preserved": True},
            },
        )
        if _inventory(root) != before:
            raise ValueError("Capture changed during conversion/readback")
        if os.path.lexists(output):
            raise FileExistsError("Dataset output appeared during conversion")
        # Output parent is operator-owned; one converter owns this output name.
        staged.rename(output)
    return {
        "status": "finalized",
        "path": str(output),
        "episodes": len(counts),
        "frames": sum(counts),
        "readback_verified": True,
        "task_success_verified": False,
        "source_preserved": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--episode", action="append", required=True)
    parser.add_argument("--repo-id", default="local/teaching")
    args = parser.parse_args()
    print(
        json.dumps(convert(args.capture, args.output, args.episode, repo_id=args.repo_id), indent=2)
    )


if __name__ == "__main__":
    main()
