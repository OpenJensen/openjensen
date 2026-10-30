"""Private atomic checkpoints for unpowered calibration; never opens hardware."""

import hashlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path

from dataset import safe_slug
from hardware import IDS, JOINTS


def file_hash(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def validate_draft(value):
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("Unsupported calibration checkpoint")
    safe_slug(value["robot_id"])
    if value["role"] not in {"follower", "leader"} or not isinstance(value["port"], str):
        raise ValueError("Invalid calibration checkpoint identity")
    state = value["state"]
    if state["phase"] not in {"center", "ranges"}:
        raise ValueError("Invalid calibration checkpoint stage")
    original = value["original"]
    if set(original) != set(JOINTS):
        raise ValueError("Incomplete calibration checkpoint")
    for joint, entry in original.items():
        if entry["id"] != IDS[joint] or entry["drive_mode"] != 0:
            raise ValueError("Invalid checkpoint motor identity")
        for key, low, high in (
            ("homing_offset", -2047, 2047),
            ("range_min", 0, 4095),
            ("range_max", 0, 4095),
        ):
            number = entry[key]
            if type(number) is not int or not low <= number <= high:
                raise ValueError("Invalid checkpoint motor configuration")
        if entry["range_min"] > entry["range_max"]:
            raise ValueError("Invalid checkpoint limits")
    if state["phase"] == "ranges":
        for key, low, high in (("offsets", -2047, 2047), ("mins", 0, 4095), ("maxes", 0, 4095)):
            if set(state[key]) != set(JOINTS) or any(
                type(n) is not int or not low <= n <= high for n in state[key].values()
            ):
                raise ValueError("Invalid captured calibration values")
        if any(state["mins"][j] > state["maxes"][j] for j in JOINTS):
            raise ValueError("Invalid captured joint ranges")
    elif any(state[key] for key in ("offsets", "mins", "maxes")):
        raise ValueError("Unexpected values before middle capture")
    digest = value.get("profile_hash")
    if digest is not None and (not isinstance(digest, str) or len(digest) != 64):
        raise ValueError("Invalid saved profile checksum")
    return value


class CalibrationDraft:
    def __init__(self, data_dir):
        self.path = Path(data_dir) / "calibration-draft.json"
        self.last_write = 0.0
        self.lock = threading.RLock()

    def read(self):
        with self.lock:
            if not self.path.exists():
                return None
            if self.path.is_symlink() or self.path.stat().st_size > 65536:
                raise ValueError("Invalid calibration checkpoint file")
            return validate_draft(json.loads(self.path.read_text()))

    def write(self, value, *, force=False):
        with self.lock:
            now = time.monotonic()
            if not force and now - self.last_write < 1:
                return
            validate_draft(value)
            descriptor, name = tempfile.mkstemp(
                prefix="calibration-draft-", suffix=".pending", dir=self.path.parent
            )
            pending = Path(name)
            try:
                with os.fdopen(descriptor, "w") as stream:
                    json.dump(value, stream, allow_nan=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                pending.replace(self.path)
            finally:
                pending.unlink(missing_ok=True)
            self.last_write = now

    def archive(self, reason):
        with self.lock:
            if self.path.exists():
                folder = self.path.parent / "calibration-history"
                folder.mkdir(mode=0o700, exist_ok=True)
                self.path.replace(folder / f"{time.time_ns()}-{reason}.json")
