"""Private, versioned calibration snapshots on the robot computer."""

import hashlib
import json
import math
import time
from pathlib import Path

from assembled_session import durable_json
from dataset import safe_slug


class CalibrationProfiles:
    def __init__(self, data_dir, validate):
        self.root = Path(data_dir) / "calibration-profiles"
        self.validate = validate

    @staticmethod
    def key(robot_id, role, calibration):
        return hashlib.sha256(
            json.dumps([robot_id, role, calibration], sort_keys=True).encode()
        ).hexdigest()

    def read(self, path):
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 65536:
            raise ValueError("Calibration must be a regular file smaller than 64 KiB")
        return json.loads(path.read_text())

    def save(self, robot_id, role, calibration, source):
        safe_slug(robot_id)
        self.validate(calibration)
        key = self.key(robot_id, role, calibration)
        if self.root.is_symlink():
            raise ValueError("Calibration history must not be a symbolic link")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self.root / (key + ".json")
        if path.is_symlink():
            raise ValueError("Calibration history file must not be a symbolic link")
        if not path.exists():
            durable_json(
                path,
                {
                    "version": 1,
                    "robot_id": robot_id,
                    "role": role,
                    "calibration": calibration,
                    "saved_at": time.time(),
                    "source": source,
                },
            )
        return key

    def entries(self, calibration_root, role):
        entries = {}
        if self.root.is_symlink():
            raise ValueError("Calibration history must not be a symbolic link")
        for path in self.root.glob("*.json"):
            try:
                record = self.read(path)
                if record.get("version") != 1 or record.get("role") != role:
                    continue
                if not isinstance(record.get("saved_at"), (int, float)) or not math.isfinite(
                    record["saved_at"]
                ):
                    continue
                safe_slug(record["robot_id"])
                calibration = self.validate(record["calibration"])
                key = self.key(record["robot_id"], role, calibration)
                if path.stem != key:
                    continue
                entries[key] = {**record, "key": key}
            except ValueError, TypeError, KeyError, OSError, AttributeError:
                continue
        folder = "robots/so_follower" if role == "follower" else "teleoperators/so_leader"
        # Include existing LeRobot files and automatic backups without moving them.
        for path in (Path(calibration_root) / folder).glob("*.json"):
            try:
                robot_id = safe_slug(path.stem.split(".backup-", 1)[0])
                calibration = self.validate(self.read(path))
                key = self.key(robot_id, role, calibration)
                if key not in entries:
                    entries[key] = {
                        "key": key,
                        "robot_id": robot_id,
                        "role": role,
                        "calibration": calibration,
                        "saved_at": path.stat().st_mtime,
                        "source": "backup" if ".backup-" in path.stem else "saved profile",
                    }
            except ValueError, TypeError, KeyError, OSError, AttributeError:
                continue
        return sorted(entries.values(), key=lambda item: item["saved_at"], reverse=True)
