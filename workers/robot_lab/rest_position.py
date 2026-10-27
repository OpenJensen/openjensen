"""Calibration-bound rest poses, stored on the robot computer."""

import hashlib
import json

from dataset import atomic_json, safe_slug
from hardware import JOINTS


def calibration_digest(calibration):
    return hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()


class RestPositions:
    def __init__(self, data_dir):
        self.root = data_dir / "rest-positions"

    def validate(self, targets, calibration):
        if not isinstance(targets, dict) or set(targets) != set(JOINTS):
            raise ValueError("A rest position requires all six joints")
        for joint, value in targets.items():
            entry = calibration[joint]
            if (
                type(value) is not int
                or not entry["range_min"] + 8 <= value <= entry["range_max"] - 8
            ):
                raise ValueError("Rest position must be inside the calibrated joint limits")
        return dict(targets)

    def load(self, robot_id, calibration):
        path = self.root / (safe_slug(robot_id) + ".json")
        try:
            if path.is_symlink() or path.stat().st_size > 65536:
                return None
            saved = json.loads(path.read_text())
            if saved["calibration"] != calibration_digest(calibration):
                return None
            return self.validate(saved["targets"], calibration)
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def save(self, robot_id, targets, calibration):
        targets = self.validate(targets, calibration)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        atomic_json(
            self.root / (safe_slug(robot_id) + ".json"),
            {
                "calibration": calibration_digest(calibration),
                "targets": targets,
            },
        )
        return targets
