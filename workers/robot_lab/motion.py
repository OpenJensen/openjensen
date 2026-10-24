"""Calibrated target limits and encoder arrival for LeRobot position control."""

import math

from hardware import normalized

MIDDLE_TICKS = 2047
TICKS_PER_DEGREE = 4095 / 360


def middle_targets(calibration):
    for joint in calibration:
        entry = calibration[joint]
        if not entry["range_min"] + 8 <= MIDDLE_TICKS <= entry["range_max"] - 8:
            raise ValueError(
                f"Recorded middle for {joint.replace('_', ' ')} is outside its usable range"
            )
    return dict.fromkeys(calibration, MIDDLE_TICKS)


def step_target(joint, delta, commanded, calibration):
    entry = calibration[joint]
    scale = (
        (entry["range_max"] - entry["range_min"]) / 100 if joint == "gripper" else TICKS_PER_DEGREE
    )
    # A lagging encoder must not reset the target on each click or reverse it.
    # Truncate magnitude so quantization never exceeds the requested step.
    target = commanded[joint] + math.trunc(delta * scale)
    target = max(entry["range_min"] + 8, min(entry["range_max"] - 8, target))
    return target


class MiddleMotion:
    """Observe arrival at the requested middle; send a full LeRobot goal once."""

    def __init__(self, initial, calibration, tolerance=2):
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(tolerance)
            or not 0.5 <= tolerance <= 5
        ):
            raise ValueError("Middle tolerance must be between 0.5 and 5 degrees / percent")
        self.targets = middle_targets(calibration)
        self.initial = dict(initial)
        self.arrival_limits = {
            joint: (calibration[joint]["range_max"] - calibration[joint]["range_min"])
            * tolerance
            / 100
            if joint == "gripper"
            else tolerance * TICKS_PER_DEGREE
            for joint in initial
        }
        self.progress = 0.0
        self.sent = False

    def arrived(self, raw):
        remaining = max(
            abs(raw[j] - self.targets[j]) / max(1, abs(self.initial[j] - self.targets[j]))
            for j in self.targets
        )
        self.progress = max(0.0, min(1.0, 1.0 - remaining))
        arrived = all(abs(raw[j] - self.targets[j]) <= self.arrival_limits[j] for j in self.targets)
        if arrived:
            self.progress = 1.0
        return arrived


def control_limits(calibration):
    return {
        joint: {
            "min": normalized(joint, entry["range_min"] + 8, calibration),
            "max": normalized(joint, entry["range_max"] - 8, calibration),
        }
        for joint, entry in calibration.items()
    }
