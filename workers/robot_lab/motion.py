"""Bounded movement to the pose captured by half-turn calibration."""

import math
import time

from hardware import JOINTS, HardwareError

MIDDLE_TICKS = 2047
TICKS_PER_DEGREE = 4095 / 360


def middle_targets(calibration):
    for joint in JOINTS:
        entry = calibration[joint]
        if not entry["range_min"] + 8 <= MIDDLE_TICKS <= entry["range_max"] - 8:
            raise ValueError(
                f"Recorded middle for {joint.replace('_', ' ')} is outside its usable range"
            )
    return dict.fromkeys(JOINTS, MIDDLE_TICKS)


def tracking_tolerance(joint, calibration):
    span = calibration[joint]["range_max"] - calibration[joint]["range_min"]
    return span * 0.05 if joint == "gripper" else 5 * TICKS_PER_DEGREE


def jog_target(joint, delta, raw, commanded, calibration):
    entry = calibration[joint]
    scale = (
        (entry["range_max"] - entry["range_min"]) / 100
        if joint == "gripper"
        else TICKS_PER_DEGREE
    )
    # A lagging encoder must not reset the target on each click or reverse it.
    # Truncate magnitude so quantization never exceeds the requested step.
    target = commanded[joint] + math.trunc(delta * scale)
    target = max(entry["range_min"] + 8, min(entry["range_max"] - 8, target))
    if abs(target - raw[joint]) > tracking_tolerance(joint, calibration):
        raise ValueError(
            f"{joint.replace('_', ' ').capitalize()} has not reached its target; "
            "wait before another step. If it stalls or shakes, stop and inspect the joint"
        )
    return target


def check_tracking(raw, commanded, calibration):
    for joint in JOINTS:
        tolerance = tracking_tolerance(joint, calibration)
        if abs(raw[joint] - commanded[joint]) > tolerance:
            raise HardwareError(
                f"{joint.replace('_', ' ').capitalize()} did not hold its commanded position "
                f"(measured {raw[joint]}, target {commanded[joint]} ticks). "
                "Motion stopped; support the arm and inspect this joint before restarting"
            )


class MiddleMotion:
    def __init__(self, initial, calibration):
        self.initial = dict(initial)
        self.targets = middle_targets(calibration)
        # Four degrees/s, or four percent/s for the gripper. Advance by elapsed
        # time, capped per tick: a delayed thread must never catch up with a jump.
        rates = {
            j: (calibration[j]["range_max"] - calibration[j]["range_min"]) * 0.04
            if j == "gripper"
            else 4 * TICKS_PER_DEGREE
            for j in JOINTS
        }
        self.duration = max(abs(self.targets[j] - initial[j]) / rates[j] for j in JOINTS)
        self.elapsed = 0.0
        self.last_time = time.monotonic()
        self.deadline = self.last_time + self.duration + 10
        self.progress = 0.0

    def advance(self):
        now = time.monotonic()
        if now > self.deadline:
            raise HardwareError("Move to recorded middle timed out; motion stopped")
        self.elapsed += max(0.0, min(0.1, now - self.last_time))
        self.last_time = now
        self.progress = min(1.0, self.elapsed / self.duration) if self.duration else 1.0
        return {
            j: round(self.initial[j] + (self.targets[j] - self.initial[j]) * self.progress)
            for j in JOINTS
        }
