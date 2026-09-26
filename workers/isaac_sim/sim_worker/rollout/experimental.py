"""Explicit motion bounds for experiments using unverified joint mappings."""

from collections.abc import Callable
from dataclasses import dataclass

from sim_worker.rollout.config import number

MAX_SPEED_RAD_S = 2.0


@dataclass(frozen=True)
class LimitedTarget:
    target: tuple[float, ...]
    limit_clipped: tuple[bool, ...]
    speed_clipped: tuple[bool, ...]


class MotionGuard:
    def __init__(self, limits: Callable[[], tuple[tuple[float, float], ...]], fps: int):
        if type(fps) is not int or fps <= 0:
            raise ValueError("Motion guard needs a positive integer control rate")
        self._limits = limits
        self._max_step = MAX_SPEED_RAD_S / fps

    def constrain(self, target: tuple[float, ...], state: tuple[float, ...]) -> LimitedTarget:
        # Limits come from the initialized simulation articulation, in policy order.
        limits = self._limits()
        if not limits or len(target) != len(limits) or len(state) != len(limits):
            raise ValueError("Motion guard dimensions differ from articulation limits")
        applied, limit_clips, speed_clips = [], [], []
        for requested, measured, bounds in zip(target, state, limits, strict=True):
            requested = number(requested, "requested joint target")
            measured = number(measured, "measured joint state")
            lower, upper = (number(value, "joint limit") for value in bounds)
            if lower > upper:
                raise ValueError("Joint limits must be ordered")
            bounded = max(lower, min(upper, requested))
            stepped = max(measured - self._max_step, min(measured + self._max_step, bounded))
            # Preserve physical limits even if a measured state overshoots slightly.
            applied.append(max(lower, min(upper, stepped)))
            limit_clips.append(bounded != requested)
            speed_clips.append(stepped != bounded)
        return LimitedTarget(tuple(applied), tuple(limit_clips), tuple(speed_clips))
