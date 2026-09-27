"""Explicit geometric task scoring, independent of policy and robot calibration.

Scoring uses the entire object's conservative USD bounds, live rigid-body poses
and velocities at every control tick. It never infers a grasp from joint motion.
"""

import hashlib
import itertools
import json
import math
import re
from dataclasses import asdict, dataclass

from sim_worker.rollout.contracts import ObjectState, Observation

_PRIM = re.compile(r"(/[A-Za-z_][A-Za-z_0-9]*)+")
_EPS = 1e-7


def _number(value, name, lower, upper):
    if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} must be finite in [{lower}, {upper}]")
    return float(value)


def _vector(value, length, name):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} numbers")
    return tuple(_number(item, name, -1e6, 1e6) for item in value)


@dataclass(frozen=True)
class EvaluationSpec:
    task: str
    object_prim: str
    support_height_m: float
    lift_clearance_m: float
    lift_hold_seconds: float
    settle_seconds: float
    maximum_linear_speed_m_s: float
    maximum_angular_speed_rad_s: float
    placement_min_m: tuple[float, float, float] | None
    placement_max_m: tuple[float, float, float] | None

    @classmethod
    def parse(cls, value):
        keys = set(cls.__dataclass_fields__)
        if not isinstance(value, dict) or set(value) != keys:
            raise ValueError(f"evaluation requires exactly: {', '.join(sorted(keys))}")
        if not isinstance(value["task"], str) or value["task"] not in {"lift_hold", "lift_place"}:
            raise ValueError("evaluation.task must be lift_hold or lift_place")
        if not isinstance(value["object_prim"], str) or not _PRIM.fullmatch(value["object_prim"]):
            raise ValueError("evaluation.object_prim must be one absolute USD prim path")
        result = dict(value)
        for name, lower, upper in (
            ("support_height_m", -1000, 1000),
            ("lift_clearance_m", 0.001, 10),
            ("lift_hold_seconds", 0.001, 600),
            ("settle_seconds", 0.001, 600),
            ("maximum_linear_speed_m_s", 0, 100),
            ("maximum_angular_speed_rad_s", 0, 100),
        ):
            result[name] = _number(value[name], name, lower, upper)
        for name in ("placement_min_m", "placement_max_m"):
            if value["task"] == "lift_place":
                result[name] = _vector(value[name], 3, name)
            elif value[name] is not None:
                raise ValueError("lift_hold must not declare a placement region")
        if value["task"] == "lift_place" and any(
            lo >= hi for lo, hi in zip(result["placement_min_m"], result["placement_max_m"])
        ):
            raise ValueError("Placement region must have positive extent on every axis")
        return cls(**result)

    @property
    def digest(self):
        return hashlib.sha256(
            json.dumps(
                asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()


def world_bounds(body: ObjectState):
    """Transform all eight local bounding-box corners using an xyzw quaternion."""
    position = _vector(body.position_m, 3, "object position")
    quaternion = _vector(body.orientation_xyzw, 4, "object quaternion")
    norm = math.sqrt(sum(item * item for item in quaternion))
    if abs(norm - 1.0) > 1e-5:
        raise ValueError("Object quaternion must be normalized (xyzw)")
    x, y, z, w = (item / norm for item in quaternion)
    lo = _vector(body.local_min_m, 3, "object minimum")
    hi = _vector(body.local_max_m, 3, "object maximum")
    if any(a >= b for a, b in zip(lo, hi)):
        raise ValueError("Object bounds must have positive extent")
    rotation = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    points = [
        tuple(position[i] + sum(rotation[i][j] * corner[j] for j in range(3)) for i in range(3))
        for corner in itertools.product(*zip(lo, hi))
    ]
    return tuple(min(p[i] for p in points) for i in range(3)), tuple(
        max(p[i] for p in points) for i in range(3)
    )


class EpisodeEvaluator:
    def __init__(self, spec: EvaluationSpec, *, fps: int, steps: int):
        # Validate direct dataclass callers too; no permissive alternate entry point.
        self.spec = EvaluationSpec.parse(asdict(spec))
        if (
            type(fps) is not int
            or not 1 <= fps <= 120
            or type(steps) is not int
            or not 1 <= steps <= 72000
        ):
            raise ValueError("Evaluation requires a bounded positive control rate and horizon")
        self.fps, self.steps = fps, steps
        self.samples = 0
        self.episode = None
        self.geometry = None
        self.lift_start = None
        self.settle_start = None
        self.lift_met = False
        self.peak_clearance = -math.inf
        self.last = None

    def observe(self, observation: Observation) -> dict:
        if (
            type(observation.step) is not int
            or observation.step != self.samples
            or observation.step > self.steps
            or type(observation.sim_time) not in (int, float)
            or not math.isfinite(observation.sim_time)
            or abs(observation.sim_time - observation.step / self.fps) > _EPS
            or not isinstance(observation.episode_id, str)
            or not observation.episode_id
            or (self.episode is not None and observation.episode_id != self.episode)
        ):
            raise ValueError("Evaluation requires every ordered tick from one episode")
        body = observation.object_state
        if not isinstance(body, ObjectState) or body.prim_path != self.spec.object_prim:
            raise ValueError("Evaluation requires live state for its exact declared object")
        lower, upper = world_bounds(body)
        linear = math.hypot(*_vector(body.linear_velocity_m_s, 3, "linear velocity"))
        angular = math.hypot(*_vector(body.angular_velocity_rad_s, 3, "angular velocity"))
        geometry = (body.local_min_m, body.local_max_m)
        if self.geometry is not None and geometry != self.geometry:
            raise ValueError("Object geometry changed during evaluation")
        self.geometry = geometry
        clearance = lower[2] - self.spec.support_height_m
        lifted = clearance >= self.spec.lift_clearance_m
        inside = self.spec.task == "lift_place" and all(
            lo <= actual_lo <= actual_hi <= hi
            for lo, actual_lo, actual_hi, hi in zip(
                self.spec.placement_min_m, lower, upper, self.spec.placement_max_m
            )
        )
        if self.samples == 0 and (lifted or inside):
            raise ValueError("Object already meets lift or placement geometry at reset")
        self.episode = observation.episode_id
        self.peak_clearance = max(self.peak_clearance, clearance)
        now = observation.sim_time
        self.lift_start = (now if self.lift_start is None else self.lift_start) if lifted else None
        lift_duration = 0.0 if self.lift_start is None else now - self.lift_start
        current_lift_met = lifted and lift_duration + _EPS >= self.spec.lift_hold_seconds
        self.lift_met |= current_lift_met
        settled = (
            (lifted if self.spec.task == "lift_hold" else inside)
            and linear <= self.spec.maximum_linear_speed_m_s
            and angular <= self.spec.maximum_angular_speed_rad_s
            and (current_lift_met if self.spec.task == "lift_hold" else self.lift_met)
        )
        self.settle_start = (
            (now if self.settle_start is None else self.settle_start) if settled else None
        )
        settle_duration = 0.0 if self.settle_start is None else now - self.settle_start
        self.last = {
            "step": observation.step,
            "sim_time": now,
            "world_min_m": lower,
            "world_max_m": upper,
            "whole_object_clearance_m": clearance,
            "inside_placement_region": inside,
            "linear_speed_m_s": linear,
            "angular_speed_rad_s": angular,
            "lift_hold_seconds": lift_duration,
            "lift_met": self.lift_met,
            "current_lift_hold_met": current_lift_met,
            "terminal_settle_seconds": settle_duration,
            "geometric_success": settled and settle_duration + _EPS >= self.spec.settle_seconds,
        }
        self.samples += 1
        return dict(self.last)

    def finish(self) -> dict:
        if self.samples != self.steps + 1:
            raise ValueError("Cannot score an incomplete episode")
        return {
            "schema_version": 1,
            "status": "measured",
            "episode_id": self.episode,
            "predicate": "conservative_object_bounds_v1",
            "criteria": asdict(self.spec),
            "criteria_sha256": self.spec.digest,
            "samples": self.samples,
            "control_fps": self.fps,
            "geometric_success": self.last["geometric_success"],
            "peak_whole_object_clearance_m": self.peak_clearance,
            "final": self.last,
            "quality_validated": False,
            "grasp_and_release_verified": False,
            "scope": (
                "sampled geometry and velocity in the configured simulation; "
                "no physical transfer claim"
            ),
        }
