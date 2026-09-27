import hashlib
from bisect import bisect_right
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from sim_worker.rollout.config import fields, number, read_document
from sim_worker.rollout.contracts import CALIBRATION_VERSION


class CalibrationUse(Enum):
    VERIFIED = "verified"
    EXPERIMENTAL = "experimental"


@dataclass(frozen=True)
class _Curve:
    name: str
    sim: tuple[float, ...]
    policy: tuple[float, ...]


def _interpolate(value, source, target, name, use):
    value = number(value, name)
    if source[0] > source[-1]:
        source, target = source[::-1], target[::-1]
    if use is CalibrationUse.VERIFIED and (value < source[0] or value > source[-1]):
        raise ValueError(f"{name}={value} is outside calibrated range [{source[0]}, {source[-1]}]")
    # Experimental extrapolation follows the nearest endpoint segment.
    index = max(0, min(bisect_right(source, value) - 1, len(source) - 2))
    fraction = (value - source[index]) / (source[index + 1] - source[index])
    return number(target[index] + fraction * (target[index + 1] - target[index]), name)


class JointMap:
    def __init__(
        self, path: Path, joints: tuple[str, ...], use: CalibrationUse = CalibrationUse.VERIFIED
    ):
        data = fields(
            read_document(path), {"api_version", "status", "source", "joints"}, "calibration"
        )
        if not isinstance(use, CalibrationUse):
            raise ValueError("Calibration use must be an explicit CalibrationUse")
        statuses = (
            ("verified", "unverified") if use is CalibrationUse.EXPERIMENTAL else ("verified",)
        )
        if data["api_version"] != CALIBRATION_VERSION or data["status"] not in statuses:
            raise ValueError("Calibration must have the supported version and status: verified")
        if not isinstance(data["source"], str) or not data["source"].strip():
            raise ValueError("Calibration requires its recording/validation source")
        entries = fields(data["joints"], set(joints), "calibration.joints")
        self._curves = tuple(self._curve(name, entries[name]) for name in joints)
        self._digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self._status = data["status"]
        self._use = use

    @property
    def digest(self) -> str:
        return self._digest

    @property
    def status(self) -> str:
        return self._status

    @property
    def use(self) -> CalibrationUse:
        return self._use

    def _curve(self, name, values):
        fields(values, {"sim_rad", "policy"}, name)
        sim, policy = values["sim_rad"], values["policy"]
        if (
            not isinstance(sim, list)
            or not isinstance(policy, list)
            or len(sim) < 2
            or len(sim) != len(policy)
        ):
            raise ValueError(
                f"{name} needs equally sized calibration arrays with at least two points"
            )
        sim = tuple(number(value, name) for value in sim)
        policy = tuple(number(value, name) for value in policy)
        if any(right <= left for left, right in zip(sim, sim[1:])):
            raise ValueError(f"{name}.sim_rad must be strictly increasing")
        differences = [right - left for left, right in zip(policy, policy[1:])]
        if not (all(value > 0 for value in differences) or all(value < 0 for value in differences)):
            raise ValueError(f"{name}.policy must be strictly monotonic")
        return _Curve(name, sim, policy)

    def to_policy(self, state: tuple[float, ...]) -> tuple[float, ...]:
        if len(state) != len(self._curves):
            raise ValueError("Joint state dimension differs from calibration")
        return tuple(
            _interpolate(value, curve.sim, curve.policy, curve.name, self._use)
            for value, curve in zip(state, self._curves)
        )

    def to_sim(self, action: tuple[float, ...]) -> tuple[float, ...]:
        if len(action) != len(self._curves):
            raise ValueError("Action dimension differs from calibration")
        return tuple(
            _interpolate(value, curve.policy, curve.sim, curve.name, self._use)
            for value, curve in zip(action, self._curves)
        )

    def action_outside(self, action: tuple[float, ...]) -> tuple[bool, ...]:
        if len(action) != len(self._curves):
            raise ValueError("Action dimension differs from calibration")
        return tuple(
            not min(curve.policy) <= number(value, curve.name) <= max(curve.policy)
            for value, curve in zip(action, self._curves)
        )


class SimulatorJointMap:
    """Radian identity for an explicitly admitted simulator policy; always guarded."""

    def __init__(self, spec):
        from .control_contract import admit

        self.record = admit(spec.control_contract, spec.sim)
        self.digest = spec.control_contract["sha256"]
        self.status = "not_applicable_simulator"
        self.use = CalibrationUse.EXPERIMENTAL
        self.control_metadata = {
            "control_contract_sha256": self.digest,
            "physical_calibration_verified": False,
            "coordinate_mapping": "simulator_native_radians",
        }

    def to_policy(self, values):
        if len(values) != len(self.record["joint_order"]):
            raise ValueError("Simulator vector dimension differs from policy contract")
        return tuple(number(value, name) for value, name in zip(values, self.record["joint_order"]))

    to_sim = to_policy

    def action_outside(self, action):
        self.to_sim(action)
        return (False,) * len(action)


def mapping_for(spec, use=CalibrationUse.VERIFIED):
    if spec.control_contract is not None:
        return SimulatorJointMap(spec)
    return JointMap(spec.calibration, spec.sim.joints, use)
