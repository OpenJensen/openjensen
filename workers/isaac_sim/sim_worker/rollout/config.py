import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from sim_worker.rollout.contracts import API_VERSION, SimSpec

_MAX_CONFIG_BYTES = 64 * 1024
_MAX_CAPTURE_DIM = 1920
_MAX_CONTROL_HZ = 120
_MAX_PHYSICS_HZ = 1000
_MAX_STEPS = 72000
_MAX_EXECUTE_STEPS = 100
_MAX_TIMEOUT_SECONDS = 300
_PRIM_PATH = re.compile(r"(/[A-Za-z_][A-Za-z_0-9]*)+")


@dataclass(frozen=True)
class RolloutSpec:
    sim: SimSpec
    steps: int
    execute_steps: int
    endpoint: str
    model_id: str
    task: str
    timeout_seconds: float
    calibration: Path


class _Loader(yaml.SafeLoader):
    def compose_node(self, parent, index):
        if self.check_event(yaml.AliasEvent):
            raise ValueError("YAML aliases are unsupported")
        return super().compose_node(parent, index)

    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result:
                raise ValueError("Configuration keys must be unique strings")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def read_document(path: Path) -> dict:
    with path.open("rb") as stream:
        contents = stream.read(_MAX_CONFIG_BYTES + 1)
    if len(contents) > _MAX_CONFIG_BYTES:
        raise ValueError("Configuration exceeds 64 KiB")
    data = yaml.load(contents, Loader=_Loader)
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a mapping")
    return data


def fields(value: object, expected: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} requires exactly: {', '.join(sorted(expected))}")
    return value


def number(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _integer(value, maximum, label):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{label} must be an integer in [1, {maximum}]")
    return value


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _path(value, parent, label):
    value = _text(value, label)
    if urlsplit(value).scheme:
        raise ValueError(f"{label} must be a local file")
    path = Path(value)
    path = path if path.is_absolute() else parent / path
    if not path.is_file():
        raise ValueError(f"{label} file not found: {path}")
    return path.resolve()


def check_endpoint(endpoint: str) -> str:
    target = urlsplit(endpoint)
    if (
        target.scheme not in {"http", "https"}
        or not target.hostname
        or target.username
        or target.password
        or target.query
        or target.fragment
        or target.path not in {"", "/"}
    ):
        raise ValueError("Policy endpoint must be an HTTP(S) origin without credentials")
    # Accessing port also rejects malformed/non-numeric ports.
    target.port
    return endpoint.rstrip("/")


def load(path: Path) -> RolloutSpec:
    data = fields(
        read_document(path),
        {"api_version", "scene", "capture", "control", "policy", "calibration"},
        "rollout",
    )
    if data["api_version"] != API_VERSION:
        raise ValueError(f"api_version must be {API_VERSION}")
    scene = fields(data["scene"], {"uri", "camera", "articulation", "joints"}, "scene")
    capture = fields(data["capture"], {"width", "height"}, "capture")
    control = fields(data["control"], {"fps", "physics_hz", "steps", "execute_steps"}, "control")
    policy = fields(data["policy"], {"endpoint", "model_id", "task", "timeout_seconds"}, "policy")
    scene_path = _path(scene["uri"], path.parent, "scene.uri")
    if scene_path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ValueError("scene.uri must be a USD file")
    for key in ("camera", "articulation"):
        if not _PRIM_PATH.fullmatch(_text(scene[key], key)):
            raise ValueError(f"{key} must be an absolute USD prim path")
    joints = scene["joints"]
    if (
        not isinstance(joints, list)
        or not joints
        or any(not isinstance(name, str) or not name.isidentifier() for name in joints)
        or len(set(joints)) != len(joints)
    ):
        raise ValueError("scene.joints must be unique joint names in policy order")
    width = _integer(capture["width"], _MAX_CAPTURE_DIM, "width")
    height = _integer(capture["height"], _MAX_CAPTURE_DIM, "height")
    if width % 2 or height % 2:
        raise ValueError("Video dimensions must be even")
    fps = _integer(control["fps"], _MAX_CONTROL_HZ, "fps")
    physics_hz = _integer(control["physics_hz"], _MAX_PHYSICS_HZ, "physics_hz")
    if physics_hz % fps:
        raise ValueError("physics_hz must be an integer multiple of fps")
    timeout = number(policy["timeout_seconds"], "timeout_seconds")
    if not 0 < timeout <= _MAX_TIMEOUT_SECONDS:
        raise ValueError(f"timeout_seconds must be in (0, {_MAX_TIMEOUT_SECONDS}]")
    return RolloutSpec(
        SimSpec(
            str(scene_path),
            scene["camera"],
            scene["articulation"],
            tuple(joints),
            width,
            height,
            fps,
            physics_hz,
        ),
        _integer(control["steps"], _MAX_STEPS, "steps"),
        _integer(control["execute_steps"], _MAX_EXECUTE_STEPS, "execute_steps"),
        check_endpoint(os.environ.get("POLICY_ENDPOINT", policy["endpoint"])),
        _text(policy["model_id"], "model_id"),
        _text(policy["task"], "task"),
        timeout,
        _path(data["calibration"], path.parent, "calibration"),
    )
