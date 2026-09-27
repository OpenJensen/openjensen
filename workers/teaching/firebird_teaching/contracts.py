"""Bounded, simulator-native teaching contract. No model-generated code or paths."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from sim_worker.rollout.contracts import SimSpec

OPERATIONS = frozenset({"task", "start", "pause", "reset", "correct", "mark_failure", "finish"})
LINEAGE = re.compile(r"[a-zA-Z0-9_.:-]{1,128}\Z")
IDENTITY = re.compile(r"[a-zA-Z0-9_-]{1,96}\Z")
MAX_JSON = 16 * 1024


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def decode(raw: bytes, *, limit: int = MAX_JSON) -> dict:
    if len(raw) > limit:
        raise ValueError("JSON exceeds size limit")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    value = json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")),
    )
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def text(value: object, label: str, maximum: int = 256) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum
        or any(ord(c) < 32 for c in value)
    ):
        raise ValueError(f"{label} must be nonempty text up to {maximum} characters")
    return value.strip()


def integer(value: object, label: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label} must be an integer in [{low}, {high}]")
    return value


def finite(value: object, label: str) -> float:
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


@dataclass(frozen=True)
class Settings:
    sim: SimSpec
    scene_sha256: str
    lineage_group: str
    max_steps: int = 300
    max_episode_bytes: int = 512 * 1024 * 1024
    origin: str = "recorded"

    @classmethod
    def load(cls, path: Path) -> Settings:
        data = decode(path.read_bytes())
        required = {
            "scene",
            "camera",
            "articulation",
            "joints",
            "width",
            "height",
            "fps",
            "physics_hz",
            "lineage_group",
            "max_steps",
            "max_episode_bytes",
        }
        if set(data) != required:
            raise ValueError("Teaching settings require exactly: " + ", ".join(sorted(required)))
        scene = (path.parent / text(data["scene"], "scene", 2048)).resolve(strict=True)
        if not scene.is_file() or scene.suffix.lower() not in {".usd", ".usda", ".usdc"}:
            raise ValueError("Scene must be an existing USD file")
        joints = data["joints"]
        if (
            not isinstance(joints, list)
            or not 1 <= len(joints) <= 32
            or any(not isinstance(j, str) or not j.isidentifier() for j in joints)
            or len(set(joints)) != len(joints)
        ):
            raise ValueError("Joints must be 1..32 distinct names in native action order")
        for name in ("camera", "articulation"):
            if not re.fullmatch(r"(/[A-Za-z_][A-Za-z_0-9]*)+", text(data[name], name)):
                raise ValueError(f"{name} must be an absolute USD prim path")
        width, height = (integer(data[k], k, 2, 1920) for k in ("width", "height"))
        if width % 2 or height % 2:
            raise ValueError("Video dimensions must be even")
        fps = integer(data["fps"], "fps", 1, 60)
        physics = integer(data["physics_hz"], "physics_hz", fps, 1000)
        if physics % fps:
            raise ValueError("Physics rate must be an integer multiple of capture rate")
        if not isinstance(data["lineage_group"], str) or not LINEAGE.fullmatch(
            data["lineage_group"]
        ):
            raise ValueError("lineage_group must contain only ASCII letters, digits, _, ., :, or -")
        return cls(
            SimSpec(
                str(scene),
                data["camera"],
                data["articulation"],
                tuple(joints),
                width,
                height,
                fps,
                physics,
            ),
            hashlib.sha256(scene.read_bytes()).hexdigest(),
            text(data["lineage_group"], "lineage_group", 128),
            integer(data["max_steps"], "max_steps", 1, 3600),
            integer(data["max_episode_bytes"], "max_episode_bytes", 1024, 2 * 1024**3),
        )

    def metadata(self) -> dict:
        return {
            "scene": self.sim.scene,
            "scene_sha256": self.scene_sha256,
            "lineage_group": self.lineage_group,
            "joint_names": list(self.sim.joints),
            "camera_key": "observation.images.front",
            "camera_prim": self.sim.camera,
            "width": self.sim.width,
            "height": self.sim.height,
            "fps": self.sim.fps,
            "physics_hz": self.sim.physics_hz,
            "controller": "joint_position_targets",
            "state_units": "radians",
            "action_units": "radians",
            "timebase": "simulation_seconds",
            "max_steps": self.max_steps,
            "max_episode_bytes": self.max_episode_bytes,
            "origin": self.origin,
            "scope": "Simulator coordinates; no physical calibration or task-success claim",
        }


@dataclass(frozen=True)
class Command:
    command_id: str
    session_id: str
    episode_id: str | None
    expected_revision: int
    operation: str
    arguments: dict

    @classmethod
    def parse(cls, value: dict) -> Command:
        if set(value) != {
            "command_id",
            "session_id",
            "episode_id",
            "expected_revision",
            "operation",
            "arguments",
        }:
            raise ValueError("Unexpected or missing command fields")
        ident = value["command_id"]
        session = value["session_id"]
        if not isinstance(session, str) or not IDENTITY.fullmatch(session):
            raise ValueError("Invalid executor session ID")
        episode = value["episode_id"]
        if not isinstance(ident, str) or not IDENTITY.fullmatch(ident):
            raise ValueError("Invalid command ID")
        if episode is not None and (
            not isinstance(episode, str) or not IDENTITY.fullmatch(episode)
        ):
            raise ValueError("Invalid episode ID")
        revision = integer(value["expected_revision"], "expected_revision", 0, 2**53 - 1)
        operation = value["operation"]
        if not isinstance(operation, str) or operation not in OPERATIONS:
            raise ValueError("Unknown teaching operation")
        args = value["arguments"]
        required = {
            "task": {"instruction"},
            "correct": {"joint", "delta_rad"},
            "mark_failure": {"reason"},
        }.get(operation, set())
        if not isinstance(args, dict) or set(args) != required:
            raise ValueError("Unexpected or missing operation arguments")
        args = dict(args)
        if operation == "task":
            args["instruction"] = text(args["instruction"], "instruction")
        if operation == "mark_failure":
            args["reason"] = text(args["reason"], "failure reason")
        if operation == "correct":
            args["joint"] = text(args["joint"], "joint", 96)
            delta = finite(args["delta_rad"], "delta_rad")
            if not 0 < abs(delta) <= 0.15:
                raise ValueError("Correction must be nonzero and at most 0.15 radians")
            args["delta_rad"] = delta
        return cls(ident, session, episode, revision, operation, args)
