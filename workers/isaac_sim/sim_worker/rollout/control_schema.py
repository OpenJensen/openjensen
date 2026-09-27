"""Versioned simulator coordinates; identical stdlib schema shipped with isolated workers.

This record binds declared dataset semantics, not task quality, scene asset closure,
physical calibration, or the truth of an operator-supplied recording.
"""

import hashlib
import json
import os
import re
import stat
from pathlib import Path

FILE = "control-contract.json"
LIMIT = 64 * 1024
SCOPE = "root USD bytes; referenced assets not inventoried"
_SHA = re.compile(r"[a-f0-9]{64}")
_PRIM = re.compile(r"(/[A-Za-z_][A-Za-z_0-9]*)+")


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _fields(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys.split()):
        raise ValueError("Invalid simulator control contract fields")


def _sha(value):
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def validate(record, config=None):
    _fields(
        record,
        "schema_version kind controller state_key action_key state_units action_units "
        "timebase joint_order camera action_fps source physical_calibration_verified "
        "task_success_verified",
    )
    constants = {
        "schema_version": 1,
        "kind": "simulator_joint_position",
        "controller": "joint_position_targets",
        "state_key": "observation.state",
        "action_key": "action",
        "state_units": "radians",
        "action_units": "radians",
        "timebase": "simulation_seconds",
        "physical_calibration_verified": False,
        "task_success_verified": False,
    }
    if any(type(record[k]) is not type(v) or record[k] != v for k, v in constants.items()):
        raise ValueError("Unsupported simulator control contract semantics")
    joints = record["joint_order"]
    if (
        not isinstance(joints, list)
        or not 1 <= len(joints) <= 32
        or any(not isinstance(n, str) or not n.isidentifier() for n in joints)
        or len(set(joints)) != len(joints)
    ):
        raise ValueError("Simulator contract requires unique ordered joint names")
    if type(record["action_fps"]) is not int or not 1 <= record["action_fps"] <= 60:
        raise ValueError("Simulator contract action FPS must be an integer in 1..60")
    camera = record["camera"]
    _fields(camera, "key width height prim")
    if (
        not isinstance(camera["key"], str)
        or not re.fullmatch(r"observation\.images\.[A-Za-z_][A-Za-z_0-9]*", camera["key"])
        or not isinstance(camera["prim"], str)
        or not _PRIM.fullmatch(camera["prim"])
        or any(
            type(camera[k]) is not int or not 2 <= camera[k] <= 1920 or camera[k] % 2
            for k in ("width", "height")
        )
    ):
        raise ValueError("Invalid simulator contract camera")
    source = record["source"]
    _fields(
        source,
        "dataset_snapshot_id dataset_manifest_sha256 demonstrations_sha256 "
        "scene_sha256 scene_hash_scope origins",
    )
    if (
        not _sha(source["dataset_manifest_sha256"])
        or source["dataset_snapshot_id"] != "sha256:" + source["dataset_manifest_sha256"]
        or not _sha(source["demonstrations_sha256"])
        or source["scene_hash_scope"] != SCOPE
    ):
        raise ValueError("Invalid simulator contract dataset provenance")
    scenes, origins = source["scene_sha256"], source["origins"]
    if (
        not isinstance(scenes, list)
        or not 1 <= len(scenes) <= 128
        or any(not _sha(s) for s in scenes)
        or scenes != sorted(set(scenes))
        or not isinstance(origins, list)
        or not origins
        or any(not isinstance(o, str) or o not in {"recorded", "synthetic"} for o in origins)
        or origins != sorted(set(origins))
    ):
        raise ValueError("Invalid simulator scene or source-origin provenance")
    if config is not None:
        if not isinstance(config, dict) or config.get("type") not in {"act", "smolvla"}:
            raise ValueError("Simulator contract supports ACT and SmolVLA policies only")
        expected_inputs = {
            "observation.state": {"type": "STATE", "shape": [len(joints)]},
            camera["key"]: {"type": "VISUAL", "shape": [3, camera["height"], camera["width"]]},
        }
        # Canonical comparison also rejects True==1 and numeric coercion.
        if canonical(config.get("input_features")) != canonical(expected_inputs) or canonical(
            config.get("output_features")
        ) != canonical({"action": {"type": "ACTION", "shape": [len(joints)]}}):
            raise ValueError("Policy features differ from simulator control contract")
    return record


def read(path):
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= LIMIT:
            raise ValueError("Control contract must be a bounded regular file")
        raw = stream.read(LIMIT + 1)
    if len(raw) > LIMIT:
        raise ValueError("Control contract exceeds 64 KiB")
    try:
        value = json.loads(raw)
        validate(value)
        if canonical(value) != raw:
            raise ValueError("Control contract must use canonical JSON")
    except (UnicodeError, TypeError, KeyError, RecursionError) as error:
        raise ValueError("Invalid simulator control contract") from error
    return value, hashlib.sha256(raw).hexdigest()


def optional(root, config=None):
    path = Path(root) / FILE
    if not os.path.lexists(path):
        return None, None
    value, digest = read(path)
    validate(value, config)
    return value, digest


def metadata(root, config=None):
    value, digest = optional(root, config)
    return {} if value is None else {"control_contract": value, "control_contract_sha256": digest}
