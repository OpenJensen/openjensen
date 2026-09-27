"""Simulator-native software admission and guarded identity mapping; no Isaac/ML."""

import copy
import hashlib
import json
from dataclasses import replace

import pytest
import yaml
from test_checkpoint_package import export

from sim_worker.rollout.calibration import SimulatorJointMap, mapping_for
from sim_worker.rollout.checkpoint import inspect_checkpoint
from sim_worker.rollout.checkpoint_package import resolve_checkpoint
from sim_worker.rollout.config import load
from sim_worker.rollout.control_contract import admit
from sim_worker.rollout.control_schema import FILE, SCOPE, canonical, read
from sim_worker.rollout.experimental import MotionGuard
from sim_worker.rollout.service import Rollout


def contract(scene):
    return {
        "schema_version": 1,
        "kind": "simulator_joint_position",
        "controller": "joint_position_targets",
        "state_key": "observation.state",
        "action_key": "action",
        "state_units": "radians",
        "action_units": "radians",
        "timebase": "simulation_seconds",
        "joint_order": [f"joint_{i}" for i in range(6)],
        "camera": {
            "key": "observation.images.front",
            "width": 32,
            "height": 32,
            "prim": "/World/Camera",
        },
        "action_fps": 20,
        "source": {
            "dataset_snapshot_id": "sha256:" + "a" * 64,
            "dataset_manifest_sha256": "a" * 64,
            "demonstrations_sha256": "b" * 64,
            "scene_sha256": [hashlib.sha256(scene.read_bytes()).hexdigest()],
            "scene_hash_scope": SCOPE,
            "origins": ["synthetic"],
        },
        "physical_calibration_verified": False,
        "task_success_verified": False,
    }


def setup(tmp_path):
    scene = tmp_path / "scene.usda"
    scene.write_text("#usda 1.0\n")
    root = export(tmp_path / "policy")
    record = contract(scene)
    (root / FILE).write_bytes(canonical(record))
    info = inspect_checkpoint(root)
    data = {
        "api_version": "firebird.rollout.v1",
        "scene": {
            "uri": str(scene),
            "camera": record["camera"]["prim"],
            "articulation": "/World/Robot",
            "joints": list(record["joint_order"]),
        },
        "capture": {"width": 32, "height": 32},
        "control": {"fps": 20, "physics_hz": 120, "steps": 2, "execute_steps": 1},
        "policy": {
            "endpoint": "http://127.0.0.1:12345",
            "model_id": info.model_id,
            "task": "generated software check",
            "timeout_seconds": 1,
        },
        "calibration": None,
        "control_contract": {
            "record": record,
            "sha256": hashlib.sha256(canonical(record)).hexdigest(),
        },
    }
    from sim_worker.rollout.contracts import API_VERSION

    data["api_version"] = API_VERSION
    path = tmp_path / "rollout.yaml"
    path.write_text(yaml.safe_dump(data))
    return root, record, data, path


def test_import_and_model_identity_include_contract(tmp_path):
    root, record, _, path = setup(tmp_path)
    first = inspect_checkpoint(root)
    with resolve_checkpoint(root) as result:
        assert result.checkpoint == first
        assert result.metadata()["checkpoint"]["control_contract"] == record
    record["action_fps"] = 30
    (root / FILE).write_bytes(canonical(record))
    assert inspect_checkpoint(root).model_id != first.model_id
    (root / FILE).unlink()
    legacy = inspect_checkpoint(root)
    assert legacy.control_contract is None and legacy.model_id != first.model_id
    # A caller still using the former model_id is rejected by the existing HTTP identity protocol.


@pytest.mark.parametrize(
    "case", ["joints", "fps", "camera", "width", "scene", "hash", "units", "null", "calibration"]
)
def test_preflight_rejects_mismatches_before_runtime(tmp_path, case):
    root, record, data, path = setup(tmp_path)
    if case == "joints":
        data["scene"]["joints"].reverse()
    elif case == "fps":
        data["control"]["fps"] = 30
    elif case == "camera":
        data["scene"]["camera"] = "/OtherCamera"
    elif case == "width":
        data["capture"]["width"] = 64
    elif case == "scene":
        (tmp_path / "scene.usda").write_text("#usda 1.0\n# changed\n")
    elif case == "hash":
        data["control_contract"]["sha256"] = "0" * 64
    elif case == "null":
        data["control_contract"] = None
    elif case == "calibration":
        data["calibration"] = "legacy.yaml"
    else:
        data["control_contract"]["record"]["action_units"] = "degrees"
    path.write_text(yaml.safe_dump(data))
    messages = {
        "joints": "joint order",
        "fps": "joint order",
        "camera": "joint order",
        "width": "joint order",
        "scene": "root scene",
        "hash": "SHA256 mismatch",
        "units": "Unsupported simulator control contract semantics",
        "null": "cannot be null",
        "calibration": "calibration: null",
    }
    with pytest.raises(ValueError, match=messages[case]):
        load(path)


def test_identity_mapping_is_finite_and_requires_live_motion_guard(tmp_path):
    _, record, _, path = setup(tmp_path)
    spec = load(path)
    mapping = mapping_for(spec)
    assert isinstance(mapping, SimulatorJointMap)
    values = (0.1, -0.2, 0.3, -0.4, 0.5, -0.6)
    assert mapping.to_sim(values) == mapping.to_policy(values) == values
    assert mapping.status == "not_applicable_simulator"
    assert mapping.control_metadata["physical_calibration_verified"] is False
    for bad in (values[:-1], (float("nan"),) + values[1:], (True,) + values[1:]):
        with pytest.raises(ValueError):
            mapping.to_sim(bad)
    with pytest.raises(ValueError, match="motion guard"):
        Rollout(spec, None, None, mapping)
    guard = MotionGuard(lambda: ((-1.0, 1.0),) * 6, spec.sim.fps)
    constrained = guard.constrain((2.0,) * 6, (0.0,) * 6)
    assert constrained.target == (0.1,) * 6
    assert all(constrained.limit_clipped) and all(constrained.speed_clipped)
    # Physics rate is not claimed by the policy contract, but control/action cadence is.
    admit(spec.control_contract, replace(spec.sim, physics_hz=240))


def test_contract_file_must_be_bounded_regular_canonical_and_typed(tmp_path):
    root, record, _, _ = setup(tmp_path)
    path = root / FILE
    for key, value in (
        ("schema_version", True),
        ("action_fps", 20.0),
        ("physical_calibration_verified", 0),
    ):
        bad = copy.deepcopy(record)
        bad[key] = value
        path.write_bytes(canonical(bad))
        with pytest.raises(ValueError):
            read(path)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="canonical"):
        read(path)
    path.write_bytes(b" " * (64 * 1024 + 1))
    with pytest.raises(ValueError, match="bounded"):
        read(path)
    path.unlink()
    path.symlink_to(tmp_path / "scene.usda")
    with pytest.raises(OSError):
        read(path)
