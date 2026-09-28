"""Contract admission happens before any SkyPilot submission; no SDK/provider calls."""

import copy
import hashlib
from types import SimpleNamespace

import pytest
from sim_worker.rollout.control_schema import SCOPE, canonical
from test_rollout import launcher


def fixture(tmp_path, monkeypatch):
    scene = tmp_path / "scene.usda"
    scene.write_text("#usda 1.0\n")
    record = {
        "schema_version": 1,
        "kind": "simulator_joint_position",
        "controller": "joint_position_targets",
        "state_key": "observation.state",
        "action_key": "action",
        "state_units": "radians",
        "action_units": "radians",
        "timebase": "simulation_seconds",
        "joint_order": ["joint"],
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
    info = SimpleNamespace(
        state_dim=1,
        action_dim=1,
        action_steps=3,
        chunk_size=8,
        width=32,
        height=32,
        camera_key="observation.images.front",
        model_id="sha256:" + "d" * 64,
        policy_type="act",
        control_contract=record,
        control_contract_sha256=hashlib.sha256(canonical(record)).hexdigest(),
    )
    monkeypatch.setattr(launcher, "_inspect", lambda _: info)
    monkeypatch.setattr(launcher, "_WORKER", tmp_path)
    data = {
        "scene": {
            "uri": "scene.usda",
            "camera": "/World/Camera",
            "articulation": "/World/Robot",
            "joints": ["joint"],
        },
        "control": {"fps": 20, "physics_hz": 120, "execute_steps": 5},
        "capture": {"width": 32, "height": 32},
        "policy": {"model_id": "old"},
        "calibration": "old-calibration.yaml",
    }
    tasks = {"vla": {"file_mounts": {}, "envs": {}}}
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    return info, data, tasks, checkpoint


def test_bind_embeds_exact_contract_and_keeps_existing_mount(tmp_path, monkeypatch):
    info, data, tasks, checkpoint = fixture(tmp_path, monkeypatch)
    launcher._bind_model(tasks, data, checkpoint, None, manifest_path=tmp_path / "run.yaml")
    assert data["control_contract"] == {
        "record": info.control_contract,
        "sha256": info.control_contract_sha256,
    }
    assert data["calibration"] is None
    assert data["control"]["execute_steps"] == 3
    assert tasks["vla"]["file_mounts"][launcher._CHECKPOINT_MOUNT] == str(checkpoint)
    assert launcher._checkpoint(tasks["vla"], data) is info
    del data["control_contract"]
    with pytest.raises(ValueError, match="missing or differs"):
        launcher._checkpoint(tasks["vla"], data)


@pytest.mark.parametrize("case", ["order", "clock", "camera", "capture", "scene", "typed"])
def test_mismatch_never_publishes_mount_or_changes_model(tmp_path, monkeypatch, case):
    info, data, tasks, checkpoint = fixture(tmp_path, monkeypatch)
    if case == "order":
        data["scene"]["joints"] = ["other"]
    elif case == "clock":
        data["control"]["fps"] = 30
    elif case == "camera":
        data["scene"]["camera"] = "/Other"
    elif case == "capture":
        data["capture"]["width"] = 64
    elif case == "scene":
        (tmp_path / "scene.usda").write_text("different root")
    else:
        record = copy.deepcopy(info.control_contract)
        record["schema_version"] = True
        data["control_contract"] = {"record": record, "sha256": info.control_contract_sha256}
    with pytest.raises(ValueError):
        launcher._bind_model(tasks, data, checkpoint, None, manifest_path=tmp_path / "run.yaml")
    assert tasks["vla"]["file_mounts"] == {}
    assert data["policy"]["model_id"] == "old"
