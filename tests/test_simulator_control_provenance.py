"""Coordinate provenance through application boundaries; no native ML or simulator."""

import hashlib
import json
from copy import deepcopy

import pytest
from vla_platform.lifecycle import control_provenance as provenance
from vla_platform.lifecycle import control_schema as schema
from vla_platform.lifecycle import isaac_runner as runner


def control_record():
    return {
        "schema_version": 1,
        "kind": "simulator_joint_position",
        "controller": "joint_position_targets",
        "state_key": "observation.state",
        "action_key": "action",
        "state_units": "radians",
        "action_units": "radians",
        "timebase": "simulation_seconds",
        "joint_order": [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ],
        "camera": {
            "key": "observation.images.front",
            "width": 640,
            "height": 360,
            "prim": "/World/Camera",
        },
        "action_fps": 30,
        "source": {
            "dataset_snapshot_id": "sha256:" + "a" * 64,
            "dataset_manifest_sha256": "a" * 64,
            "demonstrations_sha256": "b" * 64,
            "scene_sha256": ["c" * 64],
            "scene_hash_scope": "root USD bytes; referenced assets not inventoried",
            "origins": ["recorded"],
        },
        "physical_calibration_verified": False,
        "task_success_verified": False,
    }


def write_policy(root):
    root.mkdir(parents=True, exist_ok=True)
    record = control_record()
    raw = schema.canonical(record)
    (root / schema.FILE).write_bytes(raw)
    (root / "config.json").write_text(
        json.dumps(
            {
                "type": "act",
                "input_features": {
                    "observation.state": {"type": "STATE", "shape": [6]},
                    "observation.images.front": {"type": "VISUAL", "shape": [3, 360, 640]},
                },
                "output_features": {"action": {"type": "ACTION", "shape": [6]}},
            }
        )
    )
    return {"control_contract": record, "control_contract_sha256": hashlib.sha256(raw).hexdigest()}


def write_training(root):
    checkpoint = root / "checkpoint"
    metadata = write_policy(checkpoint / "pretrained_model")
    (checkpoint / schema.FILE).write_bytes(schema.canonical(metadata["control_contract"]))
    recipe = {
        "dataset_source": "local",
        "dataset_revision": "sha256:" + "a" * 64,
        "dataset_manifest_sha256": "a" * 64,
        "control_contract": metadata["control_contract"],
    }
    (checkpoint / "recipe.json").write_text(json.dumps(recipe))
    (checkpoint / "manifest.json").write_text(
        json.dumps(
            {
                "files": {
                    schema.FILE: metadata["control_contract_sha256"],
                    "pretrained_model/" + schema.FILE: metadata["control_contract_sha256"],
                }
            }
        )
    )
    return metadata


def test_recorded_coordinates_survive_training_and_nested_import_metadata(tmp_path):
    metadata = write_training(tmp_path)
    assert provenance.training_claims(tmp_path, metadata) == metadata
    assert provenance.training_claims(tmp_path, {"checkpoint": metadata}) == metadata
    assert metadata["control_contract"]["physical_calibration_verified"] is False
    assert metadata["control_contract"]["task_success_verified"] is False


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"control_contract": None, "control_contract_sha256": None},
        {"checkpoint": {"control_contract": None, "control_contract_sha256": None}},
    ],
)
def test_existing_legacy_checkpoint_with_null_optional_fields_still_works(tmp_path, metadata):
    (tmp_path / "checkpoint/pretrained_model").mkdir(parents=True)
    (tmp_path / "checkpoint/manifest.json").write_text('{"step": 1}')
    assert provenance.training_claims(tmp_path, metadata) == {}
    provenance.require_transform_support(metadata, ["config.json", "model.safetensors"])


@pytest.mark.parametrize(
    "fault",
    [
        "claim-digest",
        "claim-record",
        "nested-claim",
        "missing-file",
        "missing-claim",
        "camera",
        "coerced-dimension",
    ],
)
def test_import_cannot_change_or_drop_saved_coordinate_semantics(tmp_path, fault):
    metadata = write_policy(tmp_path)
    if fault == "claim-digest":
        metadata["control_contract_sha256"] = "e" * 64
    elif fault == "claim-record":
        metadata["control_contract"]["joint_order"].reverse()
    elif fault == "nested-claim":
        metadata["checkpoint"] = {"control_contract_sha256": "e" * 64}
    elif fault == "missing-file":
        (tmp_path / schema.FILE).unlink()
    elif fault == "missing-claim":
        metadata.clear()
    else:
        path = tmp_path / "config.json"
        config = json.loads(path.read_text())
        if fault == "camera":
            config["input_features"]["observation.images.front"]["shape"] = [3, 640, 360]
        else:
            config["output_features"]["action"]["shape"] = [6.0]
        path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        provenance.policy_claims(tmp_path, metadata)


@pytest.mark.parametrize("evidence", ["recipe", "manifest"])
def test_training_cannot_downgrade_to_legacy_after_both_sidecars_are_lost(tmp_path, evidence):
    write_training(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / schema.FILE).unlink()
    (checkpoint / "pretrained_model" / schema.FILE).unlink()
    if evidence == "recipe":
        (checkpoint / "manifest.json").write_text('{"files": {}}')
    else:
        (checkpoint / "recipe.json").write_text("{}")
    with pytest.raises(ValueError, match="lost its simulator control"):
        provenance.training_claims(tmp_path, {})


@pytest.mark.parametrize("fault", ["checkpoint-copy", "manifest", "recipe", "dataset"])
def test_export_refuses_changed_training_provenance(tmp_path, fault):
    metadata = write_training(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    if fault == "checkpoint-copy":
        (checkpoint / schema.FILE).unlink()
    elif fault == "manifest":
        (checkpoint / "manifest.json").write_text('{"files": {}}')
    else:
        path = checkpoint / "recipe.json"
        recipe = json.loads(path.read_text())
        if fault == "recipe":
            recipe["control_contract"]["action_fps"] = 15
        else:
            recipe["dataset_revision"] = "sha256:" + "e" * 64
        path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError):
        provenance.training_claims(tmp_path, metadata)


@pytest.mark.parametrize("location", ["metadata", "nested", "file"])
def test_unsupported_transform_cannot_silently_drop_coordinates(tmp_path, location):
    claims = write_policy(tmp_path)
    metadata, files = {}, []
    if location == "metadata":
        metadata = claims
    elif location == "nested":
        metadata = {"checkpoint": claims}
    else:
        files = ["policy/" + schema.FILE]
    with pytest.raises(ValueError, match="preserving its control contract"):
        provenance.require_transform_support(metadata, files)


def evidence():
    return {
        "control_contract_sha256": "a" * 64,
        "coordinate_mapping": "simulator_native_radians",
        "calibration_status": "not_applicable_simulator",
        "physical_calibration_verified": False,
    }


def test_matching_rollout_retains_simulator_identity_without_physical_claim():
    accepted = evidence()
    rollout = {**accepted, "calibration_sha256": "a" * 64}
    runner.verify_coordinate_evidence(accepted, rollout)
    runner.verify_coordinate_evidence({}, {"calibration_status": "experimental"})


@pytest.mark.parametrize(
    "fault", ["missing", "digest", "mapping", "physical", "coerced-bool", "mapping-digest"]
)
def test_rollout_rejects_changed_or_missing_coordinate_evidence(fault):
    accepted = evidence()
    rollout = deepcopy(accepted) | {"calibration_sha256": "a" * 64}
    if fault == "missing":
        rollout.clear()
    elif fault == "digest":
        rollout["control_contract_sha256"] = "b" * 64
    elif fault == "mapping":
        rollout["coordinate_mapping"] = "legacy_affine"
    elif fault == "physical":
        rollout["physical_calibration_verified"] = True
    elif fault == "coerced-bool":
        rollout["physical_calibration_verified"] = 0
    else:
        rollout["calibration_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        runner.verify_coordinate_evidence(accepted, rollout)


@pytest.mark.parametrize(
    "rollout",
    [
        evidence(),
        {"coordinate_mapping": "simulator_native_radians"},
        {"calibration_status": "not_applicable_simulator"},
        {"physical_calibration_verified": True},
    ],
)
def test_legacy_run_cannot_silently_switch_coordinate_system(rollout):
    with pytest.raises(ValueError):
        runner.verify_coordinate_evidence({}, rollout)
