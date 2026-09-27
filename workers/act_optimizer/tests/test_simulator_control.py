"""ACT export preserves portable simulator lineage; these fixtures do not run a model."""

import hashlib
from pathlib import Path

import pytest
from test_application import job
from test_application import stub_probes as stub_probes
from test_training_source import digest, edit, resign, write
from test_training_source import training_bundle as training_bundle

from firebird_act import application, export
from firebird_act.bundle import read_json
from firebird_act.control_schema import FILE, SCOPE, canonical, metadata
from firebird_act.training_source import admit_training_source


def local_control(root):
    checkpoint = root / "checkpoint"
    record = {
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
            "scene_sha256": ["c" * 64],
            "scene_hash_scope": SCOPE,
            "origins": ["synthetic"],
        },
        "physical_calibration_verified": False,
        "task_success_verified": False,
    }
    for directory in (checkpoint, checkpoint / "pretrained_model"):
        (directory / FILE).write_bytes(canonical(record))
    recipe = read_json(checkpoint / "recipe.json") | {
        "dataset_source": "local",
        "dataset_id": "firebird/local-" + "a" * 16,
        "dataset_revision": "sha256:" + "a" * 64,
        "dataset_manifest_sha256": "a" * 64,
        "control_contract": record,
    }
    write(checkpoint / "recipe.json", recipe)
    manifest = read_json(root / "manifest.json")
    manifest["metadata"].update(
        dataset={
            "source": "local",
            "format": "lerobot_v3",
            "repo_id": None,
            "revision": "metadata-sha256:" + "e" * 64,
            "metadata_sha256": "e" * 64,
            "inspection_scope": "complete_snapshot",
            "total_frames": 4,
            "total_episodes": 2,
            "snapshot": {
                "schema_version": 1,
                "format": "lerobot_v3",
                "id": "sha256:" + "a" * 64,
                "manifest_sha256": "a" * 64,
                "total_frames": 4,
                "total_episodes": 2,
            },
            "local_path": "/not/exported/private",
        },
        dataset_snapshot_id=recipe["dataset_revision"],
        dataset_manifest_sha256="a" * 64,
        **metadata(checkpoint),
    )
    write(root / "manifest.json", manifest)
    write(root / "verification.json", read_json(root / "verification.json") | metadata(checkpoint))
    resign(root)
    return record


def test_local_snapshot_export_preserves_control_and_portable_source(
    training_bundle, tmp_path, stub_probes, monkeypatch
):
    record = local_control(training_bundle)
    original = export.run_probe

    def probe(checkpoint, *args, **kwargs):
        return original(checkpoint, *args, **kwargs) | metadata(checkpoint)

    monkeypatch.setattr(export, "run_probe", probe)
    before = {
        str(p.relative_to(training_bundle)): digest(p)
        for p in training_bundle.rglob("*")
        if p.is_file()
    }
    result = application.run_job(job(training_bundle, tmp_path / "operation"))
    output = Path(result["artifact"]["path"])
    saved = read_json(output / "manifest.json")["metadata"]
    assert saved["control_contract"] == record
    assert saved["control_contract_sha256"] == hashlib.sha256(canonical(record)).hexdigest()
    assert saved["dataset"] == {
        "source": "local",
        "snapshot_id": "sha256:" + "a" * 64,
        "manifest_sha256": "a" * 64,
        "revision": "metadata-sha256:" + "e" * 64,
    }
    assert "/not/exported/private" not in (output / "manifest.json").read_text()
    assert (output / "policy" / FILE).read_bytes() == canonical(record)
    assert len(stub_probes) == 3
    assert before == {
        str(p.relative_to(training_bundle)): digest(p)
        for p in training_bundle.rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize(
    "case",
    [
        "root_missing",
        "policy_missing",
        "recipe_missing",
        "metadata_missing",
        "verify_missing",
        "snapshot",
        "joint_shape",
        "clock",
    ],
)
def test_rehashed_training_control_downgrade_or_mismatch_refused(training_bundle, case):
    local_control(training_bundle)
    checkpoint = training_bundle / "checkpoint"
    if case == "root_missing":
        (checkpoint / FILE).unlink()
    elif case == "policy_missing":
        (checkpoint / "pretrained_model" / FILE).unlink()
    elif case == "recipe_missing":
        edit(checkpoint / "recipe.json", "control_contract", None)
    elif case == "metadata_missing":
        manifest = read_json(training_bundle / "manifest.json")
        manifest["metadata"].pop("control_contract")
        write(training_bundle / "manifest.json", manifest)
    elif case == "verify_missing":
        edit(training_bundle / "verification.json", "control_contract", None)
    elif case == "snapshot":
        edit(checkpoint / "recipe.json", "dataset_manifest_sha256", "d" * 64)
    elif case == "joint_shape":
        record = read_json(checkpoint / "pretrained_model" / FILE)
        record["joint_order"].pop()
        write(checkpoint / "pretrained_model" / FILE, record)
    else:
        temporal = {"invalid": "a known temporal file cannot disagree with control FPS"}
        write(checkpoint / "pretrained_model/temporal-contract.json", temporal)
    sha = resign(training_bundle)
    with pytest.raises(ValueError):
        admit_training_source(training_bundle, artifact_id="fixture:operation", manifest_sha256=sha)


def test_export_reload_must_explicitly_preserve_control_proof(
    training_bundle, tmp_path, stub_probes
):
    local_control(training_bundle)
    with pytest.raises(ValueError, match="control provenance"):
        application.run_job(job(training_bundle, tmp_path / "operation"))
    assert not (tmp_path / "operation/inference-export").exists()


def test_old_local_checkpoint_without_new_top_level_claims_remains_admissible(training_bundle):
    local_control(training_bundle)
    checkpoint = training_bundle / "checkpoint"
    for p in (checkpoint / FILE, checkpoint / "pretrained_model" / FILE):
        p.unlink()
    for path in (checkpoint / "recipe.json", training_bundle / "verification.json"):
        value = read_json(path)
        value.pop("control_contract", None)
        value.pop("control_contract_sha256", None)
        write(path, value)
    value = read_json(training_bundle / "manifest.json")
    for key in (
        "control_contract",
        "control_contract_sha256",
        "dataset_snapshot_id",
        "dataset_manifest_sha256",
    ):
        value["metadata"].pop(key, None)
    write(training_bundle / "manifest.json", value)
    admission = admit_training_source(
        training_bundle, artifact_id="fixture:operation", manifest_sha256=resign(training_bundle)
    )
    assert admission.dataset_manifest_sha256 == "a" * 64
    assert admission.control_contract is None
