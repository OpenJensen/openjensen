"""CPU observation replay admission never implies simulated policy performance."""

import sys

import pytest
from pydantic import ValidationError
from vla_platform.lifecycle.contracts import PolicyRequest
from vla_platform.lifecycle.runtime import Runtime, RuntimeCatalog, command


def recipe():
    return {
        "adapter": "act-packed-observation-v1",
        "selection": [{"episode_index": 0, "frame_index": 3}],
        "coordinate_attestation": "policy_recorded_coordinates",
        "units": ["degrees"] * 5 + ["recorded_gripper"],
    }


def request(**updates):
    return PolicyRequest.model_validate(
        {
            "operation": "policy.run",
            "runtime_id": "replay-cpu",
            "artifact_id": "packed:policy",
            "dataset_job_id": "dataset",
            "native_replay": recipe(),
            **updates,
        }
    )


def configured(root, **updates):
    return Runtime.model_validate(
        {
            "id": "replay-cpu",
            "label": "CPU observation replay",
            "native_replay_only": True,
            "native_replay_python": sys.executable,
            "native_replay_dataset_python": sys.executable,
            "native_replay_root": str(root),
            **updates,
        }
    )


def test_explicit_recipe_roundtrips_without_scoring_or_cloud_target():
    value = request()
    assert value.timeout_seconds == 600
    assert PolicyRequest.model_validate(value.model_dump()) == value
    assert value.simulation is None and value.limits is None


@pytest.mark.parametrize(
    "changes",
    [
        {"operation": "policy.evaluate"},
        {"operation": "policy.distill"},
        {"source_id": "implicit-source"},
        {"artifact_id": None},
        {"dataset_job_id": None},
        {"resume_job_id": "resume"},
        {"training": {}},
        {"training_method": "full"},
        {"precision": {"language": "Q4_0"}},
        {"candidates": [{"language": "Q4_0"}]},
        {"evaluation": {"steps": 10}},
        {"limits": {}},
        {"simulation": {"profile_id": "cup", "experimental": True}},
        {"native_quantization": {"format": "firebird_quant", "bits": 8}},
        {"timeout_seconds": 601},
        {"timeout_seconds": 29},
        {"timeout_seconds": 600.0},
        {"timeout_seconds": True},
    ],
)
def test_replay_cannot_select_other_execution_or_claim_evaluation(changes):
    with pytest.raises(ValidationError):
        request(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"selection": []},
        {"selection": [{"episode_index": 0, "frame_index": 3}] * 2},
        {"selection": [{"episode_index": 0, "frame_index": x} for x in range(33)]},
        {"selection": [{"episode_index": False, "frame_index": 3}]},
        {"selection": [{"episode_index": 0, "frame_index": "3"}]},
        {"selection": [{"episode_index": 0, "frame_index": -1}]},
        {"selection": [{"episode_index": 0, "frame_index": 10000001}]},
        {"units": ["degrees"] * 5},
        {"units": ["x\ny"] * 6},
        {"coordinate_attestation": "guessed"},
        {"observations_path": "/tmp/pixels"},
    ],
)
def test_selections_are_bounded_explicit_unique_and_typed(changes):
    with pytest.raises(ValidationError):
        request(native_replay={**recipe(), **changes})


def test_replay_only_runtime_has_no_implicit_engine_training_or_simulator(tmp_path):
    root = tmp_path / "workers/isaac_sim"
    runtime = configured(root)
    catalog = RuntimeCatalog(runtimes=[runtime])
    assert not catalog.public()["runtimes"][0]["native_replay"]
    for path in [
        root / "sim_worker/rollout/native_replay.py",
        root / "sim_worker/rollout/native_replay_prepare.py",
        root / "sim_worker/rollout/native_replay_contracts.py",
        root / "sim_worker/rollout/server.py",
        root.parent / "firebird_quant/src/firebird_quant/native_consumer.py",
        root.parent / "act_optimizer/src/firebird_act/application.py",
        root.parent / "smolvla_qlora/src/firebird_vla/local_dataset.py",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# installed worker marker")
    public = catalog.public()["runtimes"][0]
    assert public["native_replay"] and public["native_replay_only"]
    assert not any(public[key] for key in ["training", "run", "simulation", "engine_evaluation"])
    assert "native_replay_python" not in public
    with pytest.raises(ValueError, match="observation replay only"):
        command(runtime, tmp_path / "request", tmp_path / "result", tmp_path, "owned")


@pytest.mark.parametrize(
    "updates",
    [
        {"provider": "gcp"},
        {"execution": "skypilot"},
        {"device": "cuda"},
        {"native_replay_python": None},
        {"native_replay_root": None},
        {"native_replay_dataset_python": None},
        {"native_quantization_only": True},
        {"export_only": True},
        {"native_distillation_only": True},
        {"training_python": "python"},
        {"simulator_lane": "isaac"},
    ],
)
def test_dedicated_runtime_rejects_conflicting_backends(tmp_path, updates):
    with pytest.raises(ValidationError):
        configured(tmp_path, **updates)
