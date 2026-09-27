"""Boundary tests for local ACT student jobs; no model-quality claim."""

import sys

import pytest
from pydantic import ValidationError
from vla_platform.lifecycle.contracts import PolicyRequest
from vla_platform.lifecycle.runtime import Runtime, RuntimeCatalog, command


def recipe():
    return {
        "adapter": "act-act-v1",
        "student": "act-256",
        "steps": 100,
        "learning_rate": 0.0001,
        "seed": 1729,
        "frame_stride": 30,
        "splits": {"train": [0], "validation": [1], "final": [2]},
        "coordinate_attestation": "teacher_recorded_coordinates",
        "units": ["degrees"] * 5 + ["recorded_gripper"],
    }


def request(**updates):
    return PolicyRequest.model_validate(
        {
            "operation": "policy.distill",
            "runtime_id": "student-cpu",
            "artifact_id": "teacher:policy",
            "dataset_job_id": "dataset",
            "native_distillation": recipe(),
            **updates,
        }
    )


def test_student_request_roundtrips_without_resume_or_implicit_splits():
    value = request()
    assert value.timeout_seconds == 600
    assert PolicyRequest.model_validate(value.model_dump()) == value
    assert value.native_distillation.splits.final == [2]
    assert value.native_quantization is None and value.resume_job_id is None


@pytest.mark.parametrize(
    "changes",
    [
        {"operation": "policy.quantize"},
        {"native_distillation": None},
        {"source_id": "hidden-source"},
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
        {"timeout_seconds": 3601},
        {"timeout_seconds": 29},
        {"timeout_seconds": 600.0},
    ],
)
def test_no_ambiguous_lifecycle_request(changes):
    with pytest.raises(ValidationError):
        request(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"steps": True},
        {"steps": 2.0},
        {"steps": 0},
        {"steps": 10001},
        {"learning_rate": True},
        {"learning_rate": "0.0001"},
        {"learning_rate": float("nan")},
        {"seed": -1},
        {"frame_stride": 0},
        {"frame_stride": "30"},
        {"splits": {"train": [0, 0], "validation": [1], "final": [2]}},
        {"splits": {"train": [0], "validation": [1], "final": [0]}},
        {"splits": {"train": [False], "validation": [1], "final": [2]}},
        {"splits": {"train": [0], "validation": [], "final": [2]}},
        {"splits": {"train": [0], "validation": [1]}},
        {"units": ["degrees"] * 5},
        {"units": [""] * 6},
        {"units": ["x\ny"] * 6},
        {"coordinate_attestation": "guess"},
        {"teacher_path": "/tmp/weights"},
    ],
)
def test_no_coercion_leakage_or_unattested_coordinates(changes):
    with pytest.raises(ValidationError):
        request(native_distillation={**recipe(), **changes})


def configured(root, **updates):
    return Runtime.model_validate(
        {
            "id": "student-cpu",
            "label": "ACT distillation",
            "native_distillation_only": True,
            "native_distillation_python": sys.executable,
            "native_distillation_dataset_python": sys.executable,
            "native_distillation_root": str(root),
            **updates,
        }
    )


def test_runtime_has_no_implicit_engine_or_training_capability(tmp_path):
    root = tmp_path / "workers/policy_distillation"
    runtime = configured(root)
    catalog = RuntimeCatalog(runtimes=[runtime])
    assert not catalog.public()["runtimes"][0]["native_distillation"]
    for path in [
        root / "src/firebird_distill/application.py",
        root / "src/firebird_distill/prepare.py",
        root.parent / "act_optimizer/src/firebird_act/application.py",
        root.parent / "smolvla_qlora/src/firebird_vla/local_dataset.py",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# installed worker marker")
    public = catalog.public()["runtimes"][0]
    assert public["native_distillation"] and public["native_distillation_only"]
    assert not any(public[key] for key in ["training", "run", "simulation", "engine_evaluation"])
    assert "native_distillation_python" not in public
    with pytest.raises(ValueError, match="distillation only"):
        command(runtime, tmp_path / "request", tmp_path / "result", tmp_path, "owned")


@pytest.mark.parametrize(
    "updates",
    [
        {"provider": "gcp"},
        {"execution": "skypilot"},
        {"device": "cuda"},
        {"native_distillation_python": None},
        {"native_distillation_root": None},
        {"native_distillation_dataset_python": None},
        {"native_quantization_only": True},
        {"export_only": True},
        {"training_python": "python"},
        {"simulator_lane": "isaac"},
    ],
)
def test_dedicated_runtime_rejects_conflicting_backends(tmp_path, updates):
    with pytest.raises(ValidationError):
        configured(tmp_path, **updates)
