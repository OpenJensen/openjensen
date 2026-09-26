"""Public boundary checks: invalid claims fail, existing valid JSON still round-trips."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from vla_platform.api import create_app
from vla_platform.contracts import (
    Capability,
    CapabilityEvidence,
    CapabilitySupport,
    DatasetProfile,
    IntakeRequest,
    Job,
    Project,
    WorkerRequest,
    WorkerResult,
)
from vla_platform.settings import Settings

REVISION = "ecef85bc07005f771ad86deeff1427f9d72953ed"
DIGEST = "2257f8360a272ff10bff3d5aeeb4d365c9716141dee5a74ecfe7fe5e8cfca6be"
TIMESTAMP = "2026-09-26T12:40:17.113090+00:00"


@pytest.fixture
def profile_data() -> dict[str, Any]:
    """Representative saved v1 profile; counts/features are fixture data."""
    return {
        "schema_version": 1,
        "source": "huggingface",
        "repo_id": "codywang/so101_pickup_test",
        "revision": REVISION,
        "format": "lerobot_v3",
        "robot_type": "so_follower",
        "total_episodes": 30,
        "total_frames": 4500,
        "fps": 30.0,
        "features": {"action": {"dtype": "float32", "shape": [6]}},
        "license": "apache-2.0",
        "metadata_sha256": DIGEST,
        "inspected_at": TIMESTAMP,
        "warnings": ["Metadata-only fixture; no task performance claim."],
        "inspection_scope": "metadata_only",
    }


@pytest.fixture
def job_data(profile_data: dict[str, Any]) -> dict[str, Any]:
    """Use opaque legacy IDs as well as UUID-shaped production IDs."""
    return {
        "id": "legacy-job",
        "project_id": "legacy-project",
        "kind": "dataset.inspect",
        "status": "succeeded",
        "request": IntakeRequest(repo_id=profile_data["repo_id"], revision=REVISION).model_dump(),
        "created_at": TIMESTAMP,
        "updated_at": TIMESTAMP,
        "result": profile_data,
        "error": None,
    }


def evidence_data() -> dict[str, str]:
    """A structural fixture, not a real capability support claim."""
    return {
        "reference": "fixture://reviewed-run",
        "source_revision": "89fa8ea7b8458b3874a5f14b9abd9d78383a10d1",
        "runtime": "fixture-runtime-lock-sha256",
        "device_name": "fixture CPU",
        "recorded_at": TIMESTAMP,
    }


def support_data(state: str = "tested") -> dict[str, Any]:
    return {
        "backend": "metadata",
        "os": "macos",
        "device": "cpu",
        "evidence_state": state,
        "evidence": [evidence_data()] if state == "tested" else [],
    }


def capability_data() -> dict[str, Any]:
    return {
        "stage": "Dataset",
        "operation": "dataset.inspect",
        "status": "available",
        "description": "Metadata inspection only.",
        "support": [],
    }


def test_phase_a_e6_counterexamples_are_rejected() -> None:
    examples = [
        (
            Capability,
            {
                "stage": "Imaginary",
                "operation": "arbitrary.unsupported",
                "status": "available",
                "description": "x",
            },
        ),
        (Project, {"name": "x", "id": "", "created_at": "not-a-date"}),
        (
            DatasetProfile,
            {
                "source": "huggingface",
                "repo_id": None,
                "revision": "",
                "format": "lerobot_v3",
                "total_episodes": 0,
                "total_frames": 0,
                "fps": 1,
                "features": {},
                "metadata_sha256": "",
                "inspected_at": "not-a-date",
                "warnings": [],
            },
        ),
    ]
    for model, payload in examples:
        with pytest.raises(ValidationError):
            model.model_validate(payload)


@pytest.mark.parametrize("value", ["", "   ", None])
def test_project_identity_required(value: Any) -> None:
    with pytest.raises(ValidationError):
        Project.model_validate({"name": "Project", "id": value, "created_at": TIMESTAMP})


@pytest.mark.parametrize("field", ["id", "project_id"])
@pytest.mark.parametrize("value", ["", "  ", None])
def test_job_identity_required(job_data: dict[str, Any], field: str, value: Any) -> None:
    job_data[field] = value
    with pytest.raises(ValidationError):
        Job.model_validate(job_data)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "not-a-date",
        "2026-09-26",
        "2026-09-26T12:00:00",
        "2026-02-30T12:00:00Z",
        "2026-09-26T12:00:00+99:00",
        "2026-09-26T12:00:00+24:00",
        "2026-09-26T12:00:00+00:60",
        "2026-09-26T12:00:00+01:99",
        "2026-09-26T12:00:00-00:60",
    ],
)
def test_timestamps_require_valid_date_and_offset(value: str) -> None:
    with pytest.raises(ValidationError):
        Project(name="Project", id="p", created_at=value)


@pytest.mark.parametrize(
    "value",
    [
        TIMESTAMP,
        "2026-09-26T12:40:17Z",
        "2026-09-26T16:40:17+04:00",
        "2026-09-26T12:40:17+00:59",
        "2026-09-26T12:40:17-00:59",
    ],
)
def test_timestamp_text_is_preserved(value: str) -> None:
    project = Project(name="Project", id="p", created_at=value)
    assert project.model_dump()["created_at"] == value
    assert Project.model_validate_json(project.model_dump_json()) == project


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repo_id", None),
        ("repo_id", ""),
        ("repo_id", "not-a-repository"),
        ("revision", ""),
        ("revision", "  "),
        ("revision", "main"),
        ("revision", "a" * 39),
        ("revision", "g" * 40),
        ("metadata_sha256", ""),
        ("metadata_sha256", "a" * 63),
        ("metadata_sha256", "g" * 64),
        ("features", {}),
        ("inspected_at", ""),
    ],
)
def test_hf_profile_requires_identity_and_provenance(
    profile_data: dict[str, Any], field: str, value: Any
) -> None:
    profile_data[field] = value
    with pytest.raises(ValidationError):
        DatasetProfile.model_validate(profile_data)


def test_pinned_hf_and_local_profiles_round_trip(profile_data: dict[str, Any]) -> None:
    assert DatasetProfile.model_validate_json(json.dumps(profile_data)).model_dump() == profile_data
    profile_data.update(source="local", repo_id=None, revision=f"metadata-sha256:{DIGEST}")
    assert DatasetProfile.model_validate_json(json.dumps(profile_data)).model_dump() == profile_data


@pytest.mark.parametrize(
    ("repo_id", "revision"),
    [
        ("owner/data", f"metadata-sha256:{DIGEST}"),
        (None, REVISION),
        (None, "metadata-sha256:" + "a" * 64),
        (None, ""),
    ],
)
def test_local_profile_cannot_claim_unrelated_identity(
    profile_data: dict[str, Any], repo_id: str | None, revision: str
) -> None:
    profile_data.update(source="local", repo_id=repo_id, revision=revision)
    with pytest.raises(ValidationError):
        DatasetProfile.model_validate(profile_data)


def test_saved_job_and_project_shape_is_compatible(job_data: dict[str, Any]) -> None:
    assert Job.model_validate_json(json.dumps(job_data)).model_dump() == job_data
    project = {"id": "legacy-project", "name": "Project", "created_at": TIMESTAMP}
    assert Project.model_validate_json(json.dumps(project)).model_dump() == project


@pytest.mark.parametrize(
    "payload",
    [
        {"repo_id": "owner/data", "revision": "  "},
        {"source": "local", "path": "  "},
        {"source": "local", "path": ".", "repo_id": "owner/data"},
        {"source": "huggingface", "repo_id": "owner/data", "path": "."},
    ],
)
def test_invalid_intake_source_identity_is_rejected(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        IntakeRequest.model_validate(payload)


@pytest.mark.parametrize("version", [0, 2, 99, "1"])
def test_unknown_worker_protocol_versions_fail(version: Any) -> None:
    with pytest.raises(ValidationError):
        WorkerRequest.model_validate({"schema_version": version, "intake": {"repo_id": "a/b"}})
    with pytest.raises(ValidationError):
        WorkerResult.model_validate({"schema_version": version, "error": "failure"})


def test_worker_operation_and_result_boundary(profile_data: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        WorkerRequest.model_validate({"operation": "shell.exec", "intake": {"repo_id": "a/b"}})
    for data in [{}, {"result": profile_data, "error": "failure"}]:
        with pytest.raises(ValidationError):
            WorkerResult.model_validate(data)
    invalid = deepcopy(profile_data)
    invalid["metadata_sha256"] = ""
    with pytest.raises(ValidationError):
        WorkerResult.model_validate({"result": invalid})
    result = WorkerResult.model_validate({"result": profile_data})
    assert WorkerResult.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("operation", "arbitrary.unsupported"),
        ("stage", "Imaginary"),
        ("stage", "Run"),
        ("description", "  "),
        ("status", "untested"),
        ("schema_version", 99),
    ],
)
def test_capability_rejects_invalid_claims(field: str, value: Any) -> None:
    data = capability_data()
    data[field] = value
    with pytest.raises(ValidationError):
        Capability.model_validate(data)


@pytest.mark.parametrize(
    ("stage", "operation"),
    [
        ("Fine-tune", "policy.finetune"),
        ("Distill", "policy.distill"),
        ("Quantize", "policy.quantize"),
        ("Evaluate", "policy.evaluate"),
        ("Run", "policy.run"),
    ],
)
def test_catalog_operations_cannot_advertise_execution(stage: str, operation: str) -> None:
    data = {
        "stage": stage,
        "operation": operation,
        "status": "planned",
        "description": "Later",
        "support": [],
    }
    assert Capability.model_validate(data).status == "planned"
    data["status"] = "available"
    with pytest.raises(ValidationError, match="not registered"):
        Capability.model_validate(data)
    # Even a syntactically valid evidence record cannot register executable code.
    native = support_data()
    native.update(backend="lerobot", os="windows", device="cuda")
    with pytest.raises(ValidationError, match="not registered"):
        Capability.model_validate({**data, "support": [native]})


@pytest.mark.parametrize(
    "field", ["reference", "source_revision", "runtime", "device_name", "recorded_at"]
)
def test_test_evidence_requires_provenance(field: str) -> None:
    data = evidence_data()
    data[field] = ""
    with pytest.raises(ValidationError):
        CapabilityEvidence.model_validate(data)


def test_capability_requires_explicit_support_but_allows_unknown_coverage() -> None:
    data = capability_data()
    assert Capability.model_validate(data).support == []
    del data["support"]
    with pytest.raises(ValidationError, match="support"):
        Capability.model_validate(data)
    with pytest.raises(ValidationError, match="support"):
        Capability.model_validate({**data, "support": None})


@pytest.mark.parametrize("field", ["backend", "os", "device", "evidence_state"])
def test_support_target_fields_are_required(field: str) -> None:
    data = support_data()
    del data[field]
    with pytest.raises(ValidationError, match=field):
        CapabilitySupport.model_validate(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend", "unknown"),
        ("os", "any"),
        ("device", "any"),
        ("device", "cuda"),
        ("evidence_state", "available"),
        ("evidence", []),
    ],
)
def test_support_requires_concrete_target_and_test_evidence(field: str, value: Any) -> None:
    data = support_data()
    data[field] = value
    with pytest.raises(ValidationError):
        CapabilitySupport.model_validate(data)


@pytest.mark.parametrize("state", ["planned", "untested"])
def test_planned_and_untested_targets_cannot_imply_tested_support(state: str) -> None:
    data = support_data(state)
    assert CapabilitySupport.model_validate(data).evidence == []
    with pytest.raises(ValidationError, match="at least one tested"):
        Capability.model_validate({**capability_data(), "support": [data]})
    data["evidence"] = [evidence_data()]
    with pytest.raises(ValidationError, match="cannot carry test evidence"):
        CapabilitySupport.model_validate(data)


def test_support_is_scoped_not_broadcast_to_other_targets() -> None:
    tested = support_data()
    untested = support_data("untested")
    untested["os"] = "windows"
    capability = Capability.model_validate({**capability_data(), "support": [tested, untested]})
    assert [s.evidence_state for s in capability.support] == ["tested", "untested"]
    assert capability.support[1].evidence == []
    assert Capability.model_validate_json(capability.model_dump_json()) == capability


def test_duplicate_and_wrong_backend_targets_fail() -> None:
    entry = support_data()
    with pytest.raises(ValidationError, match="unique"):
        Capability.model_validate({**capability_data(), "support": [entry, entry]})
    entry["backend"] = "lerobot"
    with pytest.raises(ValidationError, match="operation family"):
        Capability.model_validate({**capability_data(), "support": [entry]})


def test_unsupported_target_is_not_available_and_can_reference_failure() -> None:
    entry = support_data("unsupported")
    entry["evidence"] = [evidence_data()]
    data = {**capability_data(), "status": "planned", "support": [entry]}
    assert Capability.model_validate(data).support[0].evidence_state == "unsupported"
    data["status"] = "available"
    with pytest.raises(ValidationError, match="at least one tested"):
        Capability.model_validate(data)


@pytest.mark.parametrize("local_enabled", [False, True])
def test_current_api_is_compatible_without_fabricated_target_evidence(
    tmp_path: Path, local_enabled: bool
) -> None:
    settings = Settings(data_dir=tmp_path, local_root=tmp_path if local_enabled else None)
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v1/capabilities")
        assert response.status_code == 200
        assert all("support" in item for item in response.json())
        records = [Capability.model_validate(item) for item in response.json()]
        assert all(item.support == [] for item in records)
        available = {item.operation for item in records if item.status == "available"}
        assert available == (
            {"dataset.inspect", "dataset.inspect.local"} if local_enabled else {"dataset.inspect"}
        )


def test_openapi_exposes_the_support_and_identity_contract(tmp_path: Path) -> None:
    schemas = create_app(Settings(data_dir=tmp_path)).openapi()["components"]["schemas"]
    assert "support" in schemas["Capability"]["required"]
    assert "arbitrary.unsupported" not in schemas["Capability"]["properties"]["operation"]["enum"]
    assert schemas["Project"]["properties"]["id"]["minLength"] == 1
    assert schemas["Job"]["properties"]["project_id"]["minLength"] == 1
    assert (
        schemas["DatasetProfile"]["properties"]["metadata_sha256"]["pattern"] == r"^[a-f0-9]{64}$"
    )
    assert set(schemas["CapabilitySupport"]["required"]) == {
        "backend",
        "os",
        "device",
        "evidence_state",
    }
    assert schemas["Capability"]["properties"]["support"]["items"]["$ref"].endswith(
        "/CapabilitySupport"
    )


def test_extra_worker_fields_do_not_become_executable_input() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        WorkerRequest.model_validate({"intake": {"repo_id": "a/b"}, "command": "echo forged"})
