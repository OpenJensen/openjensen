"""Write receipt equality rejects JSON boolean budgets, including nested recipe values."""

import copy

import pytest
from vla_platform.cli_client import same_json_value, validate_acknowledgment
from vla_platform.tui_client import ApiClient, ApiError


def receipt(training):
    request = {
        "operation": "policy.finetune",
        "runtime_id": "fixture",
        "dataset_job_id": "data",
        "training_method": "full",
        "timeout_seconds": 60,
        "training": training,
    }
    return request, {
        "id": "job",
        "project_id": "p",
        "kind": "policy.finetune",
        "status": "queued",
        "created_at": "2026-09-27T00:00:00Z",
        "updated_at": "2026-09-27T00:00:00Z",
        "request": copy.deepcopy(request),
    }


@pytest.mark.parametrize(
    "submitted,received",
    [
        ({"steps": 1}, {"steps": True}),
        ({"batch_size": 1}, {"batch_size": True}),
        ({"learning_rate": 1.0}, {"learning_rate": True}),
        ({"nested": {"weights": [1, 0]}}, {"nested": {"weights": [True, False]}}),
        ({"enabled": True}, {"enabled": 1}),
    ],
)
def test_boolean_is_not_an_acknowledged_numeric_recipe(submitted, received):
    request, value = receipt(submitted)
    value["request"]["training"] = received
    with pytest.raises(ApiError, match="Outcome unknown"):
        validate_acknowledgment(ApiClient, "/projects/p/policy-jobs", request, value)


def test_real_number_normalization_and_catalog_enrichment_remain_valid():
    request, value = receipt({"steps": 1, "batch_size": 2, "learning_rate": 0.001})
    value["request"]["training"] = {
        "steps": 1.0,
        "batch_size": 2.0,
        "learning_rate": 0.001,
        "model_revision": "pinned",
    }
    validate_acknowledgment(ApiClient, "/projects/p/policy-jobs", request, value)
    assert same_json_value({"x": [1, {"n": 0}]}, {"x": [1.0, {"n": 0.0}]})
    assert not same_json_value([False], [0])
    assert not same_json_value({"a": 1}, {"a": 1, "b": 2})
