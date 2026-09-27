import pytest
from vla_platform.lifecycle.cloud_errors import cloud_launch_error

TARGET = {"accelerator": "A100", "region": "us-central1"}


@pytest.mark.parametrize(
    "diagnostic",
    [
        "ZONE_RESOURCE_POOL_EXHAUSTED",
        "zone_resource_pool_exhausted_with_details",
        "VM_MIN_COUNT_NOT_REACHED: ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS",
    ],
)
def test_stockout_message_names_selected_gpu_and_region(diagnostic):
    assert cloud_launch_error(diagnostic, TARGET) == (
        "Google Cloud has no available A100 capacity in us-central1 right now. "
        "Choose another GPU or try again later."
    )


def test_minimum_instance_count_is_not_itself_a_stockout():
    assert "capacity" not in cloud_launch_error("VM_MIN_COUNT_NOT_REACHED", TARGET)


@pytest.mark.parametrize(
    "diagnostic",
    [
        "QUOTA_EXCEEDED",
        "Quota exceeded for resource",
        "quotaExceeded",
        "Quota 'NVIDIA_A100_GPUS' exceeded. Limit: 0.0 in region us-central1.",
        "VM_MIN_COUNT_NOT_REACHED: QUOTA_EXCEEDED",
        "ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS in previous zone; QUOTA_EXCEEDED now",
    ],
)
def test_quota_failures_are_distinct_from_capacity(diagnostic):
    message = cloud_launch_error(diagnostic, TARGET)
    assert "enough quota to start A100 in us-central1" in message
    assert "capacity" not in message


@pytest.mark.parametrize(
    "diagnostic",
    [
        "PERMISSION_DENIED",
        "403 Forbidden",
        "Access denied: ACCESS_DENIED",
        "Required 'compute.instances.create' permission for this project",
        "ZONE_RESOURCE_POOL_EXHAUSTED earlier; PERMISSION_DENIED now",
    ],
)
def test_permissions_have_actionable_message(diagnostic):
    message = cloud_launch_error(diagnostic, TARGET)
    assert "Google Cloud denied access" in message
    assert "connected account" in message


def test_raw_diagnostics_and_unvalidated_target_values_are_never_echoed():
    secret = "AIza-raw-secret-token"
    result = cloud_launch_error(
        "ZONE_RESOURCE_POOL_EXHAUSTED " + secret,
        {"accelerator": secret, "region": "https://user:password@host/secret"},
    )
    assert secret not in result and "password" not in result and "host" not in result
    assert "GPU capacity in the selected region" in result
    assert len(result) < 250


def test_bounded_tail_captures_final_error_without_echoing_large_output():
    result = cloud_launch_error("sensitive " * 100000 + "QUOTA_EXCEEDED", TARGET)
    assert "enough quota" in result and "sensitive" not in result
    assert len(result) < 250


@pytest.mark.parametrize("output,target", [("", {}), (None, None), ("unknown failure", TARGET)])
def test_unrecognized_failures_use_safe_fallback(output, target):
    assert cloud_launch_error(output, target) == (
        "Cloud training could not start. Check the run details and try again."
    )
