"""Turn terminal provisioning diagnostics into bounded, credential-free messages."""

import re

_GPUS = {"A100", "A100-80GB", "L4", "T4"}
_REGION = re.compile(
    r"(?:us|europe|asia|australia|northamerica|southamerica|me|africa)-[a-z]+[1-9][0-9]?"
)
_FALLBACK = "Cloud training could not start. Check the run details and try again."


def cloud_launch_error(output: str, target: dict) -> str:
    """Classify a final failed launch; callers must let zone retries finish first.

    Raw diagnostics are never interpolated. Prefer explicit permission/quota
    errors over a stockout from an earlier zone in the same launch attempt.
    """
    if not isinstance(output, str):
        return _FALLBACK
    diagnostic = output[-128 * 1024 :].lower()
    target = target if isinstance(target, dict) else {}
    accelerator = target.get("accelerator")
    gpu = accelerator if isinstance(accelerator, str) and accelerator in _GPUS else "GPU"
    region = target.get("region")
    location = (
        region
        if isinstance(region, str) and len(region) <= 64 and _REGION.fullmatch(region)
        else "the selected region"
    )
    if any(
        marker in diagnostic
        for marker in (
            "permission_denied",
            "permission denied",
            "does not have permission",
            "accessnotconfigured",
            "access_denied",
            "forbidden",
        )
    ) or re.search(r"required\s+['\"][a-z0-9_.]+['\"]\s+permission", diagnostic):
        return (
            "Google Cloud denied access to start this GPU. "
            "Check the connected account's access to the project and try again."
        )
    if (
        "quota_exceeded" in diagnostic
        or "quota exceeded" in diagnostic
        or "quotaexceeded" in diagnostic
        or re.search(r"quota\s+['\"][^'\"\n]{1,128}['\"]\s+(?:was\s+)?exceeded", diagnostic)
    ):
        return (
            f"Your Google Cloud project does not have enough quota to start {gpu} in {location}. "
            "Choose another GPU or request more quota."
        )
    if "zone_resource_pool_exhausted" in diagnostic:
        return (
            f"Google Cloud has no available {gpu} capacity in {location} right now. "
            "Choose another GPU or try again later."
        )
    # VM_MIN_COUNT_NOT_REACHED alone also occurs for quota and other failures;
    # it is deliberately not evidence of a capacity stockout without the pool code.
    return _FALLBACK
