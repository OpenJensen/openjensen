"""Provider-neutral execution identity, captured at job admission."""

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field


class ExecutionBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    adapter: str
    infrastructure: str
    target_id: str
    label: str
    runtime_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    target_snapshot: dict | None = Field(default=None, exclude_if=lambda value: value is None)
    connection_sha256: str | None = Field(
        default=None, pattern=r"^[a-f0-9]{64}$", exclude_if=lambda value: value is None
    )
    accelerator: str | None = None
    gpu_count: int = Field(default=0, ge=0)


def runtime_digest(runtime):
    # Include private configuration in the digest, never in the public binding.
    record = runtime.model_dump(mode="json")
    record.pop("label", None)
    if runtime.execution == "skypilot":
        # Infrastructure is already frozen in the accepted compute_target.
        # Hash the worker recipe without mutable catalog placement metadata.
        for key in ("region", "accelerator", "gpu_count", "gpu_name", "gpu_memory_mib"):
            record.pop(key, None)
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def bind_runtime(runtime):
    return ExecutionBinding(
        adapter="skypilot" if runtime.execution == "skypilot" else "local",
        infrastructure=runtime.provider,
        target_id=runtime.id,
        label=runtime.label,
        runtime_sha256=runtime_digest(runtime),
        accelerator=runtime.accelerator or runtime.gpu_name,
        gpu_count=runtime.gpu_count
        if runtime.device == "cuda" or runtime.execution == "skypilot"
        else 0,
    )


def verify_binding(binding, runtime):
    if binding is None:  # Existing jobs retain their original recovery path.
        return
    expected = bind_runtime(runtime)
    if binding.runtime_sha256 != expected.runtime_sha256:
        raise ValueError("The accepted compute target changed. Review it and submit a new job.")
