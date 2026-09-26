"""Public application records. No native ML framework imports belong here."""

from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from vla_platform.lifecycle.contracts import LifecycleResult, PolicyRequest

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
RepositoryId = Annotated[
    str, StringConstraints(strip_whitespace=True, pattern=r"^[\w.-]+/[\w.-]+$", max_length=200)
]
GitRevision = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{40}$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]


def _timestamp(value: str) -> str:
    """Validate an offset-bearing ISO timestamp without changing its stored text."""
    datetime.fromisoformat(value.upper())
    return value


Timestamp = Annotated[
    str,
    StringConstraints(
        pattern=(
            r"^\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?"
            r"(?:[Zz]|[+-](?:[01]\d|2[0-3]):[0-5]\d)$"
        )
    ),
    AfterValidator(_timestamp),
    Field(json_schema_extra={"format": "date-time"}),
]


def now() -> str:
    return datetime.now(UTC).isoformat()


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ProjectCreate(Record):
    name: str = Field(min_length=1, max_length=100)


class Project(ProjectCreate):
    id: NonEmptyString
    created_at: Timestamp


class IntakeRequest(Record):
    source: Literal["huggingface", "local"] = "huggingface"
    repo_id: RepositoryId | None = None
    revision: NonEmptyString = Field(default="main", max_length=200)
    path: NonEmptyString | None = Field(default=None, max_length=4096)

    @model_validator(mode="after")
    def validate_source(self) -> IntakeRequest:
        if self.source == "huggingface" and (not self.repo_id or self.path is not None):
            raise ValueError("Hugging Face intake requires repo_id and no local path")
        if self.source == "local" and (not self.path or self.repo_id is not None):
            raise ValueError("Local intake requires a path and no repo_id")
        return self


class DatasetProfile(Record):
    schema_version: Literal[1] = 1
    source: Literal["huggingface", "local"]
    repo_id: RepositoryId | None = None
    revision: NonEmptyString
    format: Literal["lerobot_v2", "lerobot_v3"]
    robot_type: str | None = None
    total_episodes: int = Field(ge=0)
    total_frames: int = Field(ge=0)
    fps: float = Field(gt=0, allow_inf_nan=False)
    features: dict[str, Any] = Field(min_length=1)
    license: str | None = None
    metadata_sha256: Sha256
    inspected_at: Timestamp
    warnings: list[str]
    inspection_scope: Literal["metadata_only"] = "metadata_only"

    @model_validator(mode="after")
    def validate_provenance(self) -> DatasetProfile:
        """Require pinned HF identity or the matching local metadata identity."""
        if self.source == "huggingface":
            if self.repo_id is None:
                raise ValueError("Hugging Face profiles require repo_id")
            if len(self.revision) != 40 or any(c not in "0123456789abcdef" for c in self.revision):
                raise ValueError("Hugging Face profiles require an immutable 40-character revision")
        elif self.repo_id is not None or self.revision != f"metadata-sha256:{self.metadata_sha256}":
            raise ValueError(
                "Local profiles require no repo_id and a matching metadata-sha256 revision"
            )
        return self


JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled", "interrupted"]
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


class Job(Record):
    id: NonEmptyString
    project_id: NonEmptyString
    kind: NonEmptyString = "dataset.inspect"
    status: JobStatus = "queued"
    request: IntakeRequest | PolicyRequest
    created_at: Timestamp
    updated_at: Timestamp
    result: DatasetProfile | LifecycleResult | None = None
    error: str | None = None
    stage: str | None = None

    @model_validator(mode="after")
    def validate_operation(self):
        expected = (
            self.request.operation if isinstance(self.request, PolicyRequest) else "dataset.inspect"
        )
        if self.kind != expected:
            raise ValueError("Job kind must match its registered request operation")
        if self.result is not None and isinstance(self.request, PolicyRequest) != isinstance(
            self.result, LifecycleResult
        ):
            raise ValueError("Job result must match its operation family")
        return self


Stage = Literal["Dataset", "Fine-tune", "Distill", "Quantize", "Evaluate", "Run"]
Operation = Literal[
    "dataset.inspect",
    "dataset.inspect.local",
    "policy.finetune",
    "policy.distill",
    "policy.quantize",
    "policy.evaluate",
    "policy.run",
]
OPERATION_STAGES: dict[Operation, Stage] = {
    "dataset.inspect": "Dataset",
    "dataset.inspect.local": "Dataset",
    "policy.finetune": "Fine-tune",
    "policy.distill": "Distill",
    "policy.quantize": "Quantize",
    "policy.evaluate": "Evaluate",
    "policy.run": "Run",
}
# Catalog vocabulary is not executable registration. Extend this only alongside
# a reviewed application/worker integration, never from an incoming capability.
IMPLEMENTED_OPERATIONS = frozenset(
    {
        "dataset.inspect",
        "dataset.inspect.local",
        "policy.finetune",
        "policy.quantize",
        "policy.evaluate",
        "policy.run",
    }
)


class CapabilityEvidence(Record):
    """Reference to reviewed evidence; validation does not execute or verify it."""

    reference: NonEmptyString
    source_revision: GitRevision
    runtime: NonEmptyString
    device_name: NonEmptyString
    recorded_at: Timestamp


class CapabilitySupport(Record):
    """Evidence for one parent operation on one concrete execution target."""

    backend: Literal["metadata", "lerobot", "openvla_oft", "vla_cpp"]
    os: Literal["linux", "windows", "macos"]
    device: Literal["cpu", "cuda"]
    evidence_state: Literal["planned", "untested", "tested", "unsupported"]
    evidence: list[CapabilityEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_evidence(self) -> CapabilitySupport:
        """Keep an untested registration distinct from a tested target."""
        if self.evidence_state == "tested" and not self.evidence:
            raise ValueError("Tested support requires evidence")
        if self.evidence_state in {"planned", "untested"} and self.evidence:
            raise ValueError("Planned or untested support cannot carry test evidence")
        if self.backend == "metadata" and self.device != "cpu":
            raise ValueError("The metadata backend only registers CPU operations")
        return self


class Capability(Record):
    """Application availability plus separately scoped support evidence.

    Explicit empty support denotes unknown target coverage, not tested support.
    Callers must supply the field; CAP-001 must populate reviewed target records.
    """

    schema_version: Literal[1] = 1
    stage: Stage
    operation: Operation
    status: Literal["available", "planned"]
    description: NonEmptyString
    support: list[CapabilitySupport]

    @model_validator(mode="after")
    def validate_operation(self) -> Capability:
        """Reject unknown/mismatched operations and unsupported availability."""
        if self.stage != OPERATION_STAGES[self.operation]:
            raise ValueError("Capability stage must match its operation")
        if self.status == "available" and self.operation not in IMPLEMENTED_OPERATIONS:
            raise ValueError("Operation is not registered for application execution")
        targets = [(entry.backend, entry.os, entry.device) for entry in self.support]
        if len(targets) != len(set(targets)):
            raise ValueError("Capability support targets must be unique")
        is_metadata = self.operation in {"dataset.inspect", "dataset.inspect.local"}
        if any((entry.backend == "metadata") != is_metadata for entry in self.support):
            raise ValueError("Support backend must match the operation family")
        if self.status == "available" and self.support:
            if not any(entry.evidence_state == "tested" for entry in self.support):
                raise ValueError("Available target support requires at least one tested target")
        return self


class WorkerRequest(Record):
    schema_version: Literal[1] = 1
    operation: Literal["dataset.inspect"] = "dataset.inspect"
    intake: IntakeRequest
    local_root: str | None = None


class WorkerResult(Record):
    schema_version: Literal[1] = 1
    result: DatasetProfile | None = None
    error: str | None = None

    @model_validator(mode="after")
    def exactly_one_outcome(self) -> WorkerResult:
        if (self.result is None) == (self.error is None):
            raise ValueError("Worker must return exactly one of result or error")
        return self
