"""Public application records. No native ML framework imports belong here."""

import json
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    model_validator,
)

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


class LocalPreviewLimits(Record):
    max_rows: int = Field(default=10, ge=1, le=100, strict=True)
    max_file_bytes: int = Field(default=4194304, ge=1, le=67108864, strict=True)
    max_read_bytes: int = Field(default=16777216, ge=1, le=134217728, strict=True)
    max_decoded_bytes: int = Field(default=8388608, ge=1, le=67108864, strict=True)
    max_output_bytes: int = Field(default=65536, ge=1024, le=1048576, strict=True)
    timeout_seconds: float = Field(default=10.0, gt=0, le=30, strict=True, allow_inf_nan=False)


class LocalPreviewRequest(Record):
    source: Literal["local"] = "local"
    path: NonEmptyString = Field(max_length=4096)
    kind: Literal["frames", "episodes"] = "frames"
    parquet_path: NonEmptyString | None = Field(default=None, max_length=4096)
    limits: LocalPreviewLimits = Field(default_factory=LocalPreviewLimits)


class LocalPreviewFile(Record):
    path: NonEmptyString = Field(max_length=4096)
    size_bytes: int = Field(ge=0, strict=True)
    sha256: Sha256


class LocalDatasetPreview(Record):
    # Preserve source row text; never normalize task strings in the result.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False, allow_inf_nan=False)
    preview_schema_version: Literal[1] = 1
    inspection_scope: Literal["bounded_parquet_rows"] = "bounded_parquet_rows"
    source: Literal["local"] = "local"
    format: Literal["lerobot_v3"] = "lerobot_v3"
    kind: Literal["frames", "episodes"]
    metadata_sha256: Sha256
    file: LocalPreviewFile
    reader: Literal["pyarrow==25.0.1"]
    total_file_rows: int = Field(ge=0, strict=True)
    row_offset: Literal[0] = 0
    rows: list[dict[str, JsonValue]] = Field(max_length=100)
    returned_rows: int = Field(ge=0, le=100, strict=True)
    truncated: bool = Field(strict=True)
    columns: list[NonEmptyString] = Field(min_length=1, max_length=7)
    omitted_columns: list[NonEmptyString]
    read_bytes: int = Field(ge=0, strict=True)
    declared_decoded_bytes: int = Field(ge=0, strict=True)
    limits: LocalPreviewLimits
    warnings: list[NonEmptyString] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_preview(self) -> LocalDatasetPreview:
        if self.returned_rows != len(self.rows):
            raise ValueError("returned_rows must match rows")
        if self.returned_rows > min(self.total_file_rows, self.limits.max_rows):
            raise ValueError("Preview exceeds its row budget or declared file rows")
        if self.truncated != (self.returned_rows < self.total_file_rows):
            raise ValueError("truncated must reflect this file's returned rows")
        if self.file.size_bytes > self.limits.max_file_bytes:
            raise ValueError("Preview exceeds its stored file byte budget")
        if self.read_bytes > self.limits.max_read_bytes:
            raise ValueError("Preview exceeds its aggregate read byte budget")
        if self.declared_decoded_bytes > self.limits.max_decoded_bytes:
            raise ValueError("Preview exceeds its declared decode byte budget")
        columns = set(self.columns)
        if len(columns) != len(self.columns) or columns.intersection(self.omitted_columns):
            raise ValueError("Invalid preview column selection")
        if any(set(row) != columns for row in self.rows):
            raise ValueError("Rows must match the selected columns")
        parts = self.file.path.split("/")
        prefix = ["data"] if self.kind == "frames" else ["meta", "episodes"]
        if (
            parts[: len(prefix)] != prefix
            or any(p in ("", ".", "..") for p in parts)
            or "\\" in self.file.path
            or not self.file.path.endswith(".parquet")
        ):
            raise ValueError("Expected a contained relative Parquet file path")
        # This is the helper's canonical UTF-8 JSON encoding, not pretty JSON.
        encoded = json.dumps(
            self.model_dump(mode="json"), ensure_ascii=True, allow_nan=False, separators=(",", ":")
        ).encode()
        if len(encoded) > self.limits.max_output_bytes:
            raise ValueError("Preview exceeds its serialized JSON byte budget")
        return self


class LocalPreviewFailure(Record):
    code: Literal[
        "invalid_limits",
        "local_disabled",
        "invalid_request",
        "unsafe_path",
        "reader_unavailable",
        "reader_version",
        "reader_failed",
        "timeout",
        "output_limit",
        "unsupported_platform",
        "missing_file",
        "read_limit",
        "file_limit",
        "file_changed",
        "invalid_metadata",
        "decoded_limit",
        "unsupported_columns",
        "corrupt_payload",
    ]
    message: NonEmptyString = Field(max_length=300)


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
    inspection_scope: Literal["metadata_only", "bounded_parquet_rows"] = "metadata_only"
    # Keep existing v1 metadata-only JSON unchanged when there is no preview.
    preview: LocalDatasetPreview | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

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
        if self.inspection_scope == "metadata_only":
            if self.preview is not None:
                raise ValueError("Metadata-only profiles cannot carry a row preview")
        elif (
            self.preview is None
            or self.source != "local"
            or self.format != "lerobot_v3"
            or self.preview.metadata_sha256 != self.metadata_sha256
        ):
            raise ValueError("Bounded previews require matching local v3 metadata identity")
        return self


JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled", "interrupted"]
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


class Job(Record):
    id: NonEmptyString
    project_id: NonEmptyString
    kind: Literal["dataset.inspect"] = "dataset.inspect"
    status: JobStatus = "queued"
    request: IntakeRequest
    created_at: Timestamp
    updated_at: Timestamp
    result: DatasetProfile | None = None
    error: str | None = None


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
IMPLEMENTED_OPERATIONS = frozenset({"dataset.inspect", "dataset.inspect.local"})


class CapabilityEvidence(Record):
    """Reference to reviewed evidence; validation does not execute or verify it."""

    reference: NonEmptyString
    source_revision: GitRevision
    runtime: NonEmptyString
    device_name: NonEmptyString
    recorded_at: Timestamp


class CapabilitySupport(Record):
    """Evidence for one parent operation on one concrete execution target."""

    backend: Literal["metadata", "lerobot", "openvla_oft"]
    os: Literal["linux", "windows", "macos"]
    device: Literal["cpu", "cuda"]
    evidence_state: Literal["tested", "unsupported"]
    evidence: list[CapabilityEvidence] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_evidence(self) -> CapabilitySupport:
        """Both successful and unsupported target claims require evidence identity."""
        if self.backend == "metadata" and self.device != "cpu":
            raise ValueError("The metadata backend only registers CPU operations")
        return self


class Capability(Record):
    """Planned/untested records carry no support; outcome claims carry evidence."""

    # Expose the same conditional requirements to OpenAPI/TypeScript consumers.
    # Shared target identity/evidence constraints come from the support field below.
    model_config = ConfigDict(
        json_schema_extra={
            "oneOf": [
                {
                    "properties": {
                        "status": {"enum": ["planned", "untested"]},
                        "support": {"type": "array", "maxItems": 0},
                    },
                    "required": ["status"],
                },
                {
                    "properties": {
                        "status": {"const": "available"},
                        "support": {
                            "type": "array",
                            "minItems": 1,
                            "contains": {
                                "properties": {"evidence_state": {"const": "tested"}},
                                "required": ["evidence_state"],
                            },
                        },
                    },
                    "required": ["status", "support"],
                },
                {
                    "properties": {
                        "status": {"const": "unsupported"},
                        "support": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "properties": {"evidence_state": {"const": "unsupported"}},
                                "required": ["evidence_state"],
                            },
                        },
                    },
                    "required": ["status", "support"],
                },
            ]
        }
    )

    schema_version: Literal[1] = 1
    stage: Stage
    operation: Operation
    status: Literal["planned", "untested", "available", "unsupported"]
    description: NonEmptyString
    support: list[CapabilitySupport] = Field(default_factory=list)

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
        if self.status in {"planned", "untested"}:
            if self.support:
                raise ValueError("Planned or untested capability cannot carry support")
        elif not self.support:
            raise ValueError("Available or unsupported capability requires nonempty support")
        elif self.status == "available":
            if not any(entry.evidence_state == "tested" for entry in self.support):
                raise ValueError("Available target support requires at least one tested target")
        elif any(entry.evidence_state != "unsupported" for entry in self.support):
            raise ValueError("Unsupported capability must carry only unsupported targets")
        return self


class EpisodeSummary(Record):
    episode_index: int = Field(ge=0)
    frame_count: int = Field(ge=0)
    duration_seconds: float = Field(ge=0, allow_inf_nan=False)
    tasks: list[str]


class EpisodePage(Record):
    repo_id: str
    revision: str
    total_episodes: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=24)
    episodes: list[EpisodeSummary]
    warnings: list[str]


class CameraPreview(Record):
    key: str
    url: str
    start_seconds: float = Field(ge=0, allow_inf_nan=False)
    end_seconds: float = Field(ge=0, allow_inf_nan=False)
    width: int | None
    height: int | None
    fps: float = Field(gt=0, allow_inf_nan=False)


class FrameSample(Record):
    frame_index: int = Field(ge=0)
    timestamp: float = Field(ge=0, allow_inf_nan=False)
    action: list[float] | None
    state: list[float] | None


class EpisodePreview(EpisodeSummary):
    repo_id: str
    revision: str
    cameras: list[CameraPreview]
    action_names: list[str]
    state_names: list[str]
    samples: list[FrameSample]
    warnings: list[str]


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
