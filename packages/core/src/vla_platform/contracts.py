"""Public application records. No native ML framework imports belong here."""

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vla_platform.lifecycle.contracts import LifecycleResult, PolicyRequest


def now() -> str:
    return datetime.now(UTC).isoformat()


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ProjectCreate(Record):
    name: str = Field(min_length=1, max_length=100)


class Project(ProjectCreate):
    id: str
    created_at: str


class IntakeRequest(Record):
    source: Literal["huggingface", "local"] = "huggingface"
    repo_id: str | None = Field(default=None, pattern=r"^[\w.-]+/[\w.-]+$", max_length=200)
    revision: str = Field(default="main", min_length=1, max_length=200)
    path: str | None = Field(default=None, max_length=4096)

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
    repo_id: str | None = None
    revision: str
    format: Literal["lerobot_v2", "lerobot_v3"]
    robot_type: str | None = None
    total_episodes: int = Field(ge=0)
    total_frames: int = Field(ge=0)
    fps: float = Field(gt=0, allow_inf_nan=False)
    features: dict[str, Any]
    license: str | None = None
    metadata_sha256: str
    inspected_at: str
    warnings: list[str]
    inspection_scope: Literal["metadata_only"] = "metadata_only"


JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled", "interrupted"]
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


class Job(Record):
    id: str
    project_id: str
    kind: str = "dataset.inspect"
    status: JobStatus = "queued"
    request: IntakeRequest | PolicyRequest
    created_at: str
    updated_at: str
    result: DatasetProfile | LifecycleResult | None = None
    error: str | None = None
    stage: str | None = None


class Capability(Record):
    stage: str
    operation: str
    status: Literal["available", "planned"]
    description: str


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
