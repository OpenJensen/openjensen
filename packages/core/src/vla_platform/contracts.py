"""Public application records. No native ML framework imports belong here."""

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    kind: Literal["dataset.inspect"] = "dataset.inspect"
    status: JobStatus = "queued"
    request: IntakeRequest
    created_at: str
    updated_at: str
    result: DatasetProfile | None = None
    error: str | None = None


class CapabilityTarget(Record):
    operating_system: Literal["linux", "windows", "macos"]
    device: Literal["cpu", "cuda", "mps"]
    support: Literal["supported", "untested", "unsupported"] = "untested"
    evidence_state: Literal["untested", "fixture", "live_source"] = "untested"
    evidence_refs: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def evidence_matches_support(self) -> CapabilityTarget:
        tested = self.evidence_state != "untested"
        if tested != bool(self.evidence_refs) or any(not ref.strip() for ref in self.evidence_refs):
            raise ValueError("Tested capability evidence requires nonempty record references")
        if (self.support == "supported") != tested:
            raise ValueError("Supported targets require scoped run evidence")
        return self


class Capability(Record):
    stage: str
    operation: str
    status: Literal["available", "planned"]
    description: str
    backend: str | None = None
    implementation: Literal["registered", "planned"] = "planned"
    runnable: bool = False
    targets: list[CapabilityTarget] = Field(default_factory=list)

    @model_validator(mode="after")
    def runnable_requires_adapter(self) -> Capability:
        if self.implementation == "planned" and (
            self.status == "available"
            or self.runnable
            or any(target.support == "supported" for target in self.targets)
        ):
            raise ValueError("Planned operations cannot be available, runnable or supported")
        if self.runnable and (
            self.status != "available"
            or self.implementation != "registered"
            or not self.backend
            or not any(target.support != "unsupported" for target in self.targets)
        ):
            raise ValueError(
                "Runnable capability requires an available registered backend and target"
            )
        keys = [(target.operating_system, target.device) for target in self.targets]
        if len(keys) != len(set(keys)):
            raise ValueError("Capability targets must have unique operating-system/device pairs")
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
