"""GPU-independent application lifecycle contracts."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Precision(StrictRecord):
    language: Literal["Q4_0", "Q8_0"] = "Q8_0"
    vision: Literal["Q8_0"] | None = None


class ParityLimits(StrictRecord):
    """Operator-declared tolerances; no unmeasured universal default."""

    profile: str = Field(min_length=1, max_length=120)
    max_rmse: float = Field(ge=0, allow_inf_nan=False)
    max_abs_error: float = Field(ge=0, allow_inf_nan=False)


class Evaluation(StrictRecord):
    mode: Literal["engine", "libero"] = "engine"
    suite: Literal["libero_object", "libero_spatial"] = "libero_object"
    task_ids: list[int] | None = Field(default=None, min_length=1, max_length=10)
    parity_limits: ParityLimits | None = None
    warmups: int = Field(default=3, ge=1, le=100)
    repetitions: int = Field(default=10, ge=2, le=1000)
    task_id: int = Field(default=0, ge=0, le=9)
    initial_states: list[int] = Field(default_factory=lambda: [0, 1], min_length=1, max_length=50)
    final_states: list[int] = Field(default_factory=lambda: [2, 3], min_length=1, max_length=50)
    seed: int = Field(default=42, ge=0, le=2147483647)
    steps: int = Field(default=500, ge=1, le=500)

    @model_validator(mode="after")
    def disjoint(self):
        if self.suite == "libero_object" and self.task_ids is not None:
            raise ValueError("Object evaluation uses task_id, not task_ids")
        if self.suite == "libero_spatial":
            if "steps" not in self.model_fields_set:
                self.steps = 280
            if self.steps > 280:
                raise ValueError("Spatial evaluation supports at most its 280-step horizon")
            if self.task_ids is None:
                self.task_ids = list(range(10))
            if len(self.task_ids) != len(set(self.task_ids)) or any(
                task < 0 or task > 9 for task in self.task_ids
            ):
                raise ValueError("Spatial task IDs must be unique integers in 0..9")
        for states in (self.initial_states, self.final_states):
            if len(states) != len(set(states)) or any(x < 0 or x > 999 for x in states):
                raise ValueError("Evaluation state IDs must be unique integers in 0..999")
        if set(self.initial_states) & set(self.final_states):
            raise ValueError("Search and final evaluation states must be disjoint")
        return self


class Limits(StrictRecord):
    min_success_rate: float = Field(default=1.0, ge=0, le=1, allow_inf_nan=False)
    max_success_drop: float = Field(default=0.0, ge=0, le=1, allow_inf_nan=False)
    max_p95_ms: float = Field(default=1000.0, gt=0, allow_inf_nan=False)
    max_peak_device_mib: float = Field(default=8192.0, gt=0, allow_inf_nan=False)


class PolicyRequest(StrictRecord):
    operation: Literal[
        "policy.import",
        "policy.finetune",
        "policy.export",
        "policy.quantize",
        "policy.evaluate",
        "policy.run",
        "policy.workflow",
    ]
    runtime_id: str = Field(min_length=1, max_length=100, pattern=r"^[\w-]+$")
    source_id: str | None = Field(default=None, max_length=100, pattern=r"^[\w-]+$")
    artifact_id: str | None = None
    dataset_job_id: str | None = None
    resume_job_id: str | None = None
    training_method: str = Field(default="lora", pattern=r"^[\w-]+$")
    training: dict[str, Any] | None = None
    precision: Precision | None = None
    candidates: list[Precision] = Field(
        default_factory=lambda: [Precision(language="Q8_0")],
        min_length=1,
        max_length=4,
    )
    evaluation: Evaluation = Field(default_factory=Evaluation)
    limits: Limits | None = None
    timeout_seconds: int = Field(default=7200, ge=30, le=86400)

    @model_validator(mode="after")
    def input_contract(self):
        training = self.operation == "policy.finetune" or (
            self.operation == "policy.workflow" and self.training is not None
        )
        if self.resume_job_id and not training:
            raise ValueError("Only training operations can resume a checkpoint")
        if self.operation == "policy.import":
            if bool(self.source_id) == bool(self.artifact_id) or self.training is not None:
                raise ValueError(
                    "Import requires one configured source or native checkpoint artifact"
                )
        elif self.operation == "policy.finetune":
            if not self.dataset_job_id:
                raise ValueError("Fine-tuning requires a completed dataset intake")
        elif self.operation == "policy.workflow":
            if self.evaluation.suite == "libero_spatial" and self.evaluation.parity_limits is None:
                raise ValueError("Spatial workflows require an explicit parity tolerance profile")
            if sum((bool(self.source_id), bool(self.artifact_id), self.training is not None)) != 1:
                raise ValueError("Choose one workflow input: source, artifact, or training recipe")
            if self.training is not None and not self.dataset_job_id:
                raise ValueError("Training requires a completed dataset intake")
            if len({(x.language, x.vision) for x in self.candidates}) != len(self.candidates):
                raise ValueError("Candidate recipes must be distinct")
        elif not self.artifact_id:
            raise ValueError("This operation requires a project artifact")
        return self


class PolicyArtifact(StrictRecord):
    id: str
    project_id: str
    job_id: str
    label: str
    format: Literal["gguf", "training_checkpoint", "native_checkpoint", "deployment_package"]
    path: str
    manifest_sha256: str
    file_bytes: int = Field(ge=0)
    parent_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LifecycleResult(StrictRecord):
    artifacts: list[PolicyArtifact] = Field(default_factory=list)
    reports: list[dict[str, Any]] = Field(default_factory=list)
    selected_artifact_id: str | None = None
    decision: Literal["completed", "diagnostics_only", "no_feasible_candidate", "validated"] = (
        "completed"
    )


class JobEvent(StrictRecord):
    sequence: int
    stage: str
    message: str
    timestamp: str
