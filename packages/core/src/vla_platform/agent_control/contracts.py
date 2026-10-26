"""Strict public agent contracts. Credentials and provider continuations stay private."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

IDENTITY = r"^[A-Za-z0-9_-]{1,96}$"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ProviderInput(Strict):
    provider: Literal["mock", "openai_responses"]
    model: str = Field(
        default="gpt-6-astra", min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$"
    )
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    api_key: SecretStr | None = None

    @model_validator(mode="after")
    def key(self):
        if self.api_key is not None:
            key = self.api_key.get_secret_value()
            if not 16 <= len(key) <= 4096 or any(not 33 <= ord(c) <= 126 for c in key):
                raise ValueError("Invalid credential")
        return self


class ProviderStatus(Strict):
    id: str
    provider: Literal["mock", "openai_responses"]
    model: str
    reasoning_effort: str
    revision: str
    saved: bool
    checked: bool
    capabilities: list[str]
    message: str


class PhysicalArming(Strict):
    confirm: Literal["ARM AI CONTROL"]
    expected_generation: int = Field(ge=0)
    camera_index: int = Field(ge=0, le=31)
    operator_present: Literal[True]
    workspace_clear: Literal[True]

    @model_validator(mode="before")
    @classmethod
    def explicit_confirmation(cls, value):
        if (
            not isinstance(value, dict)
            or value.get("operator_present") is not True
            or value.get("workspace_clear") is not True
        ):
            raise ValueError("Explicit operator and workspace confirmation required")
        return value


class SessionCreate(Strict):
    provider_id: str = Field(default="mock", pattern=IDENTITY)
    task: str = Field(min_length=1, max_length=2048)
    target_kind: Literal["replay", "teaching", "robot_lab"] = "replay"
    teaching_job_id: str | None = Field(default=None, pattern=IDENTITY)
    replay_observation_id: str | None = Field(default=None, pattern=IDENTITY)
    mode: Literal["advisory", "shadow", "supervised", "bounded"] = "supervised"
    clock_policy: Literal["replay", "stepped", "continuous", "physical"] = "replay"
    max_calls: int = Field(default=4, ge=1, le=100)
    max_seconds: int = Field(default=120, ge=1, le=600)
    max_output_tokens: int = Field(default=1024, ge=128, le=4096)
    max_total_tokens: int = Field(default=16000, ge=1024, le=1000000)
    cloud_consent: bool = False
    planning_mode: Literal["joint", "task"] = "task"
    max_steps: int = Field(default=12, ge=1, le=60)
    physical: PhysicalArming | None = None

    @model_validator(mode="after")
    def target(self):
        if self.target_kind == "teaching" and (
            not self.teaching_job_id or self.clock_policy == "replay"
        ):
            raise ValueError("Teaching needs a managed session and simulation clock")
        if self.target_kind == "replay" and (self.clock_policy != "replay" or self.teaching_job_id):
            raise ValueError("Replay requires replay clock")
        if self.target_kind == "robot_lab" and (
            self.physical is None
            or self.clock_policy != "physical"
            or self.teaching_job_id
            or self.replay_observation_id
            or self.mode != "supervised"
            or self.planning_mode != "task"
        ):
            raise ValueError(
                "Physical control requires explicit arming, physical clock and supervised execution"
            )
        if self.target_kind != "robot_lab" and (
            self.physical is not None or self.clock_policy == "physical"
        ):
            raise ValueError("Physical authority belongs only to a physical target")
        if not self.task.strip():
            raise ValueError("Task is empty")
        return self


class Observation(Strict):
    id: str
    sha256: str
    target_session_id: str
    episode_id: str | None
    authority_epoch: int
    revision: int
    step: int
    captured_at: str
    expires_at: str
    clock_policy: str
    environment_kind: Literal["replay", "simulation", "physical"]
    embodiment_manifest_id: str
    scene_id: str
    joints: dict[str, float]
    units: Literal["radians", "mixed"] = "radians"
    joint_units: dict[str, str] = Field(default_factory=dict)
    calibration_sha256: str | None = None
    task_evidence: dict | None = None
    image_sha256: str | None = None
    source: str


class Arguments(Strict):
    joint: str = Field(min_length=1, max_length=128)
    delta_rad: float = Field(ge=-0.15, le=0.15)


class SegmentArguments(Strict):
    targets: dict[str, float] = Field(min_length=1, max_length=6)
    duration_seconds: float = Field(ge=0.2, le=10)


class Assessment(Strict):
    outcome: Literal["achieved", "needs_help", "cannot_proceed"]
    summary: str = Field(min_length=1, max_length=2048)


class TaskReview(Strict):
    authority_epoch: int = Field(ge=1)
    observation_id: str = Field(pattern=IDENTITY)
    outcome: Literal["success", "failure", "unknown"]
    note: str = Field(default="", max_length=1024)


class TaskReviewRecord(TaskReview):
    source: Literal["operator"] = "operator"
    reviewed_at: str


class Proposal(Strict):
    id: str
    request_id: str
    agent_session_id: str
    observation_id: str
    observation_sha256: str
    authority_epoch: int
    expected_revision: int
    provider_revision: str
    model: str
    skill: Literal["joint_correction", "joint_segment"] = "joint_correction"
    purpose: str = ""
    arguments: Arguments | SegmentArguments
    status: Literal[
        "proposed", "rejected", "accepted", "started", "completed", "stopped", "unknown"
    ]
    created_at: str
    expires_at: str


class ExecutionEvidence(Strict):
    joint_names: list[str]
    requested_targets: dict[str, float]
    applied_targets: dict[str, float]
    observed_next_state: dict[str, float]
    joint_limit_clipped: list[bool]
    speed_limit_clipped: list[bool]
    simulated_seconds: float | None = None
    execution_elapsed_seconds: float | None = None


class Receipt(Strict):
    id: str
    proposal_id: str
    status: Literal["rejected", "completed", "stopped", "unknown", "replayed"]
    arguments: Arguments | SegmentArguments
    applied_targets: dict[str, float]
    pre_observation_id: str
    post_observation_id: str | None
    elapsed_seconds: float
    reason: str
    task_success_verified: Literal[False] = False
    executor_evidence: ExecutionEvidence | None = None
    original_model_observation_id: str | None = None
    admission_observation_id: str | None = None
    joint_units: dict[str, str] = Field(default_factory=dict)
    execution_provenance: dict | None = None


class Session(Strict):
    id: str
    project_id: str
    provider_id: str
    provider_revision: str
    task: str
    target_kind: Literal["replay", "teaching", "robot_lab"]
    teaching_job_id: str | None
    mode: Literal["advisory", "shadow", "supervised", "bounded"]
    clock_policy: str
    status: Literal[
        "observing",
        "ready",
        "planning",
        "waiting_for_acceptance",
        "executing",
        "paused",
        "stopping",
        "interrupted",
        "complete",
        "stopped",
        "error",
    ]
    authority_epoch: int
    created_at: str
    updated_at: str
    observation: Observation | None = None
    pending_proposal: Proposal | None = None
    receipts: list[Receipt] = Field(default_factory=list)
    calls_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reserved_tokens: int = 0
    max_calls: int
    max_seconds: int
    max_output_tokens: int
    max_total_tokens: int
    message: str = ""
    task_success_verified: bool = False
    planning_mode: Literal["joint", "task"] = "task"
    max_steps: int = 12
    step_count: int = 0
    run_enabled: bool = False
    control_armed: bool = False
    supported_skills: list[str] = Field(default_factory=list)
    model_assessment: Assessment | None = None
    task_review: TaskReviewRecord | None = None
    execution_observation: Observation | None = None


class PlanRequest(Strict):
    request_id: str = Field(pattern=IDENTITY)


class Approval(Strict):
    proposal_id: str = Field(pattern=IDENTITY)
    authority_epoch: int = Field(ge=1)
    execution_observation_id: str | None = Field(default=None, pattern=IDENTITY)
    execution_observation_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    scene_unchanged_confirmed: bool = False


class Heartbeat(Strict):
    authority_epoch: int = Field(ge=1)
    operator_present: Literal[True]

    @model_validator(mode="before")
    @classmethod
    def present(cls, value):
        if not isinstance(value, dict) or value.get("operator_present") is not True:
            raise ValueError("Operator presence must be explicit")
        return value


class Event(Strict):
    id: int
    kind: str
    created_at: str
    record: dict
