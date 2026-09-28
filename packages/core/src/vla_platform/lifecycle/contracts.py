"""GPU-independent application lifecycle contracts."""

from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CloudExecutionTarget(StrictRecord):
    """Server-owned selection captured when a job is accepted; no credentials."""

    project_id: str = Field(pattern=r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")
    workspace: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    sky_api_endpoint: str = Field(min_length=1, max_length=2048)
    region: str = Field(pattern=r"^[a-z][a-z0-9-]+[0-9]$", max_length=64)
    accelerator: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    gpu_count: Literal[1] = 1
    disk_size_gb: int = Field(ge=100, le=2000, strict=True)
    idle_minutes: int = Field(ge=1, le=60, strict=True)
    instance_type: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]+$")

    @field_validator("sky_api_endpoint")
    @classmethod
    def safe_endpoint(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or any(character.isspace() or not character.isprintable() for character in value)
        ):
            raise ValueError("SkyPilot endpoint must be an HTTP URL without embedded credentials")
        return value.rstrip("/")


class Precision(StrictRecord):
    language: Literal["Q4_0", "Q8_0"] = "Q8_0"
    vision: Literal["Q8_0"] | None = None


class SimulationRequest(StrictRecord):
    """A registered scenario selection; paths and commands belong to the operator."""

    profile_id: str = Field(pattern=r"^[\w-]{1,100}$")
    experimental: bool = Field(default=False, strict=True)


class SimulationTarget(StrictRecord):
    """The accepted multi-worker target, kept separate from single-GPU training."""

    profile_id: str
    profile_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    provider: Literal["gcp"] = "gcp"
    policy_runtime: Literal["lerobot-cuda", "packed-act-cpu"] = "lerobot-cuda"
    accelerators: list[Literal["L4", "H100"]] = Field(default_factory=lambda: ["L4", "H100"])
    source_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_id: str | None = Field(default=None, pattern=r"^sha256:[a-f0-9]{64}$")

    @model_validator(mode="after")
    def matching_policy_resources(self):
        expected = ["L4"] if self.policy_runtime == "packed-act-cpu" else ["L4", "H100"]
        if self.accelerators != expected:
            raise ValueError("Simulation resources differ from its policy runtime")
        return self


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
            if self.mode != "libero":
                raise ValueError("Spatial evaluation requires paired LIBERO episodes (mode=libero)")
            if "steps" not in self.model_fields_set:
                self.steps = 280
            if self.steps != 280:
                raise ValueError("Spatial evaluation requires the full 280-step benchmark horizon")
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


class NativeQuantization(StrictRecord):
    """Explicit local packed ACT recipe; separate from GGUF precision."""

    format: Literal["firebird_quant"]
    bits: Literal[4, 8]
    group_size: Literal[64] = 64

    @field_validator("bits", "group_size", mode="before")
    @classmethod
    def strict_integer(cls, value):
        if type(value) is not int:
            raise ValueError("Native quantization precision and group size must be integers")
        return value


EpisodeId = Annotated[int, Field(ge=0, le=19999, strict=True)]
CoordinateUnit = Annotated[str, Field(min_length=1, max_length=80, pattern=r"^[^\x00-\x1f\x7f]+$")]


class DistillationSplits(StrictRecord):
    train: list[EpisodeId] = Field(min_length=1, max_length=256)
    validation: list[EpisodeId] = Field(min_length=1, max_length=256)
    final: list[EpisodeId] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def distinct_episodes(self):
        episodes = self.train + self.validation + self.final
        if len(episodes) != len(set(episodes)):
            raise ValueError("Distillation episode selections must be unique and disjoint")
        if len(episodes) > 256:
            raise ValueError("Distillation supports at most 256 selected episodes")
        return self


class NativeDistillation(StrictRecord):
    """Recorded-data selection and a fixed ACT teacher/student recipe; no paths."""

    adapter: Literal["act-act-v1"]
    student: Literal["act-256"]
    steps: int = Field(ge=1, le=10000, strict=True)
    learning_rate: float = Field(ge=1e-7, le=1e-3, strict=True, allow_inf_nan=False)
    seed: int = Field(ge=0, le=2147483647, strict=True)
    frame_stride: int = Field(ge=1, le=10000, strict=True)
    splits: DistillationSplits
    coordinate_attestation: Literal["teacher_recorded_coordinates", "generated_fixture"]
    units: list[CoordinateUnit] = Field(min_length=6, max_length=6)


class ReplayObservation(StrictRecord):
    episode_index: EpisodeId
    frame_index: int = Field(ge=0, le=10000000, strict=True)


class NativeReplay(StrictRecord):
    """Explicit CPU observation replay, not an environment or a scored simulation."""

    adapter: Literal["act-packed-observation-v1"]
    selection: list[ReplayObservation] = Field(min_length=1, max_length=32)
    coordinate_attestation: Literal["policy_recorded_coordinates", "generated_fixture"]
    units: list[CoordinateUnit] = Field(min_length=6, max_length=6)

    @model_validator(mode="after")
    def unique_observations(self):
        pairs = [(item.episode_index, item.frame_index) for item in self.selection]
        if len(set(pairs)) != len(pairs):
            raise ValueError("Replay observations must be distinct episode/frame pairs")
        return self


class PolicyRequest(StrictRecord):
    operation: Literal[
        "policy.import",
        "policy.finetune",
        "policy.distill",
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
    simulation: SimulationRequest | None = None
    native_quantization: NativeQuantization | None = None
    native_distillation: NativeDistillation | None = None
    native_replay: NativeReplay | None = None

    @model_validator(mode="before")
    @classmethod
    def native_deadline(cls, value):
        if isinstance(value, dict) and any(
            value.get(key) is not None
            for key in ("native_quantization", "native_distillation", "native_replay")
        ):
            value = dict(value)
            value.setdefault("timeout_seconds", 600)
            if type(value["timeout_seconds"]) is not int:
                raise ValueError("Native worker deadline must be an integer")
        return value

    @model_validator(mode="after")
    def input_contract(self):
        if self.native_replay is not None:
            if (
                self.operation != "policy.run"
                or not self.artifact_id
                or not self.dataset_job_id
                or self.source_id is not None
                or self.resume_job_id is not None
                or self.training is not None
                or self.training_method != "lora"
                or self.precision is not None
                or self.candidates != [Precision()]
                or self.evaluation != Evaluation()
                or self.limits is not None
                or self.simulation is not None
                or self.native_quantization is not None
                or self.native_distillation is not None
                or self.timeout_seconds > 600
            ):
                raise ValueError(
                    "CPU observation replay requires one local packed ACT artifact and an "
                    "explicit immutable dataset selection; simulation and other lifecycle "
                    "options are unsupported"
                )
            return self
        if self.native_distillation is not None:
            if (
                self.operation != "policy.distill"
                or not self.artifact_id
                or not self.dataset_job_id
                or self.source_id is not None
                or self.resume_job_id is not None
                or self.training is not None
                or self.training_method != "lora"
                or self.precision is not None
                or self.candidates != [Precision()]
                or self.evaluation != Evaluation()
                or self.limits is not None
                or self.simulation is not None
                or self.native_quantization is not None
                or self.timeout_seconds > 3600
            ):
                raise ValueError(
                    "Distillation requires a local teacher artifact, completed dataset intake "
                    "and explicit ACT student recipe; other lifecycle options are unsupported"
                )
            return self
        if self.operation == "policy.distill":
            raise ValueError("Distillation requires an explicit native_distillation recipe")
        if self.native_quantization is not None:
            if (
                self.operation != "policy.quantize"
                or not self.artifact_id
                or self.source_id is not None
                or self.dataset_job_id is not None
                or self.resume_job_id is not None
                or self.training is not None
                or self.training_method != "lora"
                or self.precision is not None
                or self.candidates != [Precision()]
                or self.evaluation != Evaluation()
                or self.limits is not None
                or self.simulation is not None
                or self.timeout_seconds > 600
            ):
                raise ValueError(
                    "Native quantization requires one local policy artifact and its explicit "
                    "packed recipe; training, GGUF, evaluation and simulation "
                    "options are unsupported"
                )
            return self
        if self.simulation is not None:
            if self.operation not in {"policy.import", "policy.run"}:
                raise ValueError(
                    "Isaac supports native policy import and experimental Run; "
                    "scored simulation evaluation is not yet available"
                )
            if self.runtime_id != self.simulation.profile_id:
                raise ValueError("Simulation runtime and registered profile must match")
            if (
                self.training is not None
                or self.dataset_job_id
                or self.resume_job_id
                or self.precision is not None
                or self.limits is not None
                or self.evaluation != Evaluation()
                or self.timeout_seconds > 7200
            ):
                raise ValueError("Simulation uses the registered scenario and bounded run contract")
            if self.operation == "policy.run" and (
                not self.simulation.experimental or self.source_id
            ):
                raise ValueError(
                    "Isaac Run requires explicit experimental selection and an artifact"
                )
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
        elif self.operation == "policy.export":
            if (
                not self.artifact_id
                or self.source_id
                or self.training is not None
                or self.dataset_job_id
            ):
                raise ValueError("Export requires only one registered checkpoint artifact")
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
    format: Literal[
        "gguf",
        "training_checkpoint",
        "native_checkpoint",
        "deployment_package",
        "inference_export",
        "simulation_record",
        "native_quantized",
        "native_run_record",
    ]
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
    data: dict[str, Any] = Field(default_factory=dict)


class TrainingMetric(StrictRecord):
    step: int = Field(ge=0)
    timestamp: str | None = None
    train_loss: float | None = None
    validation_loss: float | None = None
    validation_action_mse: float | None = None
    learning_rate: float | None = None
    grad_norm: float | None = None
    elapsed_seconds: float | None = None


class TrainingCheckpoint(StrictRecord):
    step: int = Field(ge=0)
    name: str
    timestamp: str | None = None
    artifact_id: str | None = None
    storage: str | None = None
    remote_uri: str | None = None


class TrainingTelemetry(StrictRecord):
    job_id: str
    status: str
    phase: str
    current_action: str
    updated_at: str
    completed_steps: int | None = None
    total_steps: int | None = None
    percent: float | None = None
    elapsed_seconds: float | None = None
    wall_seconds: float | None = None
    eta_seconds: float | None = None
    latest: TrainingMetric | None = None
    metrics: list[TrainingMetric] = Field(default_factory=list)
    metrics_truncated: bool = False
    checkpoints: list[TrainingCheckpoint] = Field(default_factory=list)
    events: list[JobEvent] = Field(default_factory=list)
    logs: list[str] = Field(default_factory=list)
    reproducibility: dict[str, Any] = Field(default_factory=dict)
