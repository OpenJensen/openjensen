"""Path-free recording selections; raw capture coordinates are never client overrides."""

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

Identity = Annotated[
    str, StringConstraints(pattern=r"^[a-f0-9]{32}$", min_length=32, max_length=32)
]
Digest = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$", min_length=64, max_length=64)]


class RecordingRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class RecordingEpisode(RecordingRecord):
    episode_id: Identity
    receipt_sha256: Digest


class RecordingCapture(RecordingRecord):
    session_id: Identity
    session_sha256: Digest
    episodes: list[RecordingEpisode] = Field(min_length=1, max_length=100)


class RecordingPreparation(RecordingRecord):
    schema_version: Literal[1] = 1
    configuration_sha256: Digest
    timeout_seconds: int = Field(default=600, ge=60, le=1800)
    captures: list[RecordingCapture] = Field(min_length=1, max_length=100)

    @field_validator("schema_version", mode="before")
    @classmethod
    def exact_schema(cls, value):
        if type(value) is not int or value != 1:
            raise ValueError("Recording preparation schema must be integer 1")
        return value

    @model_validator(mode="after")
    def unique(self):
        sessions = [c.session_id for c in self.captures]
        episodes = [e.episode_id for c in self.captures for e in c.episodes]
        if len(set(sessions)) != len(sessions) or len(set(episodes)) != len(episodes):
            raise ValueError("Select each capture and episode only once")
        if len(episodes) > 100:
            raise ValueError("Select at most 100 episodes in total")
        return self


class RecordingSummary(RecordingRecord):
    job_id: str = Field(min_length=1, max_length=100)
    selection_sha256: Digest
    source_count: int = Field(ge=1, le=100)
    lineage_group_count: int = Field(ge=1, le=100)
    writer_readback_verified: Literal[True] = True
    source_preserved: Literal[True] = True
    task_success_verified: Literal[False] = False

    @field_validator(
        "writer_readback_verified", "source_preserved", "task_success_verified", mode="before"
    )
    @classmethod
    def exact_bool(cls, value):
        if type(value) is not bool:
            raise ValueError("Verification claims must be exact booleans")
        return value


class RecordingOptions(RecordingRecord):
    configured: bool
    runtime_verified: Literal[False] = False
    configuration_sha256: Digest | None
    max_episodes: int = 100
    max_source_bytes: int = 8 * 1024**3
    setup_message: str


class RecordingEpisodeOption(RecordingEpisode):
    frames: int = Field(ge=1, le=3600)
    outcome: Literal["unknown", "operator_reported_failure"]
    termination: Literal["finish", "reset", "step_limit", "shutdown"]


class RecordingCaptureOption(RecordingRecord):
    session_id: Identity
    session_sha256: Digest
    origin: Literal["recorded", "synthetic"]
    lineage_group: str
    controller: Literal["joint_position_targets"]
    state_units: Literal["radians"]
    action_units: Literal["radians"]
    timebase: Literal["simulation_seconds"]
    camera_key: Literal["observation.images.front"]
    joint_names: list[str]
    width: int
    height: int
    fps: int
    physics_hz: int
    scene_sha256: Digest
    camera_prim: str
    episodes: list[RecordingEpisodeOption]
    content_verified: Literal[False] = False


class RecordingCatalog(RecordingRecord):
    configuration_sha256: Digest
    captures: list[RecordingCaptureOption]
    message: str
