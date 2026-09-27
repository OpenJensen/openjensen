"""Additive, path-free contracts for local Isaac teaching captures."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class TeachingCaptureRequest(StrictRecord):
    operation: Literal["teaching.capture"] = "teaching.capture"
    profile_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    profile_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    timeout_seconds: int = Field(default=300, ge=1, le=3600)


class CaptureEpisode(StrictRecord):
    episode_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    frames: int = Field(ge=1, le=3600)
    termination: Literal["finish", "reset", "step_limit", "shutdown"]
    outcome: Literal["unknown", "operator_reported_failure"]


class TeachingCaptureResult(StrictRecord):
    schema_version: Literal[1] = 1
    operation: Literal["teaching.capture"] = "teaching.capture"
    profile_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    profile_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    session_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    session_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    recording_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    episodes: list[CaptureEpisode] = Field(min_length=1, max_length=100)
    origin: Literal["recorded", "synthetic"]
    lineage_group: str = Field(pattern=r"^[a-zA-Z0-9_.:-]{1,128}$")
    published: Literal[True] = True
    content_verified: Literal[True] = True
    process_cleanup_verified: Literal[True] = True
    simulator_coordinates: Literal[True] = True
    physical_calibration_verified: Literal[False] = False
    task_success_claimed: Literal[False] = False

    @model_validator(mode="after")
    def unique_episodes(self):
        if len({item.episode_id for item in self.episodes}) != len(self.episodes):
            raise ValueError("Published episode identities must be unique")
        return self
