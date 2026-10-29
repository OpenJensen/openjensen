"""Agent output is data; only the trusted builder authors simulator assets."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


Coordinate = Annotated[float, Field(ge=-10, le=10)]
Point = Annotated[list[Coordinate], Field(min_length=3, max_length=3)]
Dimension = Annotated[float, Field(gt=0, le=3)]
Size = Annotated[list[Dimension], Field(min_length=3, max_length=3)]
Note = Annotated[str, Field(min_length=1, max_length=2000)]


Effort = Literal["low", "medium", "high", "xhigh", "max", "ultra"]
Model = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")]


class AgentSettings(Strict):
    harness: Literal["codex_cli"] = "codex_cli"
    model: Model
    reasoning_effort: Effort = "medium"


class SceneRequest(Strict):
    operation: Literal["scene.generate"] = "scene.generate"
    scope: Literal["dataset"] = "dataset"
    request_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    source_job_id: str = Field(min_length=1, max_length=64)
    source_scene_id: str | None = Field(
        default=None, pattern=r"^[a-f0-9]{32}$", exclude_if=lambda value: value is None
    )
    episode_index: int | None = Field(default=None, ge=0, le=99999, strict=True)
    camera_key: str = Field(default="observation.images.front", min_length=1, max_length=128)
    max_attempts: int = Field(default=2, ge=1, le=3, strict=True)
    model: Model | None = None
    reasoning_effort: Effort = "medium"


class Camera(Strict):
    position: Point
    target: Point
    focal: float = Field(ge=5, le=200)
    roll_degrees: float = Field(ge=-45, le=45)

    @model_validator(mode="after")
    def distinct(self):
        if sum((a - b) ** 2 for a, b in zip(self.position, self.target)) < 0.0001:
            raise ValueError("Camera position and target must differ")
        return self


class Robot(Strict):
    position: Point
    yaw_degrees: float = Field(ge=-180, le=180)


class Cup(Strict):
    position: Point
    height: float = Field(ge=0.02, le=0.5)
    radius_bottom: float = Field(ge=0.005, le=0.2)
    radius_top: float = Field(ge=0.005, le=0.2)
    thickness: float = Field(ge=0.0002, le=0.01)
    mass: float = Field(ge=0.001, le=1)

    @model_validator(mode="after")
    def wall(self):
        if self.thickness >= min(self.radius_bottom, self.radius_top) / 2:
            raise ValueError("Cup wall is too thick for the opening")
        return self


class Box(Strict):
    position: Point
    size: Size
    thickness: float = Field(ge=0.001, le=0.02)

    @model_validator(mode="after")
    def walls(self):
        if self.thickness * 2 >= min(self.size):
            raise ValueError("Box walls leave no interior")
        return self


class Proposal(Strict):
    matches_template: bool
    camera: Camera
    robot: Robot
    cup: Cup
    box: Box
    observations: list[Note] = Field(min_length=1, max_length=20)
    uncertainties: list[Note] = Field(min_length=1, max_length=20)


class Review(Strict):
    supported: bool
    consistent_with_images: bool
    issues: list[Note] = Field(max_length=20)
    uncertainties: list[Note] = Field(min_length=1, max_length=20)


class SceneMedia(Strict):
    role: Literal["environment_image", "episode_replay", "source_image", "source_video"]
    path: str = Field(min_length=1, max_length=256)
    label: str = Field(min_length=1, max_length=128)
    episode_index: int | None = None


class SceneJobResult(Strict):
    kind: Literal["scene_generation"] = "scene_generation"
    scene_id: str
    validation_status: str
    scene_artifacts: list[str] = Field(default_factory=list)


class ClipRequest(Strict):
    episode_index: int = Field(ge=0, le=99999, strict=True)
    camera_key: str = Field(min_length=1, max_length=128)
    start_fraction: float = Field(ge=0, lt=1)
    end_fraction: float = Field(gt=0, le=1)
    reason: Note

    @model_validator(mode="after")
    def interval(self):
        if self.end_fraction <= self.start_fraction:
            raise ValueError("Clip end must follow its start")
        return self


class Investigation(Strict):
    ready: bool
    clips: list[ClipRequest] = Field(max_length=4)
    replay_episode: int = Field(ge=0, le=99999, strict=True)
    observations: list[Note] = Field(max_length=20)
    uncertainties: list[Note] = Field(min_length=1, max_length=20)


class SceneRun(Strict):
    id: str
    job_id: str | None = None
    media: list[SceneMedia] = Field(default_factory=list)
    project_id: str
    request: SceneRequest
    status: Literal[
        "queued",
        "running",
        "needs_simulation",
        "needs_review",
        "failed",
        "cancelled",
        "interrupted",
    ]
    stage: str
    created_at: str
    updated_at: str
    attempt: int = 0
    message: str
    events: list[dict] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    source: dict = Field(default_factory=dict)
    agent: AgentSettings | None = None
    gates: dict = Field(default_factory=dict)
