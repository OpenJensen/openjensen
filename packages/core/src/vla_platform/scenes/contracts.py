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


class SceneRequest(Strict):
    request_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    source_job_id: str = Field(min_length=1, max_length=64)
    episode_index: int = Field(default=3, ge=0, le=99999, strict=True)
    camera_key: str = Field(default="observation.images.front", min_length=1, max_length=128)
    max_attempts: int = Field(default=2, ge=1, le=3, strict=True)


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


class SceneRun(Strict):
    id: str
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
    gates: dict = Field(default_factory=dict)
