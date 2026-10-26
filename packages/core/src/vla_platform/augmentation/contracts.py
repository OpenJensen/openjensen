from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MODEL = "gemini-omni-1.1-flash"
PRESETS = [
    {
        "id": "lighting",
        "label": "Change lighting",
        "prompt": "Change the lighting to warm late-afternoon light with soft shadows.",
    },
    {
        "id": "texture",
        "label": "Change textures",
        "prompt": (
            "Change only the tabletop surface to matte light oak. "
            "Keep the background and all other surfaces unchanged."
        ),
    },
    {"id": "custom", "label": "Custom appearance", "prompt": ""},
]
PRESERVATION_PROMPT = (
    "Edit this robotics dataset video by changing visual appearance only. "
    "Preserve the exact robot and object geometry, identities, positions, camera viewpoint, "
    "motion, contact events, action sequence, timing, duration and frame order. "
    "Keep the same scene throughout the clip with temporally consistent appearance. "
    "Do not add or remove objects, introduce camera motion, retime motion, crop the scene, "
    "or add text or audio. Change only what the edit explicitly requests. "
    "Keep all unmentioned surfaces, materials, colours, lighting and backgrounds unchanged. "
    "The requested appearance change follows:\n"
)


class AugmentationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class AugmentationSourceClip(AugmentationRecord):
    job_id: str = Field(min_length=1, max_length=200)
    clip_index: int = Field(ge=0, strict=True)
    variant: Literal["original", "augmented"]


class AugmentationDatasetScope(AugmentationRecord):
    recipe_job_id: str = Field(min_length=1, max_length=200)
    camera_keys: list[str] = Field(min_length=1, max_length=16)
    plan_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def unique_cameras(self):
        if len(set(self.camera_keys)) != len(self.camera_keys):
            raise ValueError("Choose each camera only once")
        return self


class AugmentationPlan(AugmentationRecord):
    total_episodes: int
    camera_keys: list[str]
    total_source_seconds: float
    total_clips: int
    provider_requests: int
    recipe_prompts: list[str]
    plan_sha256: str
    model: str
    warnings: list[str]


class AugmentationRequest(AugmentationRecord):
    operation: Literal["dataset.augment"] = "dataset.augment"
    source_job_id: str = Field(min_length=1, max_length=200)
    # Keep legacy normalized requests and durable fingerprints byte-for-byte compatible.
    source_clip: AugmentationSourceClip | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    dataset_scope: AugmentationDatasetScope | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    episode_indices: list[Annotated[int, Field(ge=0, strict=True)]] = Field(
        min_length=0, max_length=4
    )
    camera_key: str = Field(min_length=1, max_length=200)
    preset: Literal["lighting", "texture", "custom"] = "lighting"
    prompt: str = Field(default="", max_length=2000)
    start_seconds: float = Field(default=0, ge=0, le=86400)
    duration_seconds: float = Field(default=5, ge=1, le=10)

    @model_validator(mode="after")
    def validate_selection(self):
        if self.dataset_scope:
            if self.source_clip or self.episode_indices or self.start_seconds != 0:
                raise ValueError(
                    "A full-dataset run covers full episodes from their original videos"
                )
            if self.camera_key != self.dataset_scope.camera_keys[0]:
                raise ValueError("The primary camera must match the selected cameras")
        elif not self.episode_indices:
            raise ValueError("Choose at least one preview episode")
        if len(set(self.episode_indices)) != len(self.episode_indices):
            raise ValueError("Select each episode only once")
        if self.source_clip is not None and len(self.episode_indices) != 1:
            raise ValueError("Choose one saved clip per variation or refinement")
        if self.preset == "custom" and not self.prompt:
            raise ValueError("Custom appearance requires a prompt")
        return self

    def resolved_prompt(self) -> str:
        preset = next(item["prompt"] for item in PRESETS if item["id"] == self.preset)
        return PRESERVATION_PROMPT + (self.prompt or preset)


class AugmentationClip(AugmentationRecord):
    index: int = Field(ge=0)
    episode_index: int = Field(ge=0)
    camera_key: str
    source_start_seconds: float = Field(ge=0)
    source_end_seconds: float = Field(gt=0)
    input_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    output_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    interaction_id: str | None = None


class AugmentationResult(AugmentationRecord):
    operation: Literal["dataset.augment"] = "dataset.augment"
    model: str = MODEL
    auth_mode: Literal["google_cloud", "gemini_api_key"] = "gemini_api_key"
    google_cloud_project: str | None = None
    prompt: str
    source_job_id: str
    source_clip: AugmentationSourceClip | None = None
    repo_id: str
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    metadata_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    dataset_scope: AugmentationDatasetScope | None = None
    recipe_prompts: list[str] | None = None
    clips: list[AugmentationClip] = Field(min_length=1, max_length=1000)
    review_required: Literal[True] = True
    warnings: list[str]


class AugmentationPreset(AugmentationRecord):
    id: Literal["lighting", "texture", "custom"]
    label: str
    prompt: str


class AugmentationOptions(AugmentationRecord):
    configured: bool
    model: str = MODEL
    auth_mode: Literal["google_cloud", "gemini_api_key", "unconfigured"] = "unconfigured"
    google_cloud_project: str | None = None
    auth_message: str | None = None
    preservation_prompt: str = PRESERVATION_PROMPT
    max_clips: int = 4
    max_duration_seconds: int = 10
    presets: list[AugmentationPreset] = Field(
        default_factory=lambda: [AugmentationPreset(**preset) for preset in PRESETS]
    )
    setup_message: str | None = None
