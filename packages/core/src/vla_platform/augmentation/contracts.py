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
            "Change the tabletop texture to matte light oak and the background to plain grey."
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
    "or add text or audio. The requested appearance change follows:\n"
)


class AugmentationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class AugmentationRequest(AugmentationRecord):
    operation: Literal["dataset.augment"] = "dataset.augment"
    source_job_id: str = Field(min_length=1, max_length=200)
    episode_indices: list[Annotated[int, Field(ge=0, strict=True)]] = Field(
        min_length=1, max_length=4
    )
    camera_key: str = Field(min_length=1, max_length=200)
    preset: Literal["lighting", "texture", "custom"] = "lighting"
    prompt: str = Field(default="", max_length=2000)
    start_seconds: float = Field(default=0, ge=0, le=86400)
    duration_seconds: float = Field(default=5, ge=1, le=10)

    @model_validator(mode="after")
    def validate_selection(self):
        if len(set(self.episode_indices)) != len(self.episode_indices):
            raise ValueError("Select each episode only once")
        if self.preset == "custom" and not self.prompt:
            raise ValueError("Custom appearance requires a prompt")
        return self

    def resolved_prompt(self) -> str:
        preset = next(item["prompt"] for item in PRESETS if item["id"] == self.preset)
        return PRESERVATION_PROMPT + "\n".join(part for part in (preset, self.prompt) if part)


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
    repo_id: str
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    metadata_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    clips: list[AugmentationClip] = Field(min_length=1, max_length=4)
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
    max_clips: int = 4
    max_duration_seconds: int = 10
    presets: list[AugmentationPreset] = Field(
        default_factory=lambda: [AugmentationPreset(**preset) for preset in PRESETS]
    )
    setup_message: str | None = None
