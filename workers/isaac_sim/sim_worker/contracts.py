from dataclasses import dataclass
from enum import StrEnum


API_VERSION = "simulation.worker/v1alpha1"
BUILTIN_SCENE = "builtin:falling-cube"
RGB_CHANNELS = 3


class Status(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class RunSpec:
    """Values crossing adapters; no simulator or cloud SDK objects."""

    scene: str
    camera: str
    width: int
    height: int
    fps: int
    frames: int
    output_uri: str

    @property
    def frame_bytes(self) -> int:
        return self.width * self.height * RGB_CHANNELS
