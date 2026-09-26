from dataclasses import dataclass
from typing import Protocol

API_VERSION = "simulation.rollout/v1alpha1"
CALIBRATION_VERSION = "simulation.joints/v1alpha1"
RGB_CHANNELS = 3


@dataclass(frozen=True)
class SimSpec:
    scene: str
    camera: str
    articulation: str
    joints: tuple[str, ...]
    width: int
    height: int
    fps: int
    physics_hz: int


@dataclass(frozen=True)
class Frame:
    width: int
    height: int
    rgb: bytes


@dataclass(frozen=True)
class Observation:
    episode_id: str
    step: int
    sim_time: float
    state: tuple[float, ...]
    frame: Frame


@dataclass(frozen=True)
class ActionChunk:
    episode_id: str
    step: int
    model_id: str
    actions: tuple[tuple[float, ...], ...]


class Simulation(Protocol):
    def reset(self, episode_id: str) -> None: ...
    def observe(self) -> Observation: ...
    def apply(self, targets: tuple[float, ...]) -> None: ...
    def step(self) -> None: ...


class Policy(Protocol):
    def reset(self, episode_id: str) -> None: ...
    def predict(self, observation: Observation) -> ActionChunk: ...
