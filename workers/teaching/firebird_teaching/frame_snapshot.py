"""Immutable atomic observation envelopes, with no simulator/provider dependency."""

from __future__ import annotations

import base64
import binascii
import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .providers import FrameInput

SCHEMA_VERSION = 1
MAX_PIXELS = 1920 * 1920
MAX_AGE_NS = 30_000_000_000
IDENTITY = re.compile(r"[a-zA-Z0-9_-]{1,96}\Z")
MODES = frozenset({"idle", "running", "paused", "faulted", "closed", "starting"})
CONTEXT_FIELDS = {"session_id", "revision", "active_episode_id", "mode"}


def integer(value: object, maximum: int = 2**63 - 1) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("Invalid bounded integer in frame envelope")
    return value


def identity(value: object) -> str:
    if not isinstance(value, str) or not IDENTITY.fullmatch(value):
        raise ValueError("Invalid frame identity")
    return value


def number(value: object) -> float:
    if type(value) not in (int, float):
        raise ValueError("Invalid frame numeric value")
    try:
        result = float(cast("int | float", value))
    except OverflowError:
        raise ValueError("Frame numeric value exceeds limits") from None
    if not math.isfinite(result):
        raise ValueError("Nonfinite frame numeric value")
    return result


def context(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != CONTEXT_FIELDS:
        raise ValueError("Invalid current frame context")
    identity(value["session_id"])
    integer(value["revision"], 2**53 - 1)
    if value["active_episode_id"] is not None:
        identity(value["active_episode_id"])
    if not isinstance(value["mode"], str) or value["mode"] not in MODES:
        raise ValueError("Invalid frame mode")
    return dict(value)


@dataclass(frozen=True)
class ObservationSnapshot:
    """Pixels, measured joints and acquisition context committed together by the owner."""

    session_id: str
    revision: int
    active_episode_id: str | None
    mode: str
    episode_id: str
    step: int
    sim_time: float
    camera_key: str
    camera_prim: str
    observation_received_monotonic_ns: int
    width: int
    height: int
    rgb: bytes = field(repr=False)
    joints: tuple[str, ...]
    state_rad: tuple[float, ...]

    def __post_init__(self) -> None:
        context(self.context)
        identity(self.episode_id)
        integer(self.step, 2**53 - 1)
        integer(self.observation_received_monotonic_ns)
        if self.mode not in {"idle", "running", "paused"}:
            raise ValueError("Frame was acquired in an unavailable mode")
        if self.active_episode_id is None:
            if self.mode != "idle" or not self.episode_id.startswith("preview-"):
                raise ValueError("Idle preview identity differs from active recording")
        elif self.active_episode_id != self.episode_id or self.mode == "idle":
            raise ValueError("Frame observation differs from its active episode")
        if number(self.sim_time) < 0:
            raise ValueError("Invalid frame simulation time")
        if self.camera_key != "observation.images.front" or not isinstance(self.camera_prim, str):
            raise ValueError("Unsupported frame camera")
        if not re.fullmatch(r"(/[A-Za-z_][A-Za-z_0-9]*)+", self.camera_prim):
            raise ValueError("Invalid camera prim")
        if (
            not 1 <= integer(self.width, 1920)
            or not 1 <= integer(self.height, 1920)
            or self.width * self.height > MAX_PIXELS
            or type(self.rgb) is not bytes
            or len(self.rgb) != self.width * self.height * 3
        ):
            raise ValueError("Invalid frame dimensions or pixel length")
        if (
            not isinstance(self.joints, tuple)
            or not 1 <= len(self.joints) <= 32
            or any(
                not isinstance(name, str) or not name.isidentifier() or len(name) > 96
                for name in self.joints
            )
            or len(set(self.joints)) != len(self.joints)
            or not isinstance(self.state_rad, tuple)
            or len(self.state_rad) != len(self.joints)
        ):
            raise ValueError("Invalid native joint names or state dimensions")
        for value in self.state_rad:
            number(value)

    @property
    def context(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "revision": self.revision,
            "active_episode_id": self.active_episode_id,
            "mode": self.mode,
        }

    def payload(
        self, current_context: dict[str, Any], published_ns: int, now_ns: int
    ) -> dict[str, Any]:
        """Serialize a fresh response without changing the frozen acquisition context."""
        current = context(current_context)
        integer(published_ns)
        integer(now_ns)
        if not self.observation_received_monotonic_ns <= published_ns <= now_ns:
            raise ValueError("Frame host clock moved backwards")
        return {
            "schema_version": SCHEMA_VERSION,
            "available": True,
            **self.context,
            "current_context": current,
            "episode_id": self.episode_id,
            "step": self.step,
            "sim_time": self.sim_time,
            "camera_key": self.camera_key,
            "camera_prim": self.camera_prim,
            "observation_received_monotonic_ns": self.observation_received_monotonic_ns,
            "published_monotonic_ns": published_ns,
            "source_age_ns": now_ns - self.observation_received_monotonic_ns,
            "width": self.width,
            "height": self.height,
            "rgb_base64": base64.b64encode(self.rgb).decode("ascii"),
            "rgb_sha256": hashlib.sha256(self.rgb).hexdigest(),
            "joints": list(self.joints),
            "state_rad": list(self.state_rad),
            "units": "rad",
        }


@dataclass(frozen=True)
class AdmittedFrame:
    """A bounded received frame with conservative age, not permission to actuate."""

    observation: ObservationSnapshot
    received_monotonic_ns: int
    source_age_at_receipt_ns: int

    def provider_input(self) -> FrameInput:
        """Bridge to optional inference without importing its HTTP dependency in Isaac."""
        from .providers import FrameInput, SourceContext

        frame = self.observation
        return FrameInput(
            SourceContext(frame.session_id, frame.active_episode_id, frame.revision),
            frame.step,
            frame.width,
            frame.height,
            frame.rgb,
            self.received_monotonic_ns,
            self.source_age_at_receipt_ns,
        )


def admit_frame(
    value: dict[str, Any],
    *,
    expected_session_id: str,
    request_started_monotonic_ns: int,
    received_monotonic_ns: int,
    max_age_ns: int = 5_000_000_000,
) -> AdmittedFrame:
    """Validate one /frame response, never joined with a separately fetched /state.

    Both local times come from the same consuming process. Source age is a
    duration measured by the executor. Adding full request elapsed time is a
    conservative upper bound on age at receipt; remote clocks are not compared.
    """
    required = {
        "schema_version",
        "available",
        *CONTEXT_FIELDS,
        "current_context",
        "episode_id",
        "step",
        "sim_time",
        "camera_key",
        "camera_prim",
        "observation_received_monotonic_ns",
        "published_monotonic_ns",
        "source_age_ns",
        "width",
        "height",
        "rgb_base64",
        "rgb_sha256",
        "joints",
        "state_rad",
        "units",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Frame envelope fields are missing, unsupported or unavailable")
    if type(value["schema_version"]) is not int or value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported frame envelope schema")
    if value["available"] is not True or value["units"] != "rad":
        raise ValueError("Frame is unavailable or uses unsupported units")
    expected = identity(expected_session_id)
    frozen = context({key: value[key] for key in CONTEXT_FIELDS})
    current = context(value["current_context"])
    if frozen["session_id"] != expected or current != frozen:
        raise ValueError("Frame session or command context changed")
    if current["mode"] not in {"idle", "running", "paused"}:
        raise ValueError("Frame executor is unavailable")
    started = integer(request_started_monotonic_ns)
    received = integer(received_monotonic_ns)
    maximum = integer(max_age_ns, MAX_AGE_NS)
    if not maximum or received < started:
        raise ValueError("Invalid local frame request timing")
    acquired = integer(value["observation_received_monotonic_ns"])
    published = integer(value["published_monotonic_ns"])
    source_age = integer(value["source_age_ns"])
    age = source_age + received - started
    if acquired > published or source_age < published - acquired or age > maximum:
        raise ValueError("Frame is stale or has inconsistent source timing")
    width, height = integer(value["width"], 1920), integer(value["height"], 1920)
    if not width or not height or width * height > MAX_PIXELS:
        raise ValueError("Frame dimensions exceed limits")
    encoded = value["rgb_base64"]
    expected_bytes = width * height * 3
    if not isinstance(encoded, str) or len(encoded) != 4 * ((expected_bytes + 2) // 3):
        raise ValueError("Frame pixel encoding length differs from dimensions")
    try:
        rgb = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError("Invalid frame pixel encoding") from None
    if len(rgb) != expected_bytes or hashlib.sha256(rgb).hexdigest() != value["rgb_sha256"]:
        raise ValueError("Frame pixels do not match their fingerprint")
    if not isinstance(value["joints"], list) or not isinstance(value["state_rad"], list):
        raise ValueError("Frame native joints and state must be lists")
    observation = ObservationSnapshot(
        **frozen,
        episode_id=value["episode_id"],
        step=value["step"],
        sim_time=value["sim_time"],
        camera_key=value["camera_key"],
        camera_prim=value["camera_prim"],
        observation_received_monotonic_ns=acquired,
        width=width,
        height=height,
        rgb=rgb,
        joints=tuple(value["joints"]),
        state_rad=tuple(value["state_rad"]),
    )
    return AdmittedFrame(observation, received, age)
