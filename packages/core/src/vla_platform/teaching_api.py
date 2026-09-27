"""Typed local teaching relay. Provider secrets and executor URLs stay server-owned."""

import asyncio
import base64
import binascii
import hashlib
import json
import math
import os
import re
import stat
import time
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from vla_platform.lifecycle.contracts import StrictRecord

router = APIRouter(prefix="/api/v1/teaching", tags=["Teaching"])
IDENTITY = r"^[A-Za-z0-9_-]{1,96}$"
RELAY_DEADLINE = 4
VOICE_DEADLINE = 12


class TeachingCommand(StrictRecord):
    session_id: str = Field(pattern=IDENTITY)
    command_id: str = Field(pattern=IDENTITY)
    episode_id: str | None = Field(default=None, pattern=IDENTITY)
    expected_revision: int = Field(ge=0, le=2**53 - 1, strict=True)
    operation: Literal["task", "start", "pause", "reset", "correct", "mark_failure", "finish"]
    arguments: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def bounded_arguments(self):
        expected = {
            "task": {"instruction"},
            "correct": {"joint", "delta_rad"},
            "mark_failure": {"reason"},
        }.get(self.operation, set())
        if set(self.arguments) != expected:
            raise ValueError("Unexpected or missing teaching arguments")
        for name in expected - {"delta_rad"}:
            text = self.arguments[name]
            if (
                not isinstance(text, str)
                or not text.strip()
                or len(text) > 256
                or any(ord(c) < 32 for c in text)
            ):
                raise ValueError("Teaching text must be printable and at most 256 characters")
        if "delta_rad" in expected:
            delta = self.arguments["delta_rad"]
            if type(delta) not in (float, int) or not 0 < abs(delta) <= 0.15:
                raise ValueError("Correction must be nonzero and at most 0.15 radians")
        return self


class VoiceJoin(StrictRecord):
    session_id: str = Field(pattern=IDENTITY)


def configuration(voice=False):
    raw = os.getenv("FIREBIRD_TEACHING_VOICE_URL" if voice else "FIREBIRD_TEACHING_URL")
    token_path = os.getenv("FIREBIRD_TEACHING_CONTROL_TOKEN_FILE")
    if not raw or not token_path:
        raise HTTPException(
            503, "Connect a teaching executor in the application host configuration."
        )
    try:
        url = urlsplit(raw)
        if (
            url.scheme != "http"
            or url.hostname != "127.0.0.1"
            or not url.port
            or url.path not in {"", "/"}
            or url.query
            or url.fragment
            or url.username
            or url.password
        ):
            raise ValueError("Invalid teaching origin")
        path = Path(token_path)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        if path.is_symlink():
            raise ValueError("Token must not be a symlink")
        with os.fdopen(os.open(path, flags), "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > 4096:
                raise ValueError("Invalid token file")
            if os.name == "posix" and info.st_mode & 0o077:
                raise ValueError("Token file must be private")
            raw_token = handle.read(4097)
        if len(raw_token) > 4096:
            raise ValueError("Oversized token file")
        token = raw_token.decode("ascii").strip()
        if len(token) < 32 or len(token) > 4096 or not re.fullmatch(r"[A-Za-z0-9_.-]+", token):
            raise ValueError("Invalid token")
    except OSError, ValueError, UnicodeError:
        raise HTTPException(503, "Teaching connection configuration is invalid.") from None
    return raw.rstrip("/"), token


def finite_float(raw):
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError("Nonfinite response")
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate response field")
        result[key] = value
    return result


async def relay(path, payload=None, *, voice=False, maximum=65536):
    origin, token = configuration(voice)
    try:
        async with (
            asyncio.timeout(VOICE_DEADLINE if voice else RELAY_DEADLINE),
            httpx.AsyncClient(
                timeout=12 if voice else 4, trust_env=False, follow_redirects=False
            ) as client,
        ):
            async with client.stream(
                "POST" if payload is not None else "GET",
                origin + path,
                json=payload,
                headers={"Authorization": "Bearer " + token},
            ) as upstream:
                if not 200 <= upstream.status_code < 300:
                    raise HTTPException(
                        503 if upstream.status_code >= 500 else 409,
                        "Teaching request was not accepted. Refresh the executor state and retry.",
                    )
                raw = bytearray()
                async for chunk in upstream.aiter_bytes():
                    raw.extend(chunk)
                    if len(raw) > maximum:
                        raise ValueError("Oversized teaching response")
                value = json.loads(
                    raw.decode("utf-8"),
                    object_pairs_hook=unique_object,
                    parse_float=finite_float,
                    parse_constant=lambda _: (_ for _ in ()).throw(
                        ValueError("Nonfinite response")
                    ),
                )
                if not isinstance(value, dict):
                    raise ValueError("Invalid teaching response")
                return value
    except httpx.HTTPError, ValueError, TimeoutError:
        raise HTTPException(
            503, "Teaching executor is unavailable or returned an invalid response."
        ) from None


class PublicState(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, allow_inf_nan=False)
    mode: Literal["starting", "idle", "running", "paused", "faulted", "closed"]
    episode_id: str | None = Field(default=None, pattern=IDENTITY)
    revision: int = Field(ge=0)
    instruction: str = Field(default="", max_length=256)
    outcome: str = Field(default="unknown", max_length=256)
    steps: int = Field(default=0, ge=0)
    sim_time: float = Field(default=0, ge=0)
    state_rad: list[float] | None = Field(default=None, max_length=64)
    joints: list[str] = Field(default_factory=list, max_length=64)
    fault: str | None = None
    session_id: str | None = Field(default=None, pattern=IDENTITY)
    last_applied_command_id: str | None = Field(default=None, pattern=IDENTITY)
    last_applied_step: int | None = Field(default=None, ge=0)


def public_state(value):
    # Never forward arbitrary worker fields, local recording paths or credentials.
    try:
        result = PublicState.model_validate(value).model_dump()
    except ValidationError:
        raise HTTPException(503, "Teaching state is invalid.") from None
    if result["fault"]:
        result["fault"] = "Executor stopped after a fault; inspect its local log."
    return result


@router.get("/state")
async def state(response: Response):
    response.headers["Cache-Control"] = "no-store"
    try:
        value = public_state(await relay("/state"))
    except HTTPException as exc:
        return {"connected": False, "message": exc.detail, "state": None, "voice_configured": False}
    return {
        "connected": True,
        "message": None,
        "state": value,
        "voice_configured": bool(os.getenv("FIREBIRD_TEACHING_VOICE_URL")),
    }


FRAME_CONTEXT = {"session_id", "revision", "active_episode_id", "mode"}
FRAME_FIELDS = {
    "schema_version",
    "available",
    *FRAME_CONTEXT,
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
FRAME_MAX_AGE_NS = 5_000_000_000


def frame_integer(value, maximum=2**63 - 1):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("Invalid frame integer")
    return value


def frame_context(value):
    if not isinstance(value, dict) or set(value) != FRAME_CONTEXT:
        raise ValueError("Invalid frame context")
    if not isinstance(value["session_id"], str) or not re.fullmatch(IDENTITY, value["session_id"]):
        raise ValueError("Invalid frame session")
    frame_integer(value["revision"], 2**53 - 1)
    episode = value["active_episode_id"]
    if episode is not None and (
        not isinstance(episode, str) or not re.fullmatch(IDENTITY, episode)
    ):
        raise ValueError("Invalid active episode")
    if value["mode"] not in {"idle", "running", "paused"}:
        raise ValueError("Unavailable frame context")
    return value


def public_frame(value, *, session_id, elapsed_ns):
    """Validate the worker's atomic v1 envelope without importing simulator dependencies."""
    if (
        value == {"schema_version": 1, "available": False}
        and type(value["schema_version"]) is int
        and value["available"] is False
    ):
        return value
    if set(value) != FRAME_FIELDS or type(value["schema_version"]) is not int:
        raise ValueError("Missing or unsupported frame fields")
    if value["schema_version"] != 1 or value["available"] is not True or value["units"] != "rad":
        raise ValueError("Unsupported frame protocol")
    frozen = frame_context({key: value[key] for key in FRAME_CONTEXT})
    if frame_context(value["current_context"]) != frozen:
        raise ValueError("Frame context changed")
    if session_id is not None and frozen["session_id"] != session_id:
        raise ValueError("Frame session changed")
    episode = value["episode_id"]
    if not isinstance(episode, str) or not re.fullmatch(IDENTITY, episode):
        raise ValueError("Invalid observation identity")
    if frozen["active_episode_id"] is None:
        if frozen["mode"] != "idle" or not episode.startswith("preview-"):
            raise ValueError("Invalid preview identity")
    elif episode != frozen["active_episode_id"] or frozen["mode"] == "idle":
        raise ValueError("Observation episode changed")
    frame_integer(value["step"], 2**53 - 1)
    if (
        type(value["sim_time"]) not in (int, float)
        or not math.isfinite(value["sim_time"])
        or value["sim_time"] < 0
    ):
        raise ValueError("Invalid simulation time")
    acquired = frame_integer(value["observation_received_monotonic_ns"])
    published = frame_integer(value["published_monotonic_ns"])
    age = frame_integer(value["source_age_ns"])
    relay_age = age + frame_integer(elapsed_ns)
    if acquired > published or age < published - acquired or relay_age > FRAME_MAX_AGE_NS:
        raise ValueError("Stale or inconsistent observation time")
    width, height = frame_integer(value["width"], 1920), frame_integer(value["height"], 1920)
    if not width or not height or width * height > 1920**2:
        raise ValueError("Invalid frame dimensions")
    size = width * height * 3
    encoded = value["rgb_base64"]
    if not isinstance(encoded, str) or len(encoded) != 4 * ((size + 2) // 3):
        raise ValueError("Invalid RGB encoding length")
    rgb = base64.b64decode(encoded, validate=True)
    if len(rgb) != size or hashlib.sha256(rgb).hexdigest() != value["rgb_sha256"]:
        raise ValueError("RGB hash or length differs")
    if (
        value["camera_key"] != "observation.images.front"
        or not isinstance(value["camera_prim"], str)
        or not re.fullmatch(r"(/[A-Za-z_][A-Za-z_0-9]*)+", value["camera_prim"])
    ):
        raise ValueError("Invalid camera metadata")
    joints, state = value["joints"], value["state_rad"]
    if (
        not isinstance(joints, list)
        or not 1 <= len(joints) <= 32
        or not isinstance(state, list)
        or len(state) != len(joints)
    ):
        raise ValueError("Invalid joint dimensions")
    if any(
        not isinstance(name, str) or not name.isidentifier() or len(name) > 96 for name in joints
    ) or len(set(joints)) != len(joints):
        raise ValueError("Invalid native joint names")
    if any(type(item) not in (int, float) or not math.isfinite(item) for item in state):
        raise ValueError("Invalid native joint values")
    # Keep source timestamps intact, but provide a decimal identity for JS clients
    # whose number type cannot exactly represent every 64-bit monotonic timestamp.
    return {**value, "relay_age_ns": relay_age, "capture_id": str(acquired)}


@router.get("/frame")
async def frame(response: Response, session_id: str | None = Query(default=None, pattern=IDENTITY)):
    response.headers["Cache-Control"] = "no-store"
    started = time.monotonic_ns()
    value = await relay("/frame", maximum=16 * 1024**2)
    try:
        return public_frame(value, session_id=session_id, elapsed_ns=time.monotonic_ns() - started)
    except KeyError, TypeError, ValueError, OverflowError, binascii.Error:
        raise HTTPException(503, "Teaching frame is unavailable, stale or invalid.") from None


def public_receipt(value):
    if (
        not isinstance(value.get("command_id"), str)
        or not re.fullmatch(IDENTITY, value["command_id"])
        or not isinstance(value.get("request_sha256"), str)
        or not re.fullmatch(r"[a-f0-9]{64}", value["request_sha256"])
    ):
        raise HTTPException(503, "Teaching acknowledgement is invalid.")
    if value.get("status") not in {"queued", "executing", "acknowledged", "rejected"}:
        raise HTTPException(503, "Teaching acknowledgement is invalid.")
    for key in ("received_monotonic_ns", "acknowledged_monotonic_ns"):
        if key in value and (type(value[key]) is not int or not 0 <= value[key] < 2**63):
            raise HTTPException(503, "Teaching acknowledgement timing is invalid.")
    result = {
        key: item
        for key, item in value.items()
        if key
        in {
            "command_id",
            "status",
            "request_sha256",
            "received_monotonic_ns",
            "acknowledged_monotonic_ns",
        }
    }
    if "state" in value:
        result["state"] = public_state(value["state"])
    if value.get("status") == "rejected":
        result["error"] = "Command rejected by the executor; refresh state and inspect the session."
    return result


@router.post("/commands", status_code=202)
async def command(payload: TeachingCommand, response: Response):
    response.headers["Cache-Control"] = "no-store"
    value = public_receipt(await relay("/commands", payload.model_dump()))
    if value["command_id"] != payload.command_id:
        raise HTTPException(503, "Teaching acknowledgement identity did not match.")
    return value


@router.get("/commands/{command_id}")
async def receipt(command_id: str, response: Response):
    if not re.fullmatch(IDENTITY, command_id):
        raise HTTPException(422, "Invalid command identity")
    response.headers["Cache-Control"] = "no-store"
    value = public_receipt(await relay("/commands/" + command_id))
    if value["command_id"] != command_id:
        raise HTTPException(503, "Teaching acknowledgement identity did not match.")
    return value


@router.post("/voice/join")
async def voice_join(payload: VoiceJoin, response: Response):
    response.headers["Cache-Control"] = "no-store"
    value = await relay("/voice/join", payload.model_dump(), voice=True)
    try:
        url = urlsplit(value["url"])
        if (
            value["session_id"] != payload.session_id
            or value["status"] != "room_created"
            or not re.fullmatch(r"firebird-teaching-[a-f0-9]{32}", value["room"])
            or not re.fullmatch(r"operator-[a-f0-9]{32}", value["operator_identity"])
            or value["agent_name"] != "firebird-teaching"
            or type(value["expires_in_seconds"]) is not int
            or not 0 < value["expires_in_seconds"] <= 300
            or not isinstance(value["token"], str)
            or not 16 <= len(value["token"]) <= 16384
            or url.scheme != "wss"
            or not url.hostname
            or url.username
            or url.password
        ):
            raise ValueError("Invalid voice access")
    except KeyError, TypeError, ValueError:
        raise HTTPException(503, "Voice room did not match the teaching session.") from None
    return {
        key: value[key]
        for key in (
            "url",
            "token",
            "room",
            "operator_identity",
            "session_id",
            "expires_in_seconds",
            "agent_name",
        )
    }
