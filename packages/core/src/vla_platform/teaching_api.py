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

from vla_platform.environment import getenv
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
    raw = getenv("FIREBIRD_TEACHING_VOICE_URL" if voice else "FIREBIRD_TEACHING_URL")
    token_path = getenv("JENSEN_TEACHING_CONTROL_TOKEN_FILE")
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
    agent_owner: str | None = Field(default=None, pattern=IDENTITY)
    authority_epoch: int = Field(default=0, ge=0, le=2**53 - 1)
    agent_clock_policy: Literal["stepped", "continuous"] | None = None
    agent_capabilities: dict[str, bool | int] = Field(default_factory=dict)


def public_state(value):
    # Never forward arbitrary worker fields, local recording paths or credentials.
    try:
        result = PublicState.model_validate(value).model_dump()
    except ValidationError:
        raise HTTPException(503, "Teaching state is invalid.") from None
    caps = result["agent_capabilities"]
    if caps:
        required = {
            "schema_version",
            "named_joint_correction",
            "stepped_capture",
            "physical_execution",
        }
        extended = {
            "named_joint_segment",
            "epoch_heartbeat",
            "max_segment_ticks",
            "max_segment_duration_ms",
        }
        if (
            type(caps.get("schema_version")) is not int
            or caps["schema_version"] not in {1, 2}
            or set(caps) != (required if caps["schema_version"] == 1 else required | extended)
            or any(
                type(caps[key]) is not bool
                for key in ("named_joint_correction", "stepped_capture", "physical_execution")
            )
            or caps["physical_execution"]
        ):
            raise HTTPException(503, "Teaching agent capabilities are invalid.")
        if caps["schema_version"] == 2 and (
            type(caps["named_joint_segment"]) is not bool
            or type(caps["epoch_heartbeat"]) is not bool
            or type(caps["max_segment_ticks"]) is not int
            or not 1 <= caps["max_segment_ticks"] <= 60
            or type(caps["max_segment_duration_ms"]) is not int
            or not 100 <= caps["max_segment_duration_ms"] <= 2000
        ):
            raise HTTPException(503, "Teaching segment capabilities are invalid.")
    if result["fault"]:
        result["fault"] = "Executor stopped after a fault; inspect its local log."
    if not result["agent_capabilities"]:
        for key in ("agent_owner", "authority_epoch", "agent_clock_policy", "agent_capabilities"):
            result.pop(key, None)
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
        "voice_configured": bool(getenv("JENSEN_TEACHING_VOICE_URL")),
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


# Optional advice shares the existing local access boundary but owns no controls.
from vla_platform.teaching_intelligence_api import router as intelligence_router  # noqa: E402

router.include_router(intelligence_router)


AGENT_PATHS = {
    "/agent/acquire",
    "/agent/observe",
    "/agent/execute",
    "/agent/release",
    "/agent/heartbeat",
    "/agent/segment",
}
AGENT_OBSERVATION_FIELDS = {
    "schema_version",
    "observation_id",
    "observation_sha256",
    "target_session_id",
    "agent_session_id",
    "episode_id",
    "authority_epoch",
    "revision",
    "embodiment_manifest_id",
    "scene_sha256",
    "calibration_id",
    "clock_policy",
    "captured_monotonic_ns",
    "valid_until_monotonic_ns",
    "source_age_ms",
    "step",
    "sim_time",
    "joints",
    "state_rad",
    "units",
    "image_data_url",
    "image_sha256",
}


def validate_agent_request(path: str, payload: dict | None) -> None:
    """Private host-to-executor protocol; no URLs or unbounded tools from a model."""
    if payload is None:
        if path == "/state" or (
            path.startswith("/agent/executions/")
            and re.fullmatch(IDENTITY, path.removeprefix("/agent/executions/"))
        ):
            return
        raise ValueError("Unsupported agent executor query")
    if path not in AGENT_PATHS or set(payload) != {
        "command_id",
        "session_id",
        "agent_session_id",
        "authority_epoch",
        "arguments",
    }:
        raise ValueError("Unsupported agent executor operation")
    for key in ("command_id", "session_id", "agent_session_id"):
        if not isinstance(payload[key], str) or not re.fullmatch(IDENTITY, payload[key]):
            raise ValueError("Invalid agent executor identity")
    frame_integer(payload["authority_epoch"], 2**53 - 1)
    args = payload["arguments"]
    operation = path.rsplit("/", 1)[-1]
    expected = {
        "acquire": {"clock_policy", "expected_revision", "episode_id"},
        "observe": set(),
        "release": set(),
        "heartbeat": set(),
        "segment": {
            "observation_id",
            "observation_sha256",
            "targets",
            "duration_seconds",
            "proposal_id",
            "provider_model",
        },
        "execute": {
            "observation_id",
            "observation_sha256",
            "joint",
            "delta_rad",
            "proposal_id",
            "provider_model",
        },
    }[operation]
    optional = (
        {"max_session_seconds", "planning_snapshot_seconds"}
        if operation == "acquire"
        else {"prompt_version", "prompt_sha256"}
        if operation in {"execute", "segment"}
        else set()
    )
    if not isinstance(args, dict) or not expected <= set(args) or set(args) - expected - optional:
        raise ValueError("Unsupported agent executor arguments")
    if operation == "acquire" and (
        args["clock_policy"] not in {"stepped", "continuous"} or payload["authority_epoch"] != 0
    ):
        raise ValueError("Invalid agent acquire contract")
    if operation == "acquire":
        if "max_session_seconds" in args and (
            type(args["max_session_seconds"]) is not int
            or not 1 <= args["max_session_seconds"] <= 600
        ):
            raise ValueError("Invalid maximum agent session duration")
        if "planning_snapshot_seconds" in args and (
            type(args["planning_snapshot_seconds"]) is not int
            or not 1 <= args["planning_snapshot_seconds"] <= 120
            or args["clock_policy"] != "stepped"
        ):
            raise ValueError("Invalid frozen planning snapshot duration")
        frame_integer(args["expected_revision"], 2**53 - 1)
        if not isinstance(args["episode_id"], str) or not re.fullmatch(
            IDENTITY, args["episode_id"]
        ):
            raise ValueError("Invalid acquire episode identity")
    if operation in {"execute", "segment"}:
        prompt_keys = {"prompt_version", "prompt_sha256"}
        if prompt_keys & set(args) and (
            not prompt_keys <= set(args)
            or not isinstance(args["prompt_version"], str)
            or re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", args["prompt_version"]) is None
            or not isinstance(args["prompt_sha256"], str)
            or re.fullmatch(r"[a-f0-9]{64}", args["prompt_sha256"]) is None
        ):
            raise ValueError("Invalid prompt provenance")
        if operation == "execute":
            TeachingCommand(
                session_id=payload["session_id"],
                command_id=payload["command_id"],
                expected_revision=0,
                operation="correct",
                arguments={"joint": args["joint"], "delta_rad": args["delta_rad"]},
            )
        else:
            targets, duration = args["targets"], args["duration_seconds"]
            if (
                not isinstance(targets, dict)
                or not 1 <= len(targets) <= 32
                or any(
                    not isinstance(k, str)
                    or not k.isidentifier()
                    or type(v) not in (int, float)
                    or not math.isfinite(v)
                    for k, v in targets.items()
                )
                or type(duration) not in (int, float)
                or not math.isfinite(duration)
                or not 0.1 <= duration <= 2
            ):
                raise ValueError("Invalid simulator segment contract")
        for key in ("observation_id", "proposal_id"):
            if not isinstance(args[key], str) or not re.fullmatch(IDENTITY, args[key]):
                raise ValueError("Invalid agent proposal identity")
        if not isinstance(args["observation_sha256"], str) or not re.fullmatch(
            r"[a-f0-9]{64}", args["observation_sha256"]
        ):
            raise ValueError("Invalid original observation fingerprint")
        if (
            not isinstance(args["provider_model"], str)
            or not 1 <= len(args["provider_model"]) <= 128
            or not args["provider_model"].isprintable()
        ):
            raise ValueError("Invalid provider model identity")


def public_task_evidence(value, *, step, sim_time):
    """Measured geometry is a named criterion, never a general grasp or physical claim."""
    required = {
        "schema_version",
        "source",
        "criteria",
        "criteria_sha256",
        "object_prim",
        "step",
        "sim_time",
        "samples",
        "whole_object_clearance_m",
        "lift_hold_seconds",
        "terminal_settle_seconds",
        "geometric_success",
        "grasp_and_release_verified",
        "physical_transfer_verified",
    }
    criteria_fields = {
        "task",
        "object_prim",
        "support_height_m",
        "lift_clearance_m",
        "lift_hold_seconds",
        "settle_seconds",
        "maximum_linear_speed_m_s",
        "maximum_angular_speed_rad_s",
        "placement_min_m",
        "placement_max_m",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Invalid independent task evidence")
    criteria = value["criteria"]
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["source"] != "isaac_object_bounds_v1"
        or not isinstance(criteria, dict)
        or set(criteria) != criteria_fields
        or criteria["task"] not in {"lift_hold", "lift_place"}
        or value["object_prim"] != criteria["object_prim"]
        or not isinstance(value["object_prim"], str)
        or not re.fullmatch(r"(/[A-Za-z_][A-Za-z_0-9]*)+", value["object_prim"])
        or type(value["step"]) is not int
        or value["step"] != step
        or type(value["samples"]) is not int
        or value["samples"] != step + 1
        or type(value["sim_time"]) not in (int, float)
        or value["sim_time"] != sim_time
        or type(value["geometric_success"]) is not bool
        or value["grasp_and_release_verified"] is not False
        or value["physical_transfer_verified"] is not False
    ):
        raise ValueError("Independent task evidence does not match the observation")
    for key in ("whole_object_clearance_m", "lift_hold_seconds", "terminal_settle_seconds"):
        if type(value[key]) not in (int, float) or not math.isfinite(value[key]):
            raise ValueError("Nonfinite independent task measurement")
    for key, lower, upper in (
        ("support_height_m", -1000, 1000),
        ("lift_clearance_m", 0.001, 10),
        ("lift_hold_seconds", 0.001, 600),
        ("settle_seconds", 0.001, 600),
        ("maximum_linear_speed_m_s", 0, 100),
        ("maximum_angular_speed_rad_s", 0, 100),
    ):
        if (
            type(criteria[key]) not in (int, float)
            or not math.isfinite(criteria[key])
            or not lower <= criteria[key] <= upper
        ):
            raise ValueError("Invalid independent task criteria")
    if value["lift_hold_seconds"] < 0 or value["terminal_settle_seconds"] < 0:
        raise ValueError("Negative independent task duration")
    for key in ("placement_min_m", "placement_max_m"):
        vector = criteria[key]
        if criteria["task"] == "lift_hold":
            if vector is not None:
                raise ValueError("Lift-only criteria cannot claim placement")
        elif (
            not isinstance(vector, list)
            or len(vector) != 3
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector)
        ):
            raise ValueError("Invalid placement criteria")
    if criteria["task"] == "lift_place" and any(
        lo >= hi
        for lo, hi in zip(criteria["placement_min_m"], criteria["placement_max_m"], strict=True)
    ):
        raise ValueError("Placement criteria must have positive extent")
    digest = hashlib.sha256(
        json.dumps(criteria, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if value["criteria_sha256"] != digest:
        raise ValueError("Independent task criteria fingerprint differs")
    return dict(value)


def public_agent_observation(value: dict, *, session_id: str | None) -> dict:
    if (
        not isinstance(value, dict)
        or not AGENT_OBSERVATION_FIELDS <= set(value)
        or set(value)
        - AGENT_OBSERVATION_FIELDS
        - {
            "task_evidence",
            "scene_frozen_for_planning",
            "snapshot_policy",
            "joint_limits",
            "action_limits",
        }
    ):
        raise ValueError("Invalid agent observation fields")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["units"] != "rad"
    ):
        raise ValueError("Unsupported agent observation contract")
    for key in ("observation_id", "target_session_id", "agent_session_id", "episode_id"):
        if not isinstance(value[key], str) or not re.fullmatch(IDENTITY, value[key]):
            raise ValueError("Invalid agent observation identity")
    if session_id is not None and value["target_session_id"] != session_id:
        raise ValueError("Agent observation belongs to another executor")
    for key in ("observation_sha256", "embodiment_manifest_id", "scene_sha256", "image_sha256"):
        if not isinstance(value[key], str) or not re.fullmatch(r"[a-f0-9]{64}", value[key]):
            raise ValueError("Invalid agent observation fingerprint")
    if value["calibration_id"] is not None or value["clock_policy"] not in {
        "stepped",
        "continuous",
    }:
        raise ValueError("Unsupported simulator clock/calibration contract")
    for key in (
        "authority_epoch",
        "revision",
        "captured_monotonic_ns",
        "valid_until_monotonic_ns",
        "source_age_ms",
        "step",
    ):
        frame_integer(value[key])
    frozen = value.get("scene_frozen_for_planning")
    policy = value.get("snapshot_policy")
    if frozen is not None or policy is not None:
        if (
            type(frozen) is not bool
            or frozen != (value["clock_policy"] == "stepped")
            or policy != ("frozen_step_v1" if frozen else "continuous_age_drift_v1")
        ):
            raise ValueError("Invalid simulator planning clock declaration")
    maximum = (
        120_000_000_000
        if frozen is True
        else 2_000_000_000
        if value["clock_policy"] == "continuous"
        else 30_000_000_000
    )
    if not 0 < value["valid_until_monotonic_ns"] - value["captured_monotonic_ns"] <= maximum:
        raise ValueError("Invalid agent observation validity interval")
    joints, state = value["joints"], value["state_rad"]
    if (
        not isinstance(joints, list)
        or not 1 <= len(joints) <= 32
        or any(not isinstance(j, str) or not j.isidentifier() for j in joints)
        or len(set(joints)) != len(joints)
    ):
        raise ValueError("Invalid agent embodiment joint names")
    if (
        not isinstance(state, list)
        or len(state) != len(joints)
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in state)
    ):
        raise ValueError("Invalid agent measured state")
    if (
        type(value["sim_time"]) not in (int, float)
        or not math.isfinite(value["sim_time"])
        or value["sim_time"] < 0
    ):
        raise ValueError("Invalid agent simulation time")
    encoded = value["image_data_url"]
    if (
        not isinstance(encoded, str)
        or not encoded.startswith("data:image/png;base64,")
        or len(encoded) > 16 * 1024**2
    ):
        raise ValueError("Invalid agent image encoding")
    pixels = base64.b64decode(encoded.removeprefix("data:image/png;base64,"), validate=True)
    if (
        not pixels.startswith(b"\x89PNG\r\n\x1a\n")
        or hashlib.sha256(pixels).hexdigest() != value["image_sha256"]
    ):
        raise ValueError("Agent image fingerprint differs")
    if "joint_limits" in value:
        limits = value["joint_limits"]
        if (
            not isinstance(limits, dict)
            or set(limits) != set(value["joints"])
            or any(
                not isinstance(bounds, list)
                or len(bounds) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds)
                or bounds[0] >= bounds[1]
                for bounds in limits.values()
            )
        ):
            raise ValueError("Invalid native named joint limits")
    if "action_limits" in value:
        limits = value["action_limits"]
        if (
            not isinstance(limits, dict)
            or set(limits) != {"min_duration_seconds", "max_duration_seconds", "max_ticks"}
            or type(limits["max_ticks"]) is not int
            or limits["max_ticks"] != 60
            or any(
                type(limits[key]) not in (int, float) or not math.isfinite(limits[key])
                for key in ("min_duration_seconds", "max_duration_seconds")
            )
            or not 0.1 <= limits["min_duration_seconds"] <= limits["max_duration_seconds"] <= 2
        ):
            raise ValueError("Invalid native segment duration limits")
    if "task_evidence" in value:
        public_task_evidence(value["task_evidence"], step=value["step"], sim_time=value["sim_time"])
    immutable = {key: item for key, item in value.items() if key != "observation_sha256"}
    digest = hashlib.sha256(
        json.dumps(immutable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if digest != value["observation_sha256"]:
        raise ValueError("Original agent observation fingerprint differs")
    return dict(value)


def public_agent_receipt(value: dict, *, session_id: str | None) -> dict:
    result = public_receipt(value)
    if result.get("state") and result["state"]["session_id"] != session_id:
        raise ValueError("Agent acknowledgment belongs to another executor")
    if "observation" in value:
        result["observation"] = public_agent_observation(
            value["observation"], session_id=session_id
        )
        result["observation_age_ms"] = frame_integer(value.get("observation_age_ms", 0))
    if "execution" in value:
        execution = value["execution"]
        allowed = {
            "status",
            "execution_id",
            "observation_id",
            "started_monotonic_ns",
            "completed_monotonic_ns",
            "applied_target_rad",
            "requested_target_rad",
            "next_state_rad",
            "joint_limit_clipped",
            "speed_limit_clipped",
            "task_success_verified",
            "skill",
            "requested_duration_seconds",
            "executed_ticks",
            "simulation_elapsed_seconds",
            "stopped_monotonic_ns",
        }
        if (
            not isinstance(execution, dict)
            or not set(execution) <= allowed
            or execution.get("status") not in {"started", "completed", "stopped"}
            or execution.get("execution_id") != result["command_id"]
        ):
            raise ValueError("Invalid agent execution receipt")
        for key in ("execution_id", "observation_id"):
            if not isinstance(execution.get(key), str) or not re.fullmatch(
                IDENTITY, execution[key]
            ):
                raise ValueError("Invalid agent execution identity")
        for key in ("started_monotonic_ns", "completed_monotonic_ns", "stopped_monotonic_ns"):
            if key in execution:
                frame_integer(execution[key])
        for key in ("applied_target_rad", "requested_target_rad", "next_state_rad"):
            if key in execution and (
                not isinstance(execution[key], list)
                or not 1 <= len(execution[key]) <= 32
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in execution[key])
            ):
                raise ValueError("Invalid applied agent action")
        for key in ("joint_limit_clipped", "speed_limit_clipped"):
            if key in execution and (
                not isinstance(execution[key], list)
                or not 1 <= len(execution[key]) <= 32
                or any(type(v) is not bool for v in execution[key])
            ):
                raise ValueError("Invalid agent clipping provenance")
        if "skill" in execution:
            if (
                execution["skill"] != "joint_segment"
                or type(execution.get("requested_duration_seconds")) not in (int, float)
                or not 0.1 <= execution["requested_duration_seconds"] <= 2
                or type(execution.get("executed_ticks")) is not int
                or not 0 <= execution["executed_ticks"] <= 60
                or type(execution.get("simulation_elapsed_seconds")) not in (int, float)
                or not 0 <= execution["simulation_elapsed_seconds"] <= 2
            ):
                raise ValueError("Invalid simulator segment completion evidence")
        if execution.get("task_success_verified", False) is not False:
            raise ValueError("Agent completion cannot verify task success")
        if execution["status"] == "completed" and not {
            "applied_target_rad",
            "requested_target_rad",
            "next_state_rad",
            "joint_limit_clipped",
            "speed_limit_clipped",
            "completed_monotonic_ns",
            "task_success_verified",
        } <= set(execution):
            raise ValueError("Agent completed receipt has no measured outcome")
        result["execution"] = dict(execution)
    return result


async def agent_relay(path: str, payload: dict | None = None) -> dict:
    """Configured local executor adapter, called by Agent Control rather than a browser route."""
    try:
        validate_agent_request(path, payload)
        raw = await relay(path, payload, maximum=16 * 1024**2)
        if path == "/state":
            return public_state(raw)
        session_id = (
            payload["session_id"] if payload is not None else raw.get("state", {}).get("session_id")
        )
        result = public_agent_receipt(raw, session_id=session_id)
        if payload is not None:
            expected = hashlib.sha256(
                json.dumps(
                    {"operation": path.rsplit("/", 1)[-1], "payload": payload},
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            if (
                result["command_id"] != payload["command_id"]
                or result["request_sha256"] != expected
            ):
                raise ValueError("Agent acknowledgment request identity differs")
        elif result["command_id"] != path.rsplit("/", 1)[-1]:
            raise ValueError("Agent acknowledgment command identity differs")
        return result
    except ValueError, TypeError, KeyError, binascii.Error:
        raise HTTPException(503, "Teaching agent executor returned an invalid response.") from None
