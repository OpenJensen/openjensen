"""Explicit optional advice and private settings; no jobs, controls or provider secrets returned."""

import asyncio
import json
import math
from typing import Literal

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from vla_platform import teaching_settings

IDENTITY = r"^[A-Za-z0-9_-]{1,96}\z"
HASH = r"^[a-f0-9]{64}$"
CHOICES = {"review_instruction", "inspect_camera", "review_recording"}
MAX_BODY = 16384
BROKER_DEADLINE = 16


class PrivateRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def bounded(request: Request):
            if request.method in {"POST", "PUT"}:
                raw = bytearray()
                try:
                    async with asyncio.timeout(4):
                        async for chunk in request.stream():
                            if len(raw) + len(chunk) > MAX_BODY:
                                raise HTTPException(413, "Teaching request exceeds limits.")
                            raw.extend(chunk)
                except TimeoutError:
                    raise HTTPException(408, "Teaching request body timed out.") from None
                # Starlette's cached body allows normal typed FastAPI validation after
                # bounding the incoming stream. Never echo raw input validation details.
                request._body = bytes(raw)
            try:
                response = await handler(request)
                response.headers["Cache-Control"] = "no-store"
                return response
            except RequestValidationError:
                raise HTTPException(422, "Teaching request fields are invalid.") from None

        return bounded


router = APIRouter(tags=["Teaching intelligence"], route_class=PrivateRoute)


class Strict(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def literal_types(cls, value):
        if isinstance(value, dict):
            for name, field in cls.model_fields.items():
                annotation = field.annotation
                from typing import get_args, get_origin

                if name in value and get_origin(annotation) is Literal:
                    if not any(type(value[name]) is type(item) for item in get_args(annotation)):
                        raise ValueError("Invalid literal type")
        return value

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Context(Strict):
    session_id: str = Field(pattern=IDENTITY)
    episode_id: str | None = Field(pattern=IDENTITY)
    revision: int = Field(ge=0, le=2**53 - 1)


class AdviceRequest(Strict):
    request_id: str = Field(pattern=IDENTITY)
    session_id: str = Field(pattern=IDENTITY)
    episode_id: str | None = Field(pattern=IDENTITY)
    expected_revision: int = Field(ge=0, le=2**53 - 1)
    prompt: str = Field(min_length=1, max_length=512)
    consent: Literal[True]

    @model_validator(mode="after")
    def text(self):
        if not self.prompt.strip() or any(
            ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in self.prompt
        ):
            raise ValueError("Expected bounded printable text")
        return self


class CancelRequest(Strict):
    request_id: str = Field(pattern=IDENTITY)
    session_id: str = Field(pattern=IDENTITY)


class FrameIdentity(Strict):
    context: Context
    step: int = Field(ge=0, le=2**53 - 1)
    rgb_sha256: str = Field(pattern=HASH)
    width: int = Field(ge=1, le=1920)
    height: int = Field(ge=1, le=1920)


class Provenance(Strict):
    context: Context
    requested_model: str = Field(max_length=128)
    returned_model: str = Field(max_length=128)
    request_sha256: str = Field(pattern=HASH)
    response_id: str | None = Field(max_length=256)
    reported_cost_usd: float | None = Field(ge=0, le=1e6)
    elapsed_seconds: float = Field(ge=0, le=16)
    frame: FrameIdentity | None
    generated_proposal: Literal[True]


class DecisionProposal(Strict):
    choice: str
    probabilities: dict[str, float]
    confidence: float = Field(ge=0, le=1)
    receipt: Provenance


class Observation(Strict):
    label: str = Field(min_length=1, max_length=64)
    box: list[float] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def rectangle(self):
        a, b, c, d = self.box
        if not (0 <= a < c <= 1 and 0 <= b < d <= 1):
            raise ValueError("Invalid image rectangle")
        return self


class PerceptionProposal(Strict):
    summary: str = Field(min_length=1, max_length=2048)
    uncertain: bool
    observations: list[Observation] = Field(max_length=16)
    receipt: Provenance


class Choice(Strict):
    id: str
    description: str = Field(min_length=1, max_length=512)


class AdviceResult(Strict):
    schema_version: Literal[1]
    request_id: str = Field(pattern=IDENTITY)
    kind: Literal["decision", "perception"]
    advisory_only: Literal[True]
    current_at_return: Literal[True]
    proposal: DecisionProposal | PerceptionProposal
    choices: list[Choice] = Field(max_length=3)

    @model_validator(mode="after")
    def semantics(self):
        import re

        if self.kind == "decision":
            if not isinstance(self.proposal, DecisionProposal):
                raise ValueError("Wrong proposal kind")
            p = self.proposal
            ids = [item.id for item in self.choices]
            if (
                len(ids) not in {2, 3}
                or len(set(ids)) != len(ids)
                or not set(ids) <= CHOICES
                or set(p.probabilities) != set(ids)
                or p.choice not in ids
            ):
                raise ValueError("Invalid eligible choice IDs")
            if any(not 0 <= value <= 1 for value in p.probabilities.values()) or not math.isclose(
                sum(p.probabilities.values()), 1, abs_tol=1e-5
            ):
                raise ValueError("Invalid choice probabilities")
            if (
                p.receipt.requested_model != "typesafe/jev-1.13"
                or not re.fullmatch(r"typesafe/jev-1\.13(?:-\d{8})?", p.receipt.returned_model)
                or p.receipt.frame is not None
            ):
                raise ValueError("Wrong decision model or provenance")
        else:
            p = self.proposal
            if (
                not isinstance(p, PerceptionProposal)
                or self.choices
                or p.receipt.requested_model != "perceptron/perceptron-mk1.5"
                or p.receipt.returned_model != p.receipt.requested_model
                or p.receipt.frame is None
                or p.receipt.frame.context != p.receipt.context
                or p.receipt.frame.width * p.receipt.frame.height > 1024**2
            ):
                raise ValueError("Wrong perception model or frame provenance")
        return self


class BrokerStatus(Strict):
    schema_version: Literal[1]
    broker_reachable: Literal[True]
    configured: bool
    configuration_revision: str | None = Field(pattern=r"^[a-f0-9]{32}$")
    credential_source: Literal["environment", "managed_file", "missing_or_invalid"]
    voice_model: str | None = Field(max_length=192)
    busy: bool
    provider_access_verified: Literal[False]
    decision_model: Literal["typesafe/jev-1.13"]
    perception_model: Literal["perceptron/perceptron-mk1.5"]


class VoiceStatus(Strict):
    schema_version: Literal[1]
    broker_reachable: Literal[True]
    configuration_present: bool
    dependencies_present: bool
    voice_model: str | None = Field(max_length=192)
    provider_access_verified: Literal[False]
    agent_connected: Literal[False]
    message: str = Field(max_length=512)


class CancelResult(Strict):
    schema_version: Literal[1]
    request_id: str = Field(pattern=IDENTITY)
    status: Literal["cancellation_requested", "not_active"]
    billing_verified: Literal[False]


def directory(request: Request):
    return request.app.state.execution.settings.data_dir


async def broker(path, payload=None):
    from vla_platform.teaching_api import configuration, finite_float, unique_object

    origin, token = configuration(voice=True)
    try:
        async with (
            asyncio.timeout(BROKER_DEADLINE),
            httpx.AsyncClient(
                timeout=httpx.Timeout(BROKER_DEADLINE, connect=2),
                trust_env=False,
                follow_redirects=False,
            ) as client,
        ):
            async with client.stream(
                "GET" if payload is None else "POST",
                origin + path,
                json=payload,
                headers={"Authorization": "Bearer " + token, "Accept-Encoding": "identity"},
            ) as response:
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise ValueError("Unsupported broker encoding")
                raw = bytearray()
                async for part in response.aiter_raw():
                    if len(raw) + len(part) > 128 * 1024:
                        raise ValueError("Oversized broker response")
                    raw.extend(part)
                value = json.loads(
                    raw.decode("utf-8"),
                    object_pairs_hook=unique_object,
                    parse_float=finite_float,
                    parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
                )
                if not isinstance(value, dict):
                    raise ValueError("Invalid broker response")
                if response.status_code != 200:
                    code = value.get("code")
                    messages = {
                        "stale_frame": (
                            "The camera frame changed or expired. The proposal was withheld; "
                            "request a fresh observation explicitly."
                        ),
                        "stale_context": (
                            "The teaching session or revision changed. Review its state before "
                            "another request."
                        ),
                        "cancelled": "Request cancelled. Provider billing remains unverified.",
                        "duplicate_request": (
                            "This request was already attempted. It was not submitted again."
                        ),
                        "busy": "Another intelligence request is still in progress.",
                        "configuration_missing": (
                            "Intelligence credentials or the configured worker are unavailable."
                        ),
                        "authentication_failed": "OpenRouter rejected its configured key.",
                        "credit_required": "OpenRouter requires account credit.",
                        "rate_limited": (
                            "OpenRouter rate limited this request. No automatic retry was made."
                        ),
                        "timeout": (
                            "The provider deadline expired. Billing and request completion "
                            "remain unverified."
                        ),
                    }
                    raise HTTPException(
                        409
                        if code
                        in {
                            "stale_frame",
                            "stale_context",
                            "cancelled",
                            "duplicate_request",
                            "busy",
                        }
                        else 503,
                        messages.get(
                            code,
                            (
                                "Intelligence request was unavailable or invalid. No automatic "
                                "retry was made."
                            ),
                        ),
                    )
                return value
    except httpx.HTTPError, ValueError, TimeoutError, RecursionError:
        raise HTTPException(
            503,
            (
                "Intelligence broker is unavailable. Request completion and "
                "billing may be unverified; no automatic retry was made."
            ),
        ) from None


@router.get("/intelligence/settings")
async def get_settings(request: Request):
    return teaching_settings.status(directory(request))


@router.put("/intelligence/settings")
async def save_settings(payload: teaching_settings.IntelligenceSettingsInput, request: Request):
    try:
        return teaching_settings.save(directory(request), payload)
    except OSError, ValueError:
        raise HTTPException(503, "Could not save private intelligence settings.") from None


@router.delete("/intelligence/settings")
async def remove_settings(request: Request):
    try:
        return teaching_settings.remove(directory(request))
    except OSError, ValueError:
        raise HTTPException(
            503, "Could not remove the app-managed intelligence settings."
        ) from None


@router.get("/intelligence/status")
async def intelligence_status():
    try:
        return BrokerStatus.model_validate(await broker("/intelligence/status"))
    except HTTPException, ValidationError:
        return {
            "schema_version": 1,
            "broker_reachable": False,
            "configured": False,
            "busy": False,
            "provider_access_verified": False,
            "message": (
                "The optional teaching intelligence broker is not connected or "
                "returned invalid readiness. Existing manual teaching remains "
                "available."
            ),
        }


@router.get("/voice/status")
async def voice_status():
    try:
        return VoiceStatus.model_validate(await broker("/voice/status"))
    except HTTPException, ValidationError:
        return {
            "schema_version": 1,
            "broker_reachable": False,
            "configuration_present": False,
            "dependencies_present": False,
            "provider_access_verified": False,
            "agent_connected": False,
            "voice_model": None,
            "message": (
                "Voice broker is not connected. No microphone, agent or provider "
                "connection has been established."
            ),
        }


async def complete_owned(task):
    interrupted = False
    while True:
        try:
            return await asyncio.shield(task), interrupted
        except asyncio.CancelledError:
            if task.done():
                raise
            interrupted = True


async def cancel_owned(payload):
    try:
        async with asyncio.timeout(5):
            await broker(
                "/intelligence/cancel",
                {"request_id": payload.request_id, "session_id": payload.session_id},
            )
    except HTTPException, TimeoutError:
        pass  # Best effort only; the worker's independent deadline remains active.


async def advise(kind, payload, request):
    call = asyncio.create_task(broker("/intelligence/" + kind, payload.model_dump()))

    async def watch_disconnect():
        while not await request.is_disconnected():
            await asyncio.sleep(0.1)

    disconnected = asyncio.create_task(watch_disconnect())
    try:
        done, _ = await asyncio.wait({call, disconnected}, return_when=asyncio.FIRST_COMPLETED)
        if disconnected in done:
            raise asyncio.CancelledError
        value = await call
    except BaseException:
        call.cancel()
        await complete_owned(asyncio.create_task(cancel_owned(payload)))
        raise
    finally:
        disconnected.cancel()
        _, interrupted = await complete_owned(
            asyncio.ensure_future(asyncio.gather(call, disconnected, return_exceptions=True))
        )
        if interrupted:
            await complete_owned(asyncio.create_task(cancel_owned(payload)))
            raise asyncio.CancelledError
    try:
        result = AdviceResult.model_validate(value)
        expected = Context(
            session_id=payload.session_id,
            episode_id=payload.episode_id,
            revision=payload.expected_revision,
        )
        if (
            result.kind != kind
            or result.request_id != payload.request_id
            or result.proposal.receipt.context != expected
        ):
            raise ValueError("Mismatched request receipt")
        return result
    except ValidationError, ValueError:
        raise HTTPException(
            503, "Intelligence receipt did not match the request; proposal withheld."
        ) from None


@router.post("/intelligence/decision", response_model=AdviceResult)
async def decision(payload: AdviceRequest, request: Request):
    return await advise("decision", payload, request)


@router.post("/intelligence/perception", response_model=AdviceResult)
async def perception(payload: AdviceRequest, request: Request):
    return await advise("perception", payload, request)


@router.post("/intelligence/cancel", response_model=CancelResult)
async def cancel(payload: CancelRequest):
    try:
        result = CancelResult.model_validate(
            await broker("/intelligence/cancel", payload.model_dump())
        )
        if result.request_id != payload.request_id:
            raise ValueError("Mismatched cancellation identity")
        return result
    except ValidationError, ValueError:
        raise HTTPException(503, "Cancellation outcome is unverified.") from None
