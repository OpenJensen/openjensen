"""Manual session-bound advice through the existing authenticated loopback broker."""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from dataclasses import asdict

import httpx

from .contracts import IDENTITY, canonical, decode
from .frame_snapshot import admit_frame
from .intelligence_config import load_config
from .providers import Choice, ProviderError, Providers, ProviderSettings, SourceContext

DEADLINE = 12
CHOICES = {
    "review_instruction": "Review or clarify the task instruction with the operator.",
    "inspect_camera": "Inspect a fresh camera observation with the operator.",
    "review_recording": "Review the recorded demonstration before using it for training.",
}


class IntelligenceService:
    """One bounded request, fixed destinations, no control methods or automatic retry."""

    def __init__(
        self, control, *, provider_factory=Providers, config_loader=load_config, transport=None
    ):
        self.control = control
        self.provider_factory = provider_factory
        self.config_loader = config_loader
        self.transport = transport
        self.lock = threading.Lock()
        self.active = None
        self.consumed = deque(maxlen=256)

    def status(self):
        try:
            cfg = self.config_loader()
            configured, revision, source, model = (
                True,
                cfg.revision,
                cfg.key_source,
                cfg.voice_model,
            )
        except (ValueError, OSError):
            configured, revision, source, model = False, None, "missing_or_invalid", None
        with self.lock:
            busy = self.active is not None
        return {
            "schema_version": 1,
            "broker_reachable": True,
            "configured": configured,
            "configuration_revision": revision,
            "credential_source": source,
            "voice_model": model or None,
            "busy": busy,
            "provider_access_verified": False,
            "decision_model": "typesafe/jev-1.13",
            "perception_model": "perceptron/perceptron-mk1.5",
        }

    @staticmethod
    def validate(payload, *, cancel=False):
        expected = {"request_id", "session_id"}
        if not cancel:
            expected |= {"episode_id", "expected_revision", "prompt", "consent"}
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ProviderError("invalid_input", "Invalid intelligence request fields.")
        for name in ("request_id", "session_id"):
            if not isinstance(payload[name], str) or not IDENTITY.fullmatch(payload[name]):
                raise ProviderError("invalid_input", "Invalid intelligence request identity.")
        if cancel:
            return
        episode, revision, prompt = (
            payload["episode_id"],
            payload["expected_revision"],
            payload["prompt"],
        )
        if episode is not None and (
            not isinstance(episode, str) or not IDENTITY.fullmatch(episode)
        ):
            raise ProviderError("invalid_input", "Invalid episode identity.")
        if type(revision) is not int or not 0 <= revision <= 2**53 - 1:
            raise ProviderError("invalid_input", "Invalid context revision.")
        if (
            payload["consent"] is not True
            or not isinstance(prompt, str)
            or not prompt.strip()
            or not 1 <= len(prompt) <= 512
            or any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in prompt)
        ):
            raise ProviderError(
                "invalid_input", "Explicit consent and a bounded prompt are required."
            )

    def cancel(self, payload):
        self.validate(payload, cancel=True)
        with self.lock:
            active = self.active
            matched = (
                active is not None
                and active["request_id"] == payload["request_id"]
                and active["session_id"] == payload["session_id"]
                and not active.get("completed", False)
            )
            if matched:
                active["cancelled"] = True
                if active.get("task") is not None:
                    try:
                        active["loop"].call_soon_threadsafe(active["task"].cancel)
                    except RuntimeError:
                        matched = False  # The completed event loop has already closed.
        return {
            "schema_version": 1,
            "request_id": payload["request_id"],
            "status": "cancellation_requested" if matched else "not_active",
            "billing_verified": False,
        }

    def run(self, kind, payload):
        if kind not in {"decision", "perception"}:
            raise ProviderError("invalid_input", "Unknown intelligence request.")
        self.validate(payload)
        with self.lock:
            if payload["request_id"] in self.consumed:
                raise ProviderError(
                    "duplicate_request", "This request was already attempted; no automatic retry."
                )
            if self.active is not None:
                raise ProviderError("busy", "Another intelligence request is in progress.")
            record = {
                "request_id": payload["request_id"],
                "session_id": payload["session_id"],
                "cancelled": False,
            }
            self.active = record
            self.consumed.append(payload["request_id"])
        try:
            return asyncio.run(self._owned(kind, payload, record))
        finally:
            with self.lock:
                if self.active is record:
                    self.active = None

    async def _owned(self, kind, payload, record):
        with self.lock:
            record.update(loop=asyncio.get_running_loop(), task=asyncio.current_task())
            cancelled = record["cancelled"]
        if cancelled:
            raise ProviderError("cancelled", "Request cancelled; provider billing is unverified.")
        try:
            async with asyncio.timeout(DEADLINE):
                result = await self._run(kind, payload)
                # Linearize response completion against the separate cancellation thread.
                with self.lock:
                    if record["cancelled"]:
                        raise ProviderError(
                            "cancelled", "Request cancelled; billing is unverified."
                        )
                    record["completed"] = True
                return result
        except asyncio.CancelledError:
            raise ProviderError(
                "cancelled", "Request cancelled; provider billing is unverified."
            ) from None
        except TimeoutError:
            raise ProviderError(
                "timeout", "Intelligence deadline exceeded; no automatic retry."
            ) from None
        except (ValueError, OSError) as error:
            if isinstance(error, ProviderError):
                raise
            raise ProviderError(
                "configuration_missing", "Intelligence configuration or observation is unavailable."
            ) from None

    async def read(self, path):
        try:
            async with httpx.AsyncClient(
                timeout=3, trust_env=False, follow_redirects=False, transport=self.transport
            ) as client:
                async with client.stream(
                    "GET",
                    self.control.origin + path,
                    headers={
                        "Authorization": "Bearer " + self.control.token,
                        "Accept-Encoding": "identity",
                    },
                ) as response:
                    if (
                        response.status_code != 200
                        or response.headers.get("Content-Encoding", "identity") != "identity"
                    ):
                        raise ValueError("Unavailable executor")
                    raw = bytearray()
                    async for chunk in response.aiter_raw():
                        if len(raw) + len(chunk) > 16 * 1024**2:
                            raise ValueError("Oversized executor response")
                        raw.extend(chunk)
            return decode(bytes(raw), limit=16 * 1024**2)
        except (httpx.HTTPError, ValueError):
            raise ProviderError(
                "executor_unavailable", "The teaching executor or observation is unavailable."
            ) from None

    @staticmethod
    def context(state, payload):
        context = SourceContext(
            state.get("session_id"), state.get("episode_id"), state.get("revision")
        )
        expected = SourceContext(
            payload["session_id"], payload["episode_id"], payload["expected_revision"]
        )
        if context != expected or state.get("mode") not in {"idle", "running", "paused"}:
            raise ProviderError(
                "stale_context", "The teaching context changed. Review it before another request."
            )
        return context

    async def frame(self, payload):
        started = time.monotonic_ns()
        raw = await self.read("/frame")
        try:
            admitted = admit_frame(
                raw,
                expected_session_id=payload["session_id"],
                request_started_monotonic_ns=started,
                received_monotonic_ns=time.monotonic_ns(),
            )
            if (
                admitted.observation.revision != payload["expected_revision"]
                or admitted.observation.active_episode_id != payload["episode_id"]
            ):
                raise ValueError("Context changed")
            return admitted
        except (ValueError, KeyError, TypeError):
            raise ProviderError(
                "stale_frame", "A fresh unchanged frame is required. No stale image was accepted."
            ) from None

    async def _run(self, kind, payload):
        cfg = self.config_loader()
        provider = self.provider_factory(ProviderSettings(cfg.key))
        state = await self.read("/state")
        context = self.context(state, payload)
        if kind == "decision":
            choices = tuple(
                Choice(name, description)
                for name, description in CHOICES.items()
                if name != "review_recording"
                or (type(state.get("steps")) is int and state["steps"] > 0)
            )
            observed = {
                name: state.get(name) for name in ("instruction", "mode", "steps", "sim_time")
            }
            summary = canonical(observed).decode()
            result = await provider.decide(
                context, summary, payload["prompt"], choices, current=lambda: context
            )
        else:
            admitted = await self.frame(payload)
            source = admitted.provider_input()
            result = await provider.perceive(
                source, payload["prompt"], current=lambda: source.identity
            )
            # A changed capture or elapsed source age invalidates the returned proposal.
            final = await self.frame(payload)
            if final.observation != admitted.observation:
                raise ProviderError(
                    "stale_frame",
                    "The camera observation changed during inference; result withheld.",
                )
            provider._fresh(source, lambda: final.provider_input().identity)
        self.context(await self.read("/state"), payload)
        proposal = asdict(result)
        if kind == "decision":
            proposal["probabilities"] = dict(result.probabilities)
        return {
            "schema_version": 1,
            "request_id": payload["request_id"],
            "kind": kind,
            "advisory_only": True,
            "current_at_return": True,
            "proposal": proposal,
            "choices": [{"id": item.id, "description": item.description} for item in choices]
            if kind == "decision"
            else [],
        }
