"""Durable, single-flight planning sessions above separately owned local executors."""

import asyncio
import hashlib
import json
import time
from datetime import UTC, datetime, timedelta
from uuid import NAMESPACE_URL, uuid4, uuid5

from sqlalchemy import insert, select, update

from vla_platform.storage import (
    agent_events,
    agent_observations,
    agent_proposals,
    agent_receipts,
    agent_sessions,
)

from . import runtime
from .contracts import (
    Approval,
    Arguments,
    Event,
    ExecutionEvidence,
    Observation,
    Proposal,
    Receipt,
    Session,
    SessionCreate,
)
from .settings import Profiles

TERMINAL = {"stopped", "interrupted", "complete", "error"}


class AgentError(ValueError):
    def __init__(self, message, status=409):
        super().__init__(message)
        self.status = status


def now():
    return datetime.now(UTC).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


class AgentControl:
    def __init__(self, storage, settings, execution, planner=None):
        self.storage = storage
        self.settings = settings
        self.execution = execution
        self.profiles = Profiles(settings.data_dir)
        self.planner = planner or runtime.run
        self.tasks = {}
        self.checks = set()
        self.locks = {}
        self.creation_lock = asyncio.Lock()
        self.closed = False

    @property
    def pending(self):
        return bool(self.checks) or any(not t.done() for t in self.tasks.values())

    def lock(self, identity):
        return self.locks.setdefault(identity, asyncio.Lock())

    async def load(self, identity):
        async with self.storage.engine.connect() as c:
            row = (
                (await c.execute(select(agent_sessions).where(agent_sessions.c.id == identity)))
                .mappings()
                .first()
            )
        if row is None:
            raise AgentError("Agent session not found", 404)
        return Session.model_validate(row["record"]), row["private"]

    async def get(self, identity):
        return (await self.load(identity))[0]

    async def list(self, project_id):
        async with self.storage.engine.connect() as c:
            rows = (
                (
                    await c.execute(
                        select(agent_sessions.c.record).where(
                            agent_sessions.c.project_id == project_id
                        )
                    )
                )
                .scalars()
                .all()
            )
        return sorted(
            (Session.model_validate(r) for r in rows), key=lambda r: r.created_at, reverse=True
        )

    async def save(
        self,
        session,
        private,
        kind,
        record=None,
        *,
        observation=None,
        proposal=None,
        receipt=None,
        receipt_update=False,
        new=False,
    ):
        session.updated_at = now()
        async with self.storage.engine.begin() as c:
            fields = dict(
                id=session.id,
                project_id=session.project_id,
                status=session.status,
                record=session.model_dump(),
                private=private,
            )
            if new:
                await c.execute(insert(agent_sessions).values(**fields))
            else:
                await c.execute(
                    update(agent_sessions).where(agent_sessions.c.id == session.id).values(**fields)
                )
            if observation is not None:
                await c.execute(
                    insert(agent_observations).values(
                        id=session.observation.id,
                        session_id=session.id,
                        project_id=session.project_id,
                        record={"public": session.observation.model_dump(), "raw": observation},
                    )
                )
            if proposal is not None:
                await c.execute(
                    insert(agent_proposals).values(
                        id=proposal.id,
                        session_id=session.id,
                        request_id=proposal.request_id,
                        request_sha256=private["requests"][proposal.request_id],
                        record=proposal.model_dump(),
                    )
                )
            if proposal is None and session.pending_proposal:
                await c.execute(
                    update(agent_proposals)
                    .where(agent_proposals.c.id == session.pending_proposal.id)
                    .values(record=session.pending_proposal.model_dump())
                )
            if receipt is not None:
                if receipt_update:
                    await c.execute(
                        update(agent_receipts)
                        .where(agent_receipts.c.id == receipt.id)
                        .values(record=receipt.model_dump())
                    )
                else:
                    await c.execute(
                        insert(agent_receipts).values(
                            id=receipt.id,
                            session_id=session.id,
                            proposal_id=receipt.proposal_id,
                            record=receipt.model_dump(),
                        )
                    )
                saved_proposal = (
                    await c.execute(
                        select(agent_proposals.c.record).where(
                            agent_proposals.c.id == receipt.proposal_id
                        )
                    )
                ).scalar_one_or_none()
                if saved_proposal:
                    saved_proposal["status"] = (
                        "completed" if receipt.status == "replayed" else receipt.status
                    )
                    await c.execute(
                        update(agent_proposals)
                        .where(agent_proposals.c.id == receipt.proposal_id)
                        .values(record=saved_proposal)
                    )
            await c.execute(
                insert(agent_events).values(
                    session_id=session.id,
                    kind=kind,
                    created_at=now(),
                    record=record or {"status": session.status, "message": session.message},
                )
            )

    async def events(self, identity, after):
        await self.get(identity)
        async with self.storage.engine.connect() as c:
            rows = (
                (
                    await c.execute(
                        select(agent_events)
                        .where(agent_events.c.session_id == identity, agent_events.c.id > after)
                        .order_by(agent_events.c.id)
                        .limit(500)
                    )
                )
                .mappings()
                .all()
            )
        return [
            Event(id=r["id"], kind=r["kind"], created_at=r["created_at"], record=r["record"])
            for r in rows
        ]

    async def reconcile(self):
        async with self.storage.engine.connect() as c:
            rows = (
                (
                    await c.execute(
                        select(agent_sessions.c.id).where(
                            agent_sessions.c.status.not_in(list(TERMINAL))
                        )
                    )
                )
                .scalars()
                .all()
            )
        for identity in rows:
            s, p = await self.load(identity)
            s.authority_epoch += 1
            s.status = "interrupted"
            s.message = (
                "Application restarted; authority revoked. Start a new session "
                "explicitly. No command was replayed."
            )
            receipt = None
            if s.pending_proposal:
                proposal = s.pending_proposal
                unknown = proposal.status in {"accepted", "started"}
                proposal.status = "unknown" if unknown else "stopped"
                receipt = Receipt(
                    id=p.get("execution_id", uuid4().hex) if unknown else uuid4().hex,
                    proposal_id=proposal.id,
                    status="unknown" if unknown else "stopped",
                    arguments=proposal.arguments,
                    applied_targets={},
                    pre_observation_id=proposal.observation_id,
                    post_observation_id=None,
                    elapsed_seconds=0.0,
                    reason="Restart interrupted this proposal; execution was not retried.",
                )
                s.receipts.append(receipt)
            p.pop("continuation", None)
            await self.save(s, p, "restart_interrupted", receipt=receipt)

    async def check_provider(self, identity):
        if self.closed:
            raise AgentError("Agent service is closing")
        profile = self.profiles.get(identity)
        task = asyncio.current_task()
        self.checks.add(task)
        try:
            result = await self.planner(
                profile,
                {
                    "operation": "check",
                    "max_output_tokens": 256,
                    "max_total_tokens": 4096,
                    "timeout_seconds": 30,
                },
            )
            caps = result.get("capabilities", {})
            if isinstance(caps, dict):
                caps = [k for k, v in caps.items() if v is True and k != "actuation"]
            if result.get("status") != "checked":
                raise AgentError("Provider capability check did not complete", 503)
            return self.profiles.checked(identity, profile["revision"], caps)
        except ValueError, OSError, TimeoutError:
            raise AgentError(
                (
                    "Provider capability check failed. No motion was requested; "
                    "billing may be unverified."
                ),
                503,
            ) from None
        finally:
            self.checks.discard(task)

    def fixture(self, s):
        raw = dict(
            target_session_id="replay-named-joint-v1",
            episode_id="synthetic-fixture",
            authority_epoch=s.authority_epoch,
            revision=0,
            step=0,
            clock_policy="replay",
            environment_kind="replay",
            embodiment_manifest_id="synthetic-six-joint-radians-v1",
            scene_id="synthetic-named-joint-fixture-v1",
            joints={
                j: 0.0
                for j in [
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                    "gripper",
                ]
            },
            units="radians",
            source="Built-in synthetic protocol fixture; no camera or robot execution.",
        )
        return raw

    async def create(self, project_id, payload: SessionCreate):
        async with self.creation_lock:
            if self.closed:
                raise AgentError("Agent service is closing")
            profile = self.profiles.get(payload.provider_id)
            if profile["provider"] != "mock" and (
                not profile.get("checked") or not payload.cloud_consent
            ):
                raise AgentError(
                    
                        "Check provider capabilities and explicitly consent to sending "
                        "task/state and any admitted image to OpenAI."
                    
                )
            if payload.target_kind == "teaching":
                active = await self.list(project_id)
                if any(
                    a.target_kind == "teaching"
                    and a.teaching_job_id == payload.teaching_job_id
                    and a.status not in TERMINAL
                    for a in active
                ):
                    raise AgentError("This Teaching target already has an active agent session")
            stamp = now()
            s = Session(
                id=uuid4().hex,
                project_id=project_id,
                provider_id=payload.provider_id,
                provider_revision=profile["revision"],
                task=payload.task,
                target_kind=payload.target_kind,
                teaching_job_id=payload.teaching_job_id,
                mode=payload.mode,
                clock_policy=payload.clock_policy,
                status="observing",
                authority_epoch=1,
                created_at=stamp,
                updated_at=stamp,
                max_calls=payload.max_calls,
                max_seconds=payload.max_seconds,
                max_output_tokens=payload.max_output_tokens,
                max_total_tokens=payload.max_total_tokens,
            )
            p = {
                "requests": {},
                "cloud_consent": payload.cloud_consent,
                "raw_observation": None,
                "executor_session_id": None,
            }
            await self.save(s, p, "created", new=True)
            task = asyncio.create_task(self.initialize(s, p, payload))
            self.tasks[s.id] = task
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done() and task.cancelled():
                return await self.get(s.id)
            # Request cancellation cannot leave an unowned initialization behind.
            await self.interrupt(s.id, "stop")
            raise

    async def initialize(self, s, p, payload):
        try:
            if payload.target_kind == "teaching":
                state = await self.execution.teaching.relay(
                    s.project_id, payload.teaching_job_id, "/state"
                )
                p["executor_session_id"] = state["session_id"]
                async with self.lock(s.id):
                    current, _ = await self.load(s.id)
                    if (
                        current.status != "observing"
                        or current.authority_epoch != s.authority_epoch
                    ):
                        raise AgentError("Session interrupted before target acquisition")
                    p["acquire_command_id"] = uuid4().hex
                    await self.save(
                        s, p, "acquisition_started", {"command_id": p["acquire_command_id"]}
                    )
                p["acquire_dispatched"] = True
                receipt = await self.target_command(
                    s,
                    p,
                    "acquire",
                    {
                        "clock_policy": payload.clock_policy,
                        "expected_revision": state["revision"],
                        "episode_id": state["episode_id"],
                    },
                    epoch=0,
                    command_id=p["acquire_command_id"],
                )
                p["executor_epoch"] = receipt["state"]["authority_epoch"]
                async with self.lock(s.id):
                    current, _ = await self.load(s.id)
                    if (
                        current.status != "observing"
                        or current.authority_epoch != s.authority_epoch
                    ):
                        raise AgentError("Session interrupted during target acquisition")
                    s.authority_epoch = p["executor_epoch"]
                    await self.save(s, p, "target_acquired")
                raw = (await self.target_command(s, p, "observe", {}))["observation"]
            elif payload.replay_observation_id:
                async with self.storage.engine.connect() as c:
                    row = (
                        await c.execute(
                            select(agent_observations.c.record).where(
                                agent_observations.c.id == payload.replay_observation_id,
                                agent_observations.c.project_id == s.project_id,
                            )
                        )
                    ).scalar_one_or_none()
                if row is None:
                    raise AgentError("Replay observation not found in this project", 404)
                raw = dict(row["raw"], replay_source_observation_id=payload.replay_observation_id)
            else:
                raw = self.fixture(s)
            async with self.lock(s.id):
                current, _ = await self.load(s.id)
                if current.status != "observing" or current.authority_epoch != s.authority_epoch:
                    raise AgentError("Session interrupted during observation capture")
                self.bind_observation(s, p, raw)
                s.status = "ready"
                s.message = (
                    "Observation retained. Replay records decisions without changing its source."
                    if s.target_kind == "replay"
                    else (
                        "Simulator ownership acquired; each action is admitted against "
                        "its original observation."
                    )
                )
                await self.save(s, p, "observed", observation=raw)
            return s
        except BaseException as error:
            # Even cancellation after dispatch reconciles the existing acquire ID.
            if p.get("acquire_dispatched") and not p.get("executor_epoch"):
                try:
                    async with asyncio.timeout(2):
                        while True:
                            receipt = await self.execution.teaching.agent_relay(
                                s.project_id,
                                s.teaching_job_id,
                                "/agent/executions/" + p["acquire_command_id"],
                            )
                            if receipt.get("status") not in {"queued", "executing"}:
                                break
                            await asyncio.sleep(0.025)
                        state = receipt.get("state", {})
                        if (
                            receipt.get("status") == "acknowledged"
                            and state.get("agent_owner") == s.id
                        ):
                            p["executor_epoch"] = state["authority_epoch"]
                except Exception:
                    p["acquisition_unknown"] = True
            async with self.lock(s.id):
                current, latest = await self.load(s.id)
                if current.status == "observing" and current.authority_epoch == s.authority_epoch:
                    current.authority_epoch += 1
                    current.status = (
                        "interrupted" if isinstance(error, asyncio.CancelledError) else "error"
                    )
                    current.message = (
                        "Target acquisition was interrupted. No motion was requested."
                        if isinstance(error, asyncio.CancelledError)
                        else (
                            "Target observation or ownership was unavailable. No motion was "
                            "requested."
                        )
                    )
                for key in (
                    "executor_session_id",
                    "executor_epoch",
                    "acquire_command_id",
                    "acquisition_unknown",
                ):
                    if p.get(key) is not None:
                        latest[key] = p[key]
                await self.save(current, latest, "acquisition_interrupted")
            await self.release_terminal(s.id)
            if isinstance(error, asyncio.CancelledError):
                raise
            current = await self.get(s.id)
            raise AgentError(current.message, 503) from None

    def bind_observation(self, s, private, raw):
        captured = now()
        ttl = (
            s.max_seconds if s.target_kind == "replay" else 30 if s.clock_policy == "stepped" else 2
        )
        age = private.get("observation_age_ms", raw.get("source_age_ms", 0))
        if not isinstance(age, (int, float)) or isinstance(age, bool) or age < 0:
            raise AgentError("Invalid executor observation age", 503)
        expires = (datetime.now(UTC) + timedelta(seconds=max(0, ttl - age / 1000))).isoformat()
        joints = raw.get("joints", {})
        if isinstance(joints, list):
            names, values = joints, raw.get("state_rad", [])
            if len(names) != len(values) or len(set(names)) != len(names):
                raise AgentError("Invalid observed joint dimensions", 503)
            joints = dict(zip(names, values, strict=True))
        if not joints or len(joints) > 64:
            raise AgentError("Target has no admitted joint manifest", 503)
        s.observation = Observation(
            id=uuid4().hex,
            sha256=digest(raw),
            target_session_id=str(raw["target_session_id"]),
            episode_id=raw.get("episode_id"),
            authority_epoch=s.authority_epoch,
            revision=raw.get("revision", 0),
            step=raw.get("step", 0),
            captured_at=captured,
            expires_at=expires,
            clock_policy=s.clock_policy,
            environment_kind="replay" if s.target_kind == "replay" else "simulation",
            embodiment_manifest_id=raw["embodiment_manifest_id"],
            scene_id=raw.get("scene_id", raw.get("scene_sha256", "")),
            joints=joints,
            image_sha256=raw.get("image_sha256"),
            source=raw.get(
                "source",
                "Immutable recorded observation"
                if s.target_kind == "replay"
                else "Simulator owner-thread capture",
            ),
        )
        private["raw_observation"] = raw

    async def target_command(self, s, p, operation, arguments, *, epoch=None, command_id=None):
        identity = command_id or uuid4().hex
        payload = {
            "command_id": identity,
            "session_id": p["executor_session_id"],
            "agent_session_id": s.id,
            "authority_epoch": s.authority_epoch if epoch is None else epoch,
            "arguments": arguments,
        }
        relay = self.execution.teaching.agent_relay
        async with asyncio.timeout(8):
            try:
                value = await relay(s.project_id, s.teaching_job_id, "/agent/" + operation, payload)
            except Exception:
                # A lost acknowledgement never causes a second POST. Reconcile
                # the exact persisted command identity through the read endpoint.
                value = await relay(
                    s.project_id, s.teaching_job_id, "/agent/executions/" + identity
                )
            while value.get("status") in {"queued", "executing"}:
                await asyncio.sleep(0.025)
                value = await relay(
                    s.project_id, s.teaching_job_id, "/agent/executions/" + identity
                )
        if value.get("status") != "acknowledged" or (
            operation == "execute" and value.get("execution", {}).get("status") != "completed"
        ):
            raise AgentError(
                "Executor rejected the original context or ownership; request a fresh observation."
            )
        if operation == "observe":
            # Keep the executor's hashed source immutable. Receipt age belongs in
            # host admission metadata and must not be added to the source hash.
            p["observation_age_ms"] = value.get("observation_age_ms", 0)
        return value

    def budget(self, s, p):
        if datetime.now(UTC) >= datetime.fromisoformat(s.created_at) + timedelta(
            seconds=s.max_seconds
        ):
            raise AgentError("Session wall-time budget expired. Start a new session.")
        if s.calls_used >= s.max_calls:
            raise AgentError("Provider call budget exhausted")

        # Reserve the exact serialized input estimate plus maximum output before dispatch.
        def input_bound(value):
            # Inline image pixels are sent through input_image detail=low, not as text.
            # Reserve 4096 tokens per image conservatively, and 1 token per UTF-8
            # byte for all text/metadata. The UI displays this as a reservation.
            if isinstance(value, dict):
                return sum(
                    4096
                    if k in {"image_data_url", "image_url"}
                    and isinstance(v, str)
                    and v.startswith("data:image/")
                    else len(k.encode()) + input_bound(v)
                    for k, v in value.items()
                )
            if isinstance(value, list):
                return sum(input_bound(v) for v in value)
            return len(canonical(value))

        estimated_input = (
            input_bound(p.get("continuation") or p["raw_observation"]) + len(s.task.encode()) + 2048
        )
        if (
            s.input_tokens + s.output_tokens + estimated_input + s.max_output_tokens
            > s.max_total_tokens
        ):
            raise AgentError(
                "Token reservation exceeds the remaining budget; no provider call started"
            )

    async def observe(self, identity):
        async with self.lock(identity):
            s, p = await self.load(identity)
            if s.status in TERMINAL or self.active(identity):
                raise AgentError("Stop active work or start a new session before observing")
            if s.target_kind == "teaching":
                raw = (await self.target_command(s, p, "observe", {}))["observation"]
            else:
                raw = p["raw_observation"]
            discarded = None
            if s.pending_proposal:
                proposal = s.pending_proposal
                proposal.status = "stopped"
                discarded = Receipt(
                    id=uuid4().hex,
                    proposal_id=proposal.id,
                    status="stopped",
                    arguments=proposal.arguments,
                    applied_targets={},
                    pre_observation_id=proposal.observation_id,
                    post_observation_id=None,
                    elapsed_seconds=0.0,
                    reason=(
                        "Fresh observation superseded this proposal; no execution was dispatched."
                    ),
                )
                s.receipts.append(discarded)
                async with self.storage.engine.begin() as c:
                    await c.execute(
                        update(agent_proposals)
                        .where(agent_proposals.c.id == proposal.id)
                        .values(record=proposal.model_dump())
                    )
            p.pop("continuation", None)
            s.pending_proposal = None
            self.bind_observation(s, p, raw)
            s.status = "ready"
            s.message = "Fresh admission context captured; previous proposals cannot be approved."
            await self.save(s, p, "observed", observation=raw, receipt=discarded)
            return s

    def active(self, identity):
        return identity in self.tasks and not self.tasks[identity].done()

    async def propose(self, identity, request):
        async with self.lock(identity):
            s, p = await self.load(identity)
            if request.request_id in p["requests"]:
                return s  # Durable same identity never starts another inference or motion.
            if s.status in TERMINAL or self.active(identity) or s.pending_proposal:
                raise AgentError("Session is terminal, busy or has an unresolved proposal")
            profile = self.profiles.get(s.provider_id)
            if profile["revision"] != s.provider_revision:
                raise AgentError("Provider profile changed. Start a new session.")
            self.budget(s, p)
            if s.observation is None or datetime.now(UTC) > datetime.fromisoformat(
                s.observation.expires_at
            ):
                raise AgentError("Observation expired. Capture a fresh observation.")
            p["requests"][request.request_id] = digest(
                {
                    "observation": s.observation.model_dump(),
                    "task": s.task,
                    "provider_revision": s.provider_revision,
                    "request_id": request.request_id,
                }
            )
            s.calls_used += 1
            s.status, s.message = (
                "planning",
                "Provider request in progress. Cancellation revokes this proposal context.",
            )
            await self.save(
                s,
                p,
                "planning",
                {"request_id": request.request_id, "observation_id": s.observation.id},
            )
            self.tasks[identity] = asyncio.create_task(self.plan(s, p, profile, request.request_id))
            return s

    async def plan(self, snapshot, private, profile, request_id):
        identity, epoch, observation_id = (
            snapshot.id,
            snapshot.authority_epoch,
            snapshot.observation.id,
        )
        try:
            remaining = (
                datetime.fromisoformat(snapshot.created_at)
                + timedelta(seconds=snapshot.max_seconds)
                - datetime.now(UTC)
            ).total_seconds()
            if remaining < 0.01:
                raise AgentError("Session wall-time budget expired before provider dispatch")
            payload = {
                "operation": "continue" if private.get("continuation") else "plan",
                "task": snapshot.task,
                "observation": dict(
                    private["raw_observation"], **snapshot.observation.model_dump()
                ),
                "joint_names": list(snapshot.observation.joints),
                "max_output_tokens": snapshot.max_output_tokens,
                "max_total_tokens": snapshot.max_total_tokens
                - snapshot.input_tokens
                - snapshot.output_tokens,
                "timeout_seconds": min(30, remaining),
            }
            if private.get("continuation"):
                payload.update(
                    continuation=private["continuation"], receipt=snapshot.receipts[-1].model_dump()
                )
            started = time.monotonic()
            result = await self.planner(profile, payload)
            elapsed = time.monotonic() - started
            async with self.lock(identity):
                s, p = await self.load(identity)
                if (
                    s.status != "planning"
                    or s.authority_epoch != epoch
                    or s.observation.id != observation_id
                    or self.profiles.get(s.provider_id)["revision"] != s.provider_revision
                ):
                    await self.save(s, p, "late_result_rejected", {"request_id": request_id})
                    return
                usage = result.get("usage", {})
                for field in ("input_tokens", "output_tokens"):
                    count = usage.get(field, 0)
                    if type(count) is not int or count < 0:
                        raise AgentError("Invalid provider usage", 503)
                    setattr(s, field, getattr(s, field) + count)
                p["continuation"] = result.get("continuation")
                if (
                    datetime.now(UTC)
                    >= datetime.fromisoformat(s.created_at) + timedelta(seconds=s.max_seconds)
                    or s.input_tokens + s.output_tokens > s.max_total_tokens
                ):
                    s.status = "error"
                    s.message = (
                        "Provider result arrived outside the session budget. No proposal "
                        "was admitted."
                    )
                    await self.save(
                        s,
                        p,
                        "budget_result_rejected",
                        {"request_id": request_id, "elapsed_seconds": elapsed, "usage": usage},
                    )
                    return
                if result.get("status") == "complete":
                    s.status = "complete"
                    s.message = (
                        "Provider turn complete. Task outcome has not been independently verified."
                    )
                    await self.save(
                        s,
                        p,
                        "provider_complete",
                        {"model": result.get("model"), "elapsed_seconds": elapsed, "usage": usage},
                    )
                    return
                args = Arguments.model_validate(result.get("arguments"))
                if args.joint not in s.observation.joints:
                    raise AgentError("Provider selected a joint outside the embodiment", 503)
                model = result.get("model")
                if not isinstance(model, str) or not model or len(model) > 192:
                    raise AgentError("Provider returned invalid model identity", 503)
                proposal = Proposal(
                    id=uuid5(NAMESPACE_URL, identity + ":" + request_id).hex,
                    request_id=request_id,
                    agent_session_id=identity,
                    observation_id=s.observation.id,
                    observation_sha256=s.observation.sha256,
                    authority_epoch=epoch,
                    expected_revision=s.observation.revision,
                    provider_revision=s.provider_revision,
                    model=model,
                    arguments=args,
                    status="proposed",
                    created_at=now(),
                    expires_at=s.observation.expires_at,
                )
                s.pending_proposal = proposal
                s.status = "waiting_for_acceptance"
                s.message = "Review this exact proposal and observation before accepting."
                await self.save(
                    s,
                    p,
                    "proposed",
                    {"proposal": proposal.model_dump(), "elapsed_seconds": elapsed, "usage": usage},
                    proposal=proposal,
                )
            if snapshot.mode == "bounded":
                await self.execute(
                    identity, Approval(proposal_id=proposal.id, authority_epoch=epoch)
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            async with self.lock(identity):
                s, p = await self.load(identity)
                if s.status in {"planning", "executing"} and s.authority_epoch == epoch:
                    s.status = "error"
                    s.message = (
                        "Planning or execution failed. No automatic retry was made; "
                        "inspect the preserved evidence."
                    )
                    await self.save(s, p, "failed")
        finally:
            await self.release_terminal(identity)

    async def release_terminal(self, identity):
        async with self.lock(identity):
            s, p = await self.load(identity)
            if (
                s.status not in TERMINAL
                or s.target_kind != "teaching"
                or p.get("released")
                or not p.get("executor_epoch")
            ):
                return
            try:
                await self.target_command(s, p, "release", {}, epoch=p["executor_epoch"])
                p["released"] = True
                await self.save(s, p, "target_released")
            except Exception:
                s.message += (
                    " Target release was not acknowledged; use the parent Teaching Stop control."
                )
                await self.save(s, p, "release_unknown")

    async def approve(self, identity, approval):
        # The execution task outlives the HTTP request, but never an explicit stop.
        async with self.lock(identity):
            s, p = await self.load(identity)
            known = next((r for r in s.receipts if r.proposal_id == approval.proposal_id), None)
            if known:
                if (
                    known.status in {"completed", "replayed"}
                    and approval.authority_epoch == s.authority_epoch
                ):
                    return s
                raise AgentError(
                    "This proposal was stopped, rejected or interrupted; it cannot be approved"
                )
            if self.active(identity):
                raise AgentError("Another agent operation is active")
            self.validate_approval(s, p, approval)
            self.tasks[identity] = asyncio.create_task(self.execute(identity, approval))
            return s

    def validate_approval(self, s, p, approval):
        proposal, observation = s.pending_proposal, s.observation
        if s.mode in {"advisory", "shadow"}:
            raise AgentError("This mode only logs proposals; it cannot execute them")
        if (
            s.status != "waiting_for_acceptance"
            or not proposal
            or proposal.id != approval.proposal_id
            or s.authority_epoch != approval.authority_epoch
            or proposal.authority_epoch != s.authority_epoch
            or proposal.status != "proposed"
        ):
            raise AgentError("Approval does not match the exact current proposal and epoch")
        if datetime.now(UTC) > datetime.fromisoformat(proposal.expires_at):
            raise AgentError("Proposal observation expired; capture a new observation")
        if (
            proposal.observation_id != observation.id
            or proposal.observation_sha256 != observation.sha256
            or proposal.expected_revision != observation.revision
            or digest(p["raw_observation"]) != observation.sha256
        ):
            raise AgentError("Observation binding changed; proposal withheld")
        if self.profiles.get(s.provider_id)["revision"] != proposal.provider_revision:
            raise AgentError("Provider profile changed; proposal withheld")
        if s.input_tokens + s.output_tokens > s.max_total_tokens or datetime.now(
            UTC
        ) > datetime.fromisoformat(s.created_at) + timedelta(seconds=s.max_seconds):
            raise AgentError("Session budget expired; proposal withheld")

    async def execute(self, identity, approval):
        dispatched, receipt = False, None
        started = time.monotonic()
        async with self.lock(identity):
            s, p = await self.load(identity)
            try:
                self.validate_approval(s, p, approval)
            except AgentError:
                if (
                    s.status == "waiting_for_acceptance"
                    and s.pending_proposal
                    and s.pending_proposal.id == approval.proposal_id
                ):
                    proposal = s.pending_proposal
                    proposal.status = "rejected"
                    receipt = Receipt(
                        id=uuid4().hex,
                        proposal_id=proposal.id,
                        status="rejected",
                        arguments=proposal.arguments,
                        applied_targets={},
                        pre_observation_id=proposal.observation_id,
                        post_observation_id=None,
                        elapsed_seconds=0.0,
                        reason="Proposal expired or its original context changed before dispatch.",
                    )
                    s.receipts.append(receipt)
                    s.status, s.message = "paused", receipt.reason
                    await self.save(s, p, "receipt", receipt.model_dump(), receipt=receipt)
                return
            proposal = s.pending_proposal
            proposal.status = "started"
            s.status = "executing"
            execution_id = uuid5(NAMESPACE_URL, proposal.id + ":execution").hex
            p["execution_id"] = execution_id
            await self.save(
                s,
                p,
                "execution_started",
                {"execution_id": execution_id, "proposal_id": proposal.id},
            )
        try:
            applied, evidence = {}, None
            if s.target_kind == "teaching":
                raw = p["raw_observation"]
                dispatched = True
                result = await self.target_command(
                    s,
                    p,
                    "execute",
                    {
                        "observation_id": raw["observation_id"],
                        "observation_sha256": raw["observation_sha256"],
                        "joint": proposal.arguments.joint,
                        "delta_rad": proposal.arguments.delta_rad,
                        "proposal_id": proposal.id,
                        "provider_model": proposal.model,
                    },
                    command_id=execution_id,
                )
                executed = result["execution"]
                names = list(s.observation.joints)
                applied = dict(zip(names, executed["applied_target_rad"], strict=True))
                elapsed_ns = executed["completed_monotonic_ns"] - executed["started_monotonic_ns"]
                evidence = ExecutionEvidence(
                    joint_names=names,
                    requested_targets=dict(
                        zip(names, executed["requested_target_rad"], strict=True)
                    ),
                    applied_targets=applied,
                    observed_next_state=dict(zip(names, executed["next_state_rad"], strict=True)),
                    joint_limit_clipped=executed["joint_limit_clipped"],
                    speed_limit_clipped=executed["speed_limit_clipped"],
                    execution_elapsed_seconds=max(0.0, elapsed_ns / 1e9),
                )
            async with self.lock(identity):
                current, p = await self.load(identity)
                if (
                    current.authority_epoch != approval.authority_epoch
                    or current.status != "executing"
                ):
                    return
                receipt = Receipt(
                    id=execution_id,
                    proposal_id=proposal.id,
                    status="completed" if dispatched else "replayed",
                    arguments=proposal.arguments,
                    applied_targets=applied,
                    pre_observation_id=current.observation.id,
                    post_observation_id=None,
                    elapsed_seconds=time.monotonic() - started,
                    executor_evidence=evidence,
                    reason=(
                        "Local simulator completed the bounded correction; task success "
                        "remains unverified."
                    )
                    if dispatched
                    else (
                        "Replay only: no actuator was dispatched and no applied action "
                        "label was generated."
                    ),
                )
                current.pending_proposal.status = "completed"
                current.receipts.append(receipt)
                current.pending_proposal = None
                current.status = "observing" if dispatched else "ready"
                current.message = receipt.reason
                # Completion is durable before a separate observation request.
                # Its loss cannot turn a known completed action into unknown.
                await self.save(current, p, "receipt", receipt.model_dump(), receipt=receipt)
            if dispatched:
                post_raw = (await self.target_command(current, p, "observe", {}))["observation"]
                post_age = p.get("observation_age_ms", 0)
                async with self.lock(identity):
                    current, p = await self.load(identity)
                    if (
                        current.authority_epoch != approval.authority_epoch
                        or current.status != "observing"
                    ):
                        return
                    p["observation_age_ms"] = post_age
                    self.bind_observation(current, p, post_raw)
                    receipt = current.receipts[-1]
                    receipt.post_observation_id = current.observation.id
                    receipt.executor_evidence.simulated_seconds = max(
                        0.0, post_raw["sim_time"] - raw["sim_time"]
                    )
                    current.status = "ready"
                    await self.save(
                        current,
                        p,
                        "post_execution_observed",
                        receipt.model_dump(),
                        observation=post_raw,
                        receipt=receipt,
                        receipt_update=True,
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            async with self.lock(identity):
                current, p = await self.load(identity)
                if current.authority_epoch == approval.authority_epoch and current.status in {
                    "executing",
                    "observing",
                }:
                    if current.status == "observing":
                        current.status = "error"
                        current.message = (
                            "Simulator correction completed, but the next observation was "
                            "unavailable. Its completed receipt is preserved; no retry was "
                            "made."
                        )
                        await self.save(current, p, "post_observation_failed")
                    else:
                        outcome = "unknown" if dispatched else "rejected"
                        receipt = Receipt(
                            id=execution_id,
                            proposal_id=proposal.id,
                            status=outcome,
                            arguments=proposal.arguments,
                            applied_targets={},
                            pre_observation_id=proposal.observation_id,
                            post_observation_id=None,
                            elapsed_seconds=time.monotonic() - started,
                            reason="Execution outcome could not be reconciled; it was not retried."
                            if dispatched
                            else "Proposal was rejected before dispatch.",
                        )
                        current.pending_proposal.status = outcome
                        current.receipts.append(receipt)
                        current.status, current.message = "error", receipt.reason
                        await self.save(
                            current, p, "receipt", receipt.model_dump(), receipt=receipt
                        )
        finally:
            await self.release_terminal(identity)

    async def interrupt(self, identity, operation):
        async with self.lock(identity):
            s, p = await self.load(identity)
            if s.status in TERMINAL:
                return s
            epoch = s.authority_epoch
            s.authority_epoch += 1
            receipt = None
            if s.pending_proposal:
                proposal = s.pending_proposal
                unknown = proposal.status == "started"
                proposal.status = "unknown" if unknown else "stopped"
                receipt = Receipt(
                    id=p.get("execution_id", uuid4().hex) if unknown else uuid4().hex,
                    proposal_id=proposal.id,
                    status="unknown" if unknown else "stopped",
                    arguments=proposal.arguments,
                    applied_targets={},
                    pre_observation_id=proposal.observation_id,
                    post_observation_id=None,
                    elapsed_seconds=0.0,
                    reason=(
                        "Authority revoked; in-flight execution must be reconciled and is "
                        "never automatically retried."
                    )
                    if unknown
                    else "Proposal invalidated before execution.",
                )
                s.receipts.append(receipt)
            s.pending_proposal = None
            s.status = "stopped" if operation == "stop" else "paused"
            s.message = (
                "Agent authority revoked. Simulator release is requested "
                "independently of provider cooperation."
            )
            p.pop("continuation", None)
            await self.save(s, p, operation, receipt=receipt)
            task = self.tasks.get(identity)
            if task and not task.done():
                task.cancel()
        if task:
            await asyncio.gather(task, return_exceptions=True)
        async with self.lock(identity):
            s, p = await self.load(identity)
            if (
                s.target_kind == "teaching"
                and not p.get("released")
                and not p.get("executor_epoch")
            ):
                if p.get("acquisition_unknown"):
                    s.status = "interrupted"
                    s.message = (
                        "Agent authority revoked; target acquisition outcome is unknown. "
                        "Use the parent Teaching Stop control."
                    )
                else:
                    p["released"] = True  # Initialization ended before an owner was acquired.
            if s.target_kind == "teaching" and not p.get("released") and p.get("executor_epoch"):
                try:
                    await self.target_command(
                        s, p, "release", {}, epoch=p.get("executor_epoch", epoch)
                    )
                    p["released"] = True
                except Exception:
                    s.status = "interrupted"
                    s.message = (
                        "Agent authority revoked; target release was not acknowledged. "
                        "Use the parent Teaching Stop control."
                    )
                    await self.save(s, p, "release_unknown")
            # A paused simulation no longer owns the target; explicit new session required.
            if s.target_kind == "teaching":
                if s.status == "paused":
                    s.status = "interrupted"
                await self.save(s, p, "target_released" if p.get("released") else "release_unknown")
        return await self.get(identity)

    async def close(self):
        self.closed = True
        async with self.storage.engine.connect() as c:
            identities = (
                (
                    await c.execute(
                        select(agent_sessions.c.id).where(
                            agent_sessions.c.status.not_in(list(TERMINAL))
                        )
                    )
                )
                .scalars()
                .all()
            )
        await asyncio.gather(
            *(self.interrupt(i, "stop") for i in identities), return_exceptions=True
        )
        # Terminal sessions can still be reconciling a local release. Drain those
        # owned tasks before the executor transports and database are closed.
        remaining = [task for task in self.tasks.values() if not task.done()]
        for task in remaining:
            task.cancel()
        await asyncio.gather(*remaining, return_exceptions=True)
        for task in list(self.checks):
            task.cancel()
        await asyncio.gather(*self.checks, return_exceptions=True)
