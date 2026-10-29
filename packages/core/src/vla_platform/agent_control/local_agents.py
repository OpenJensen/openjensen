"""Session-scoped external planners; execution remains in AgentControl."""

import asyncio
import hashlib
import hmac
import secrets
from datetime import UTC, datetime, timedelta

from .contracts import Assessment, SegmentArguments


def error(message, status=409):
    from .service import AgentError

    return AgentError(message, status)


def fingerprint(value):
    from .service import digest

    return digest(value)


class LocalAgents:
    def local_session(self, session):
        if session.provider_id != "local-mcp" or session.target_kind != "robot_lab":
            raise error("Local agent access requires a local MCP physical session")
        if not session.control_armed or session.status in {
            "stopped",
            "interrupted",
            "complete",
            "error",
        }:
            raise error("Local agent authority has ended", 403)

    def local_authorize(self, session, private, token):
        self.local_session(session)
        expected = private.get("local_token_sha256", "")
        if not isinstance(token, str) or not 32 <= len(token) <= 128 or not expected:
            raise error("Local agent access is missing or expired", 401)
        if not hmac.compare_digest(hashlib.sha256(token.encode()).hexdigest(), expected):
            raise error("Local agent access is missing or expired", 401)

    async def local_access(self, identity):
        async with self.lock(identity):
            s, p = await self.load(identity)
            self.local_session(s)
            if self.active(identity) or s.pending_proposal:
                raise error("Pause planning before replacing local agent access")
            token = secrets.token_urlsafe(32)
            p["local_token_sha256"] = hashlib.sha256(token.encode()).hexdigest()
            await self.save(s, p, "local_access_issued")
            return {
                "session_id": s.id,
                "access_token": token,
                "source_commit": self.local_source_commit,
            }

    async def local_revoke(self, identity):
        async with self.lock(identity):
            s, p = await self.load(identity)
            self.local_session(s)
            p.pop("local_token_sha256", None)
            await self.save(s, p, "local_access_revoked")
        return await self.pause_planning(identity, "local_agent_disconnected")

    async def local_context(self, identity, token):
        async with self.lock(identity):
            s, p = await self.load(identity)
            self.local_authorize(s, p, token)
            pending = self.local_pending.get(identity)
            return {
                "session": s.model_dump(),
                "planning_request": pending["context"] if pending else None,
                "action_space": self.action_space(s, p),
                "execution_policy": {
                    "human_approval_required": True,
                    "controller": "OpenJensen calibrated local position controller",
                    "gripper_units": "percent",
                    "executor_limits": p["raw_observation"].get("limits", {}),
                    "operator_heartbeat_owned_by": "OpenJensen operator page",
                },
            }

    async def local_image(self, identity, token, observation_id):
        async with self.lock(identity):
            s, p = await self.load(identity)
            self.local_authorize(s, p, token)
        data = await self.observation_image(identity, observation_id)
        async with self.lock(identity):
            s, p = await self.load(identity)
            self.local_authorize(s, p, token)
        return data

    async def local_stop(self, identity, token):
        async with self.lock(identity):
            s, p = await self.load(identity)
            self.local_authorize(s, p, token)
        return await self.interrupt(identity, "stop")

    async def local_submit(self, identity, token, request):
        async with self.lock(identity):
            s, p = await self.load(identity)
            self.local_authorize(s, p, token)
            submitted = p.get("local_submissions", {}).get(request.request_id)
            digest = fingerprint(request.model_dump())
            if submitted:
                if submitted != digest:
                    raise error("This planning request already has a different result")
                return s
            pending = self.local_pending.get(identity)
            if not pending or pending["future"].done() or s.status != "planning":
                raise error("There is no pending local planning request")
            context = pending["context"]
            if (
                any(
                    getattr(request, field) != context[field]
                    for field in (
                        "request_id",
                        "authority_epoch",
                        "observation_id",
                        "observation_sha256",
                    )
                )
                or s.authority_epoch != context["authority_epoch"]
            ):
                raise error("Local result does not match the pending observation and authority")
            if s.observation is None or s.observation.id != context["observation_id"]:
                raise error("The planning observation changed")
            if request.arguments is not None:
                args = SegmentArguments.model_validate(request.arguments.model_dump())
                if not set(args.targets) <= set(s.observation.joints):
                    raise error("Local agent selected an unknown joint", 422)
                result = {
                    "status": "proposed",
                    "skill": "joint_segment",
                    "arguments": args.model_dump(),
                    "purpose": request.purpose,
                    "model": request.agent_name,
                }
            else:
                result = {
                    "status": "assessed",
                    "assessment": Assessment.model_validate(
                        request.assessment.model_dump()
                    ).model_dump(),
                    "model": request.agent_name,
                }
            p.setdefault("local_submissions", {})[request.request_id] = digest
            await self.save(
                s,
                p,
                "local_result_received",
                {
                    "request_id": request.request_id,
                    "agent_name": request.agent_name,
                    "observation_id": request.observation_id,
                },
            )
            pending["future"].set_result(result)
            return s

    async def local_plan(self, snapshot, request_id, timeout_seconds):
        future = asyncio.get_running_loop().create_future()
        pending = {
            "future": future,
            "context": {
                "request_id": request_id,
                "authority_epoch": snapshot.authority_epoch,
                "observation_id": snapshot.observation.id,
                "observation_sha256": snapshot.observation.sha256,
                "deadline_at": (datetime.now(UTC) + timedelta(seconds=timeout_seconds)).isoformat(),
            },
        }
        async with self.lock(snapshot.id):
            s, p = await self.load(snapshot.id)
            self.local_session(s)
            if s.status != "planning" or s.authority_epoch != snapshot.authority_epoch:
                raise error("Local planning context changed")
            self.local_pending[s.id] = pending
            s.message = "Waiting for your local MCP agent to propose the next bounded action."
            await self.save(s, p, "awaiting_local_agent", pending["context"])
        try:
            async with asyncio.timeout(timeout_seconds):
                return await future
        finally:
            if self.local_pending.get(snapshot.id) is pending:
                self.local_pending.pop(snapshot.id, None)
