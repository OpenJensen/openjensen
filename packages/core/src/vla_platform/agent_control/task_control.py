"""Goal-loop ownership, fresh physical re-admission and operator outcome review."""

import asyncio
import base64
import hashlib
import json
import time
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import select

from vla_platform.storage import agent_observations

from . import kinematics
from .contracts import (
    Observation,
    PlanRequest,
    PositionArguments,
    PositionEstimate,
    SegmentArguments,
    TaskReviewRecord,
)


def error(message, status=409):
    from .service import AgentError

    return AgentError(message, status)


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class TaskControl:
    def robot(self):
        if self.robot_target is None:
            if self.robot_bridge is None:
                raise error("The physical Robot Lab companion is not configured", 503)
            from .robot_target import RobotTarget

            self.robot_target = RobotTarget(self.robot_bridge)
        return self.robot_target

    def normalize_observation(self, s, p, raw, *, execution=False):
        physical = s.target_kind == "robot_lab"
        ttl = (
            1
            if physical
            else s.max_seconds
            if s.target_kind == "replay"
            else p.get("planning_snapshot_seconds", 30)
            if s.clock_policy == "stepped"
            else 2
        )
        if s.target_kind != "replay":
            begin, deadline = raw.get("captured_monotonic_ns"), raw.get("valid_until_monotonic_ns")
            if type(begin) is int and type(deadline) is int:
                ttl = min(ttl, max(0, (deadline - begin) / 1e9))
        age = p.get("observation_age_ms", raw.get("source_age_ms", 0))
        if type(age) not in {int, float} or not 0 <= age < 1e9:
            raise error("Invalid executor observation age", 503)
        joints = raw.get("joints", {})
        if isinstance(joints, list):
            values = raw.get("state_rad", [])
            if len(joints) != len(values) or len(set(joints)) != len(joints):
                raise error("Invalid observed joint dimensions", 503)
            joints = dict(zip(joints, values, strict=True))
        if not isinstance(joints, dict) or not 1 <= len(joints) <= 64:
            raise error("Target has no admitted joint manifest", 503)
        units = raw.get("units") if physical else {j: "rad" for j in joints}
        if not isinstance(units, dict) or set(units) != set(joints):
            raise error("Observation units do not match the embodiment", 503)
        stamp = datetime.now(UTC)
        obs = Observation(
            id=uuid4().hex,
            sha256=fingerprint(raw),
            target_session_id=str(raw["target_session_id"]),
            episode_id=raw.get("episode_id"),
            authority_epoch=s.authority_epoch,
            revision=raw.get("revision", 0),
            step=raw.get("step", 0),
            captured_at=stamp.isoformat(),
            expires_at=(stamp + timedelta(seconds=max(0, ttl - age / 1000))).isoformat(),
            clock_policy=s.clock_policy,
            environment_kind="physical"
            if physical
            else "replay"
            if s.target_kind == "replay"
            else "simulation",
            embodiment_manifest_id=raw["embodiment_manifest_id"],
            scene_id=raw.get("scene_id", raw.get("scene_sha256", raw.get("robot_id", ""))),
            joints=joints,
            units="mixed" if physical else "radians",
            joint_units=units,
            calibration_sha256=raw.get("calibration_sha256"),
            task_evidence=raw.get("task_evidence"),
            image_sha256=raw.get("image_sha256"),
            source=raw.get(
                "source",
                "Fresh companion read interval; camera sensor timestamp is unverified"
                if physical
                else "Immutable recorded observation"
                if s.target_kind == "replay"
                else "Simulator owner-thread capture",
            ),
        )
        if physical and s.gripper_position_control:
            obs.gripper_position_estimate = PositionEstimate.model_validate(
                {
                    "model": kinematics.MODEL,
                    "frame": kinematics.FRAME,
                    "units": "metres",
                    "position_m": kinematics.forward(joints),
                    "evidence": "model_estimate",
                    "orientation_controlled": False,
                }
            )
        if execution:
            s.execution_observation = obs
            p["execution_raw_observation"] = raw
        else:
            s.observation = obs
            p["raw_observation"] = raw
        return obs

    def action_space(self, s, p):
        physical = s.target_kind == "robot_lab"
        raw, obs = p["raw_observation"], s.observation
        limits = {}
        for joint in obs.joints:
            bounds = raw.get("joint_limits", {}).get(joint)
            if isinstance(bounds, dict):
                lower, upper = bounds.get("min"), bounds.get("max")
                if lower is None:
                    lower = bounds.get("minimum")
                if upper is None:
                    upper = bounds.get("maximum")
                bounds = [lower, upper]
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                if physical:
                    raise error("Physical calibration limits are missing", 503)
                bounds = [-3.141592653589793, 3.141592653589793]
            limits[joint] = list(bounds)
        units = obs.joint_units or {j: "rad" for j in obs.joints}
        result = {
            "kind": "physical" if physical else "simulation",
            "joint_units": units,
            "joint_limits": limits,
            "max_delta": {
                j: 8.0 if units[j] == "percent" else 4.0 if physical else 0.15 for j in obs.joints
            },
            "min_duration_seconds": 0.5 if physical else 0.2,
            "max_duration_seconds": 5.0 if physical else 2.0,
        }
        if not physical and raw.get("action_limits"):
            local = raw["action_limits"]
            maximum = min(2.0, local["max_duration_seconds"])
            result["max_duration_seconds"] = maximum
            result["min_duration_seconds"] = min(maximum, max(0.2, local["min_duration_seconds"]))
        if physical:
            result["trajectory_profile"] = {
                "kind": "quintic-position",
                "max_speed_units_per_second": 10.0,
                "max_acceleration_units_per_second2": 20.0,
            }
        if physical and s.gripper_position_control:
            result["gripper_position"] = {
                "model": kinematics.MODEL,
                "frame": kinematics.FRAME,
                "units": "metres",
                "current_m": kinematics.forward(obs.joints),
                "max_step_m": kinematics.MAX_STEP_M,
                "tolerance_m": kinematics.TOLERANCE_M,
                "orientation_controlled": False,
                "camera_to_base_calibrated": False,
                "evidence": "model_estimate",
            }
        return result

    def position_proposal(self, s, p, arguments):
        if s.target_kind != "robot_lab" or not s.gripper_position_control:
            raise error(
                "Enable standard SO101 gripper position targets when arming this session", 422
            )
        if any(s.observation.joint_units.get(j) != "degrees" for j in kinematics.JOINTS):
            raise error("Position targets need calibrated SO101 joint angles in degrees", 422)
        try:
            args = PositionArguments.model_validate(arguments)
            space = self.action_space(s, p)
            segment, plan = kinematics.waypoint(
                [args.x_m, args.y_m, args.z_m],
                s.observation.joints,
                space["joint_limits"],
                space["max_delta"],
                args.duration_seconds,
            )
            return SegmentArguments.model_validate(segment), plan
        except (ValueError, KeyError) as problem:
            raise error(str(problem), 422) from None

    async def capture_target(self, s, p):
        if s.target_kind == "robot_lab":
            raw = await self.robot().observe(s.id, p["executor_epoch"])
            ages = getattr(self.robot(), "observation_age_ms", {})
            p["observation_age_ms"] = ages.get(
                (s.id, p["executor_epoch"]), raw.get("source_age_ms", 0)
            )
            return raw
        if s.target_kind == "teaching":
            return (await self.target_command(s, p, "observe", {}))["observation"]
        return p["raw_observation"]

    async def release_target(self, s, p):
        if s.target_kind == "robot_lab":
            await self.robot().release(s.id, p["executor_epoch"])
        elif s.target_kind == "teaching":
            await self.target_command(s, p, "release", {}, epoch=p["executor_epoch"])

    def begin_supervision(self, s):
        if s.target_kind == "replay":
            return
        self.operator_heartbeats[(s.id, s.authority_epoch)] = time.monotonic()
        if s.id not in self.supervisors or self.supervisors[s.id].done():
            self.supervisors[s.id] = asyncio.create_task(self.supervise(s.id))

    async def supervise(self, identity):
        try:
            first = True
            while not self.closed:
                s, p = await self.load(identity)
                if not s.control_armed or s.status in {
                    "stopped",
                    "interrupted",
                    "complete",
                    "error",
                }:
                    return
                if datetime.now(UTC) >= datetime.fromisoformat(s.created_at) + timedelta(
                    seconds=s.max_seconds
                ):
                    await self.interrupt(identity, "wall_budget_expired")
                    return
                if s.target_kind == "robot_lab":
                    heartbeat = self.operator_heartbeats.get((identity, s.authority_epoch), 0)
                    if time.monotonic() - heartbeat > 1.5:
                        await self.interrupt(identity, "operator_lease_expired")
                        return
                    await self.robot().heartbeat(identity, p["executor_epoch"])
                    await asyncio.sleep(0.4)
                else:
                    # Provider work never supplies the local controller lease.
                    # The executor also enforces its fixed maximum arm duration.
                    if not first:
                        await self.target_command(s, p, "heartbeat", {})
                    first = False
                    remaining = (
                        datetime.fromisoformat(s.created_at)
                        + timedelta(seconds=s.max_seconds)
                        - datetime.now(UTC)
                    ).total_seconds()
                    await asyncio.sleep(max(0.05, min(5, remaining)))
        except asyncio.CancelledError:
            raise
        except Exception:
            if not self.closed:
                await self.interrupt(identity, "executor_lease_failed")

    async def heartbeat(self, identity, payload):
        s, _ = await self.load(identity)
        if (
            self.closed
            or s.target_kind != "robot_lab"
            or not s.control_armed
            or payload.authority_epoch != s.authority_epoch
            or s.status in {"stopped", "interrupted", "complete", "error"}
        ):
            raise error("The physical operator lease is not current")
        self.operator_heartbeats[(identity, s.authority_epoch)] = time.monotonic()
        return s

    async def run_task(self, identity):
        async with self.lock(identity):
            s, p = await self.load(identity)
            if self.closed or s.status in {"stopped", "interrupted", "complete", "error"}:
                raise error("Start a new agent session before running a task")
            if s.target_kind == "robot_lab" and (s.mode != "supervised" or not s.control_armed):
                raise error("Physical task execution requires an armed supervised session")
            if s.mode in {"advisory", "shadow"}:
                raise error("Advice and shadow modes log one proposal at a time")
            if s.pending_proposal and s.pending_proposal.status != "proposed":
                raise error("Capture a fresh observation before resuming a rejected proposal")
            s.run_enabled = True
            p["planning_paused"] = False
            s.message = (
                "Task loop enabled. Every proposal retains the original "
                "observation and local authority."
            )
            if s.status == "paused" and not s.pending_proposal:
                s.status = "ready"
            await self.save(s, p, "task_run_started")
            if identity not in self.runners or self.runners[identity].done():
                self.runners[identity] = asyncio.create_task(self.run_loop(identity))
            return s

    async def pause_run(self, identity, message):
        async with self.lock(identity):
            s, p = await self.load(identity)
            s.run_enabled = False
            if s.status not in {"stopped", "interrupted", "complete", "error"}:
                s.status = "paused"
            s.message = message
            await self.save(s, p, "task_run_paused")

    async def run_loop(self, identity):
        try:
            while not self.closed:
                s, p = await self.load(identity)
                if not s.run_enabled or s.status in {"stopped", "interrupted", "complete", "error"}:
                    return
                if datetime.now(UTC) >= datetime.fromisoformat(s.created_at) + timedelta(
                    seconds=s.max_seconds
                ):
                    await self.pause_run(identity, "The task wall-time budget expired.")
                    return
                # An already reserved final call/action is allowed to finish.
                # Budgets gate the NEXT planning boundary, never its receipt.
                if s.status == "ready" and not self.active(identity):
                    if s.step_count >= s.max_steps or s.calls_used >= s.max_calls:
                        await self.pause_run(
                            identity,
                            (
                                "The task step or provider-call budget is exhausted. "
                                "Review its trajectory before starting another session."
                            ),
                        )
                        return
                    expired = s.observation is None or datetime.now(UTC) >= datetime.fromisoformat(
                        s.observation.expires_at
                    )
                    if s.target_kind != "replay" and expired:
                        await self.observe(identity)
                    await self.propose(identity, PlanRequest(request_id=uuid4().hex))
                await asyncio.sleep(0.075)
        except asyncio.CancelledError:
            raise
        except Exception:
            await self.pause_run(
                identity,
                (
                    "Task planning or fresh observation was unavailable. No automatic "
                    "retry was dispatched; inspect the preserved evidence."
                ),
            )

    async def execution_observe(self, identity):
        async with self.lock(identity):
            s, p = await self.load(identity)
            if (
                s.target_kind != "robot_lab"
                or not s.control_armed
                or s.status != "waiting_for_acceptance"
                or not s.pending_proposal
                or self.active(identity)
            ):
                raise error("A current physical proposal is required for re-admission")
            epoch, proposal_id = s.authority_epoch, s.pending_proposal.id
        raw = await self.capture_target(s, p)
        age = p.get("observation_age_ms", 0)
        async with self.lock(identity):
            s, p = await self.load(identity)
            if (
                s.authority_epoch != epoch
                or s.status != "waiting_for_acceptance"
                or not s.pending_proposal
                or s.pending_proposal.id != proposal_id
            ):
                raise error("Physical proposal changed while observing")
            p["observation_age_ms"] = age
            self.normalize_observation(s, p, raw, execution=True)
            # Preview refresh replaces only the current unapproved read. The
            # exact accepted snapshot becomes immutable in approve().
            await self.save(
                s,
                p,
                "execution_preview",
                {"observation_id": s.execution_observation.id, "proposal_id": proposal_id},
            )
            return s

    def validate_physical_admission(self, s, p, approval):
        admitted = p.get("approved_admission")
        frozen = admitted and admitted.get("proposal_id") == approval.proposal_id
        observed = (
            Observation.model_validate(admitted["public"]) if frozen else s.execution_observation
        )
        raw = admitted["raw"] if frozen else p.get("execution_raw_observation")
        if (
            not approval.scene_unchanged_confirmed
            or not observed
            or approval.execution_observation_id != observed.id
            or approval.execution_observation_sha256 != observed.sha256
        ):
            raise error("Review and confirm the exact fresh physical observation")
        if (
            datetime.now(UTC) > datetime.fromisoformat(observed.expires_at)
            or observed.authority_epoch != s.authority_epoch
            or fingerprint(raw) != observed.sha256
        ):
            raise error("Physical re-admission observation expired or changed")
        original = s.observation
        if (
            observed.calibration_sha256 != original.calibration_sha256
            or observed.embodiment_manifest_id != original.embodiment_manifest_id
            or observed.joint_units != original.joint_units
            or set(observed.joints) != set(original.joints)
        ):
            raise error("Physical embodiment, calibration or units changed")
        if any(
            abs(observed.joints[j] - original.joints[j])
            > (2 if observed.joint_units[j] == "percent" else 0.5)
            for j in observed.joints
        ):
            raise error("Physical state drifted from the model observation; request a new proposal")
        if time.monotonic() - self.operator_heartbeats.get((s.id, s.authority_epoch), 0) > 1.5:
            raise error("Operator presence lease expired")

    async def observation_image(self, identity, observation_id):
        s, p = await self.load(identity)
        if s.execution_observation and observation_id == s.execution_observation.id:
            raw = p.get("execution_raw_observation", {})
        else:
            async with self.storage.engine.connect() as c:
                record = (
                    await c.execute(
                        select(agent_observations.c.record).where(
                            agent_observations.c.id == observation_id,
                            agent_observations.c.session_id == identity,
                        )
                    )
                ).scalar_one_or_none()
            if record is None:
                raise error("Observation image not found", 404)
            raw = record["raw"]
        value = raw.get("image_data_url", "")
        if not isinstance(value, str) or not value.startswith(
            ("data:image/png;base64,", "data:image/jpeg;base64,")
        ):
            raise error("This observation has no admitted image", 404)
        try:
            data = base64.b64decode(value.split(",", 1)[1], validate=True)
        except ValueError:
            raise error("Admitted image is invalid", 503) from None
        if len(data) > 8 * 1024**2 or hashlib.sha256(data).hexdigest() != raw.get("image_sha256"):
            raise error("Admitted image identity changed", 503)
        return data, "image/png" if value.startswith("data:image/png") else "image/jpeg"

    async def review_task(self, identity, payload):
        async with self.lock(identity):
            s, p = await self.load(identity)
            observation = s.execution_observation or s.observation
            if self.active(identity):
                raise error(
                    "Wait for current planning or execution before reviewing a task outcome"
                )
            if (
                payload.authority_epoch != s.authority_epoch
                or not observation
                or payload.observation_id != observation.id
            ):
                raise error("Task review must name the exact current observation and epoch")
            s.task_review = TaskReviewRecord(
                **payload.model_dump(), reviewed_at=datetime.now(UTC).isoformat()
            )
            s.task_success_verified = payload.outcome == "success"
            if payload.outcome in {"success", "failure"}:
                s.run_enabled = False
            await self.save(s, p, "operator_task_review", s.task_review.model_dump())
            return s

    async def pause_planning(self, identity, operation):
        async with self.lock(identity):
            s, p = await self.load(identity)
            if s.status in {"stopped", "interrupted", "complete", "error"}:
                return s
            s.run_enabled = False
            p["planning_paused"] = True
            receipt = None
            executing = s.status in {"executing", "observing"} and self.active(identity)
            if s.pending_proposal and not executing:
                from .contracts import Receipt

                proposal = s.pending_proposal
                proposal.status = "stopped"
                receipt = Receipt(
                    id=uuid4().hex,
                    proposal_id=proposal.id,
                    status="stopped",
                    arguments=proposal.arguments,
                    applied_targets={},
                    pre_observation_id=proposal.observation_id,
                    post_observation_id=None,
                    elapsed_seconds=0.0,
                    reason=(
                        "Planning paused before dispatch; the controller continues "
                        "its supervised local hold."
                    ),
                )
                s.receipts.append(receipt)
                s.pending_proposal = None
                s.execution_observation = None
            if not executing:
                s.status = "paused"
            s.message = (
                (
                    "Planning paused. The current bounded action finishes locally, then holds; "
                    "operator heartbeat and Stop remain active."
                )
                if executing
                else (
                    "Planning paused. Local controlled hold remains armed while operator "
                    "presence and the session budget are current."
                )
            )
            p.pop("continuation", None)
            await self.save(s, p, operation, receipt=receipt)
            owned = [self.runners.get(identity)]
            if not executing:
                owned.append(self.tasks.get(identity))
            owned = [
                task
                for task in owned
                if task and not task.done() and task is not asyncio.current_task()
            ]
            for task in owned:
                task.cancel()
        if owned:
            await asyncio.wait(owned, timeout=0.025)
        return await self.get(identity)
