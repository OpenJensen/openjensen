"""Owner-thread agent authority and observation-bound named-joint corrections.

This module controls only the simulator. It does not authorize physical motion.
"""

from __future__ import annotations

import base64
import hashlib
import math
import struct
import time
import uuid
import zlib
from dataclasses import dataclass

from .contracts import IDENTITY, Command, canonical, finite, integer, text

AGENT_OPERATIONS = frozenset({"acquire", "observe", "execute", "release", "heartbeat", "segment"})


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not IDENTITY.fullmatch(value):
        raise ValueError(f"Invalid {label}")
    return value


def png_data(rgb: bytes, width: int, height: int) -> bytes:
    def chunk(kind, content):
        return (
            struct.pack("!I", len(content))
            + kind
            + content
            + struct.pack("!I", zlib.crc32(kind + content) & 0xFFFFFFFF)
        )

    rows = b"".join(b"\0" + rgb[n : n + width * 3] for n in range(0, len(rgb), width * 3))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack("!2I5B", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


@dataclass(frozen=True)
class AgentCommand:
    command_id: str
    session_id: str
    agent_session_id: str
    authority_epoch: int
    operation: str
    arguments: dict

    @classmethod
    def parse(cls, operation: str, value: dict) -> AgentCommand:
        if operation not in AGENT_OPERATIONS or set(value) != {
            "command_id",
            "session_id",
            "agent_session_id",
            "authority_epoch",
            "arguments",
        }:
            raise ValueError("Unexpected or missing agent command fields")
        args = value["arguments"]
        required = {
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
        if (
            not isinstance(args, dict)
            or not required <= set(args)
            or set(args) - required - optional
        ):
            raise ValueError("Unexpected or missing agent arguments")
        args = dict(args)
        if operation == "acquire":
            if not isinstance(args["clock_policy"], str) or args["clock_policy"] not in {
                "stepped",
                "continuous",
            }:
                raise ValueError("Unsupported simulator clock policy")
            integer(args["expected_revision"], "expected revision", 0, 2**53 - 1)
            _identity(args["episode_id"], "episode ID")
            if "max_session_seconds" in args:
                integer(args["max_session_seconds"], "maximum session seconds", 1, 600)
            if "planning_snapshot_seconds" in args:
                integer(args["planning_snapshot_seconds"], "planning snapshot seconds", 1, 120)
                if args["clock_policy"] != "stepped":
                    raise ValueError("Extended planning snapshots require a frozen stepped scene")
        if operation in {"execute", "segment"}:
            _identity(args["observation_id"], "observation ID")
            _identity(args["proposal_id"], "proposal ID")
            digest = args["observation_sha256"]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError("Invalid observation fingerprint")
            args["provider_model"] = text(args["provider_model"], "provider model", 128)
            prompt_keys = {"prompt_version", "prompt_sha256"}
            if prompt_keys & set(args) and (
                not prompt_keys <= set(args)
                or not isinstance(args["prompt_version"], str)
                or not 1 <= len(args["prompt_version"]) <= 96
                or any(
                    c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
                    for c in args["prompt_version"]
                )
                or not isinstance(args["prompt_sha256"], str)
                or len(args["prompt_sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in args["prompt_sha256"])
            ):
                raise ValueError("Invalid prompt provenance")
            if operation == "execute":
                args["joint"] = text(args["joint"], "joint", 96)
                delta = finite(args["delta_rad"], "delta_rad")
                if not 0 < abs(delta) <= 0.15:
                    raise ValueError("Correction must be nonzero and at most 0.15 radians")
                args["delta_rad"] = delta
            else:
                targets = args["targets"]
                if not isinstance(targets, dict) or not 1 <= len(targets) <= 32:
                    raise ValueError("Segment requires bounded named targets")
                args["targets"] = {
                    text(k, "joint", 96): finite(v, "target_rad") for k, v in targets.items()
                }
                duration = finite(args["duration_seconds"], "duration_seconds")
                if not 0.1 <= duration <= 2:
                    raise ValueError("Simulator segment duration must be within 0.1..2 seconds")
                args["duration_seconds"] = duration
        return cls(
            _identity(value["command_id"], "command ID"),
            _identity(value["session_id"], "executor session ID"),
            _identity(value["agent_session_id"], "agent session ID"),
            integer(value["authority_epoch"], "authority epoch", 0, 2**53 - 1),
            operation,
            args,
        )


class AgentAuthority:
    """One local owner; only the simulator thread may mutate this state."""

    def __init__(self, session, *, clock_ns=time.monotonic_ns):
        self.session = session
        self.clock_ns = clock_ns
        self.epoch = 0
        self.owner = None
        self.clock_policy = None
        self.observations = {}
        self.pending = None
        self.step_permitted = False
        self.segment_remaining = 0
        self.lease_deadline_ns = 0
        self.arm_deadline_ns = 0
        self.planning_snapshot_seconds = 30

    def status(self) -> dict:
        return {
            "agent_owner": self.owner,
            "authority_epoch": self.epoch,
            "agent_clock_policy": self.clock_policy,
            "agent_capabilities": {
                "schema_version": 2,
                "named_joint_correction": True,
                "named_joint_segment": True,
                "epoch_heartbeat": True,
                "max_segment_ticks": 60,
                "max_segment_duration_ms": min(2000, 60000 // self.session.settings.sim.fps),
                "stepped_capture": True,
                "physical_execution": False,
            },
        }

    def revoke(self) -> None:
        self.session._owned()
        if self.owner is not None:
            self.epoch += 1
            self.owner = None
            self.observations.clear()
            if self.pending is not None:
                self.pending.update(
                    status="stopped",
                    stopped_monotonic_ns=self.clock_ns(),
                    task_success_verified=False,
                )
            self.pending = None
            self.step_permitted = False
            self.segment_remaining = 0
            self.clock_policy = None
            self.arm_deadline_ns = 0
            self.lease_deadline_ns = 0
            self.session.target = None
            if self.session.mode == "running":
                self.session.mode = "paused"
            self.session.target_actor = {
                "actor": "human_operator",
                "environment_kind": "simulation",
            }

    def can_tick(self) -> bool:
        if self.owner is not None and self.clock_ns() >= self.lease_deadline_ns:
            self.revoke()
            self.session.mode = "paused"
            self.session.revision += 1
            return False
        return self.owner is None or self.clock_policy != "stepped" or self.step_permitted

    def completed_step(self, record: dict) -> None:
        if self.pending is not None:
            ticks = self.pending.get("executed_ticks", 0) + 1
            segment = self.pending.get("skill") == "joint_segment"
            self.pending.update(
                {
                    "applied_target_rad": list(record["applied_target_rad"]),
                    "requested_target_rad": list(record["requested_target_rad"]),
                    "next_state_rad": list(record["next_state_rad"]),
                    "joint_limit_clipped": [
                        a or b
                        for a, b in zip(
                            self.pending.get(
                                "joint_limit_clipped", [False] * len(record["joint_limit_clipped"])
                            ),
                            record["joint_limit_clipped"],
                            strict=True,
                        )
                    ],
                    "speed_limit_clipped": [
                        a or b
                        for a, b in zip(
                            self.pending.get(
                                "speed_limit_clipped", [False] * len(record["speed_limit_clipped"])
                            ),
                            record["speed_limit_clipped"],
                            strict=True,
                        )
                    ],
                    "task_success_verified": False,
                }
            )
            if segment:
                self.segment_remaining -= 1
                self.pending.update(
                    executed_ticks=ticks,
                    simulation_elapsed_seconds=ticks / self.session.settings.sim.fps,
                )
            if not segment or self.segment_remaining == 0:
                self.pending.update(status="completed", completed_monotonic_ns=self.clock_ns())
                self.pending = None
        self.step_permitted = self.segment_remaining > 0
        # A segment owns its fixed target until the last local tick, then holds
        # the measured result; cloud reasoning never streams motor timing.
        if self.owner is not None and not self.step_permitted:
            self.session.target = tuple(record["next_state_rad"])

    def execute(self, command: AgentCommand) -> dict:
        session = self.session
        session._owned()
        if command.session_id != session.journal.session_id:
            raise ValueError("Executor session changed")
        if session.mode != "running" or session.episode_id is None:
            raise ValueError("Agent control requires an active running demonstration")
        if command.operation == "acquire":
            if self.owner is not None:
                raise ValueError("Simulator already has an agent controller owner")
            if command.authority_epoch != 0:
                raise ValueError("Acquire requires a fresh authority request")
            if (
                command.arguments["expected_revision"] != session.revision
                or command.arguments["episode_id"] != session.episode_id
            ):
                raise ValueError("Acquire command context changed before admission")
            self.epoch += 1
            self.owner = command.agent_session_id
            self.clock_policy = command.arguments["clock_policy"]
            self.planning_snapshot_seconds = command.arguments.get("planning_snapshot_seconds", 30)
            self.observations.clear()
            now = self.clock_ns()
            self.arm_deadline_ns = (
                now + command.arguments.get("max_session_seconds", 30) * 1_000_000_000
            )
            self.lease_deadline_ns = min(now + 30_000_000_000, self.arm_deadline_ns)
            session.target = session.observation.state
            return {"state": session.snapshot()}
        if (
            self.owner != command.agent_session_id
            or self.epoch != command.authority_epoch
            or self.clock_ns() >= self.lease_deadline_ns
        ):
            raise ValueError("Agent ownership lease or authority epoch is stale")
        if command.operation == "heartbeat":
            now = self.clock_ns()
            if now >= self.arm_deadline_ns:
                raise ValueError("Agent arming duration expired")
            self.lease_deadline_ns = min(now + 30_000_000_000, self.arm_deadline_ns)
            return {"state": session.snapshot()}
        if command.operation == "release":
            self.revoke()
            session.mode = "paused"
            session.revision += 1
            return {"state": session.snapshot()}
        if self.pending is not None:
            raise ValueError("An agent action is already in flight")
        if command.operation == "observe":
            # New pixels and measured state are acquired atomically by the owner,
            # including while stepped physics is frozen for model planning.
            session.observation, session.observation_received_monotonic_ns = session._observe(
                session.observation.step
            )
            session._freeze_observation()
            frame = session.frame_snapshot
            now = self.clock_ns()
            if now >= self.lease_deadline_ns or now >= self.arm_deadline_ns:
                self.revoke()
                raise ValueError("Agent authority expired during owner-thread capture")
            image = png_data(frame.rgb, frame.width, frame.height)
            manifest = {
                "scene_sha256": session.settings.scene_sha256,
                "joints": list(frame.joints),
                "units": "rad",
                "skill_manifest_version": 2,
            }
            envelope = {
                "schema_version": 1,
                "observation_id": uuid.uuid4().hex,
                "target_session_id": session.journal.session_id,
                "agent_session_id": self.owner,
                "episode_id": session.episode_id,
                "authority_epoch": self.epoch,
                "revision": session.revision,
                "embodiment_manifest_id": hashlib.sha256(canonical(manifest)).hexdigest(),
                "scene_sha256": session.settings.scene_sha256,
                "calibration_id": None,
                "clock_policy": self.clock_policy,
                "captured_monotonic_ns": now,
                "valid_until_monotonic_ns": min(
                    self.arm_deadline_ns,
                    now
                    + (
                        self.planning_snapshot_seconds * 1_000_000_000
                        if self.clock_policy == "stepped"
                        else 2_000_000_000
                    ),
                ),
                "scene_frozen_for_planning": self.clock_policy == "stepped",
                "snapshot_policy": "frozen_step_v1"
                if self.clock_policy == "stepped"
                else "continuous_age_drift_v1",
                "source_age_ms": 0,
                "step": frame.step,
                "sim_time": frame.sim_time,
                "joints": list(frame.joints),
                "state_rad": list(frame.state_rad),
                "joint_limits": {
                    name: list(bounds)
                    for name, bounds in zip(frame.joints, session.sim.joint_limits, strict=True)
                },
                "action_limits": {
                    "min_duration_seconds": 0.1,
                    "max_duration_seconds": min(2.0, 60 / session.settings.sim.fps),
                    "max_ticks": 60,
                },
                "units": "rad",
                "image_data_url": "data:image/png;base64,"
                + base64.b64encode(image).decode("ascii"),
                "image_sha256": hashlib.sha256(image).hexdigest(),
            }
            evidence = getattr(session.sim, "task_evidence", None)
            if callable(evidence):
                measured = evidence()
                if measured is not None:
                    envelope["task_evidence"] = measured
            envelope["observation_sha256"] = hashlib.sha256(canonical(envelope)).hexdigest()
            self.observations[envelope["observation_id"]] = envelope
            while len(self.observations) > 8:
                del self.observations[next(iter(self.observations))]
            return {"state": session.snapshot(), "observation": dict(envelope)}
        args = command.arguments
        observed = self.observations.get(args["observation_id"])
        if observed is None or args["observation_sha256"] != observed["observation_sha256"]:
            raise ValueError("Unknown or changed original observation")
        if self.clock_ns() >= observed["valid_until_monotonic_ns"]:
            raise ValueError("Agent observation expired before execution")
        if observed["episode_id"] != session.episode_id or observed["revision"] != session.revision:
            raise ValueError("Original observation command context changed")
        cached_state = session.observation.state
        if self.clock_policy == "stepped":
            if observed["step"] != session.observation.step:
                raise ValueError("Stepped scene changed during planning")
            # Recheck the native clock and measured state on the owner thread.
            # The original planning envelope retains its capture age and hash.
            session.observation, session.observation_received_monotonic_ns = session._observe(
                observed["step"]
            )
            session._freeze_observation()
            now = self.clock_ns()
            if now >= self.lease_deadline_ns or now >= observed["valid_until_monotonic_ns"]:
                raise ValueError("Original snapshot or authority expired during admission capture")
        if (
            max(
                abs(a - b)
                for state in (cached_state, session.observation.state)
                for a, b in zip(observed["state_rad"], state, strict=True)
            )
            > 0.03
        ):
            raise ValueError("Measured simulator state drift exceeds admission tolerance")
        values = list(session.observation.state)
        if command.operation == "segment":
            ticks = math.ceil(args["duration_seconds"] * session.settings.sim.fps)
            if not 1 <= ticks <= 60:
                raise ValueError("Simulator segment exceeds sixty local ticks")
            if not set(args["targets"]) <= set(session.settings.sim.joints):
                raise ValueError("Unknown simulator segment joint")
            for joint, target in args["targets"].items():
                index = session.settings.sim.joints.index(joint)
                if (
                    max(abs(target - values[index]), abs(target - observed["state_rad"][index]))
                    > 0.15 + 1e-12
                ):
                    raise ValueError("Segment target exceeds the 0.15 radian observation bound")
                values[index] = target
            self.segment_remaining = ticks
        else:
            if args["joint"] not in session.settings.sim.joints:
                raise ValueError("Unknown simulator joint")
            values[session.settings.sim.joints.index(args["joint"])] += args["delta_rad"]
            self.segment_remaining = 0
        session.target = tuple(values)
        session.revision += 1
        session.last_command = session.target_command = command.command_id
        session.target_actor = {
            "actor": "llm_agent",
            "environment_kind": "simulation",
            "agent_session_id": self.owner,
            "proposal_id": args["proposal_id"],
            "execution_id": command.command_id,
            "provider_model": args["provider_model"],
            "observation_id": args["observation_id"],
            "observation_sha256": args["observation_sha256"],
            "authority_epoch": self.epoch,
            "clock_policy": self.clock_policy,
            "skill_manifest_version": 2 if command.operation == "segment" else 1,
        }
        session.target_actor.update(
            {key: args[key] for key in ("prompt_version", "prompt_sha256") if key in args}
        )
        event = Command(
            command.command_id,
            command.session_id,
            session.episode_id,
            observed["revision"],
            "segment" if command.operation == "segment" else "correct",
            {"targets": args["targets"], "duration_seconds": args["duration_seconds"]}
            if command.operation == "segment"
            else {"joint": args["joint"], "delta_rad": args["delta_rad"]},
        )
        session._event(event, actor=session.target_actor)
        self.observations.clear()
        self.step_permitted = True
        self.pending = {
            "status": "started",
            "execution_id": command.command_id,
            "observation_id": args["observation_id"],
            "started_monotonic_ns": self.clock_ns(),
            **(
                {
                    "skill": "joint_segment",
                    "requested_duration_seconds": args["duration_seconds"],
                    "executed_ticks": 0,
                    "simulation_elapsed_seconds": 0.0,
                }
                if command.operation == "segment"
                else {}
            ),
        }
        return {"state": session.snapshot(), "execution": self.pending}
