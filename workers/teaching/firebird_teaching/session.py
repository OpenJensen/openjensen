"""One-thread simulator executor. Network/voice callbacks never touch Isaac."""

from __future__ import annotations

import math
import threading
import time
import uuid

from sim_worker.rollout.contracts import Observation, Simulation
from sim_worker.rollout.experimental import MotionGuard

from .contracts import Command, Settings
from .frame_snapshot import ObservationSnapshot
from .journal import Journal


class Session:
    def __init__(self, settings: Settings, simulation: Simulation, journal: Journal):
        self.settings = settings
        self.sim = simulation
        self.journal = journal
        self.owner = threading.get_ident()
        self.mode = "idle"
        self.episode_id = None
        self.revision = 0
        self.instruction = ""
        self.outcome = "unknown"
        self.observation = None
        self.observation_received_monotonic_ns = None
        self.frame_snapshot = None
        self.target = None
        self.last_command = None
        self.target_command = None
        self.last_applied_command_id = None
        self.last_applied_step = None
        self.guard = None
        self.fault = None

    def _owned(self) -> None:
        if threading.get_ident() != self.owner:
            raise RuntimeError("Only the simulator thread may execute teaching operations")

    def _observe(self, step: int) -> tuple[Observation, int]:
        obs = self.sim.observe()
        spec = self.settings.sim
        if (
            obs.episode_id != self.episode_id
            or obs.step != step
            or not math.isfinite(obs.sim_time)
            or abs(obs.sim_time - step / spec.fps) > 1e-6
        ):
            raise ValueError("Simulator episode, step or clock differs from the capture contract")
        if len(obs.state) != len(spec.joints) or any(not math.isfinite(x) for x in obs.state):
            raise ValueError("Simulator state does not match the native joint order")
        if (obs.frame.width, obs.frame.height) != (spec.width, spec.height) or len(
            obs.frame.rgb
        ) != spec.width * spec.height * 3:
            raise ValueError("Simulator RGB frame differs from the capture contract")
        return obs, time.monotonic_ns()

    def _freeze_observation(self) -> None:
        self._owned()
        obs = self.observation
        if obs is None:
            self.frame_snapshot = None
            return
        self.frame_snapshot = ObservationSnapshot(
            self.journal.session_id,
            self.revision,
            self.episode_id,
            self.mode,
            obs.episode_id,
            obs.step,
            obs.sim_time,
            "observation.images.front",
            self.settings.sim.camera,
            self.observation_received_monotonic_ns,
            obs.frame.width,
            obs.frame.height,
            obs.frame.rgb,
            tuple(self.settings.sim.joints),
            tuple(obs.state),
        )

    def prepare(self) -> None:
        """Validate scene/camera/joints and capture idle preview before advertising readiness."""
        self._owned()
        if self.mode != "idle" or self.journal.current is not None:
            raise ValueError("Scene preparation requires an idle recorder")
        self.episode_id = "preview-" + uuid.uuid4().hex
        try:
            self.sim.reset(self.episode_id)
            self.observation, self.observation_received_monotonic_ns = self._observe(0)
        except BaseException as error:
            self.fail(error)
            raise
        finally:
            self.episode_id = None
        self._freeze_observation()

    def snapshot(self) -> dict:
        return {
            "mode": self.mode,
            "episode_id": self.episode_id,
            "revision": self.revision,
            "instruction": self.instruction,
            "outcome": self.outcome,
            "steps": self.observation.step if self.observation else 0,
            "sim_time": self.observation.sim_time if self.observation else 0,
            "state_rad": list(self.observation.state) if self.observation else None,
            "joints": list(self.settings.sim.joints),
            "fault": self.fault,
            "session_id": self.journal.session_id,
            "last_applied_command_id": self.last_applied_command_id,
            "last_applied_step": self.last_applied_step,
        }

    def execute(self, command: Command) -> dict:
        self._owned()
        if command.session_id != self.journal.session_id:
            raise ValueError("Executor session changed; refresh session state")
        if command.episode_id != self.episode_id or command.expected_revision != self.revision:
            raise ValueError("Stale episode or command revision; refresh session state")
        if self.mode in {"faulted", "closed"}:
            raise ValueError("Session faulted; preserve capture and start a new process")
        prior_observation = self.observation
        op = command.operation
        if self.journal.current is not None and len(self.journal.events) >= 512:
            error = RuntimeError("Episode intervention limit exceeded")
            self.fail(error)
            raise error
        if op == "task":
            self.instruction = command.arguments["instruction"]
        elif op == "start":
            if not self.instruction:
                raise ValueError("Set the demonstration task first")
            if self.mode == "running":
                raise ValueError("Demonstration is already running")
            if self.mode == "idle":
                self.episode_id = uuid.uuid4().hex
                try:
                    self.sim.reset(self.episode_id)
                    self.observation, self.observation_received_monotonic_ns = self._observe(0)
                    self.guard = MotionGuard(lambda: self.sim.joint_limits, self.settings.sim.fps)
                    self.journal.begin(self.episode_id, self.instruction)
                except BaseException as error:
                    self.fail(error)
                    raise
                self.outcome = "unknown"
            self.target = self.observation.state
            self.mode = "running"
        elif op == "pause":
            if self.mode == "idle":
                raise ValueError("No active demonstration to pause")
            self.mode = "paused"
            self.target = None
        elif op == "correct":
            if self.mode != "running":
                raise ValueError("Corrections require a running demonstration")
            joint = command.arguments["joint"]
            if joint not in self.settings.sim.joints:
                raise ValueError("Unknown simulator joint")
            values = list(self.observation.state)
            values[self.settings.sim.joints.index(joint)] += command.arguments["delta_rad"]
            self.target = tuple(values)
        elif op == "mark_failure":
            if self.mode == "idle":
                raise ValueError("No active demonstration to annotate")
            self.outcome = "operator_reported_failure"
        elif op in {"finish", "reset"}:
            if op == "finish" and self.mode == "idle":
                raise ValueError("No active demonstration to finish")
            self._event(command)
            try:
                self.journal.finish(op, self.outcome)
            except BaseException as error:
                self.fail(error)
                raise
            self.target = None
            self.mode = "idle"
            self.episode_id = None
            self.observation = None
            if op == "reset":
                try:
                    self.prepare()
                except BaseException as error:
                    self.fail(error)
                    raise
        self.revision += 1
        self.last_command = command.command_id
        if op in {"start", "correct"}:
            self.target_command = command.command_id
        if op not in {"finish", "reset"}:
            self._event(command)
        if self.observation is not prior_observation:
            self._freeze_observation()
        return self.snapshot()

    def _event(self, command: Command) -> None:
        try:
            self.journal.event(
                {
                    "command_id": command.command_id,
                    "operation": command.operation,
                    "arguments": command.arguments,
                    "source": "operator_command",
                    "step": self.observation.step if self.observation else 0,
                    "sim_time": self.observation.sim_time if self.observation else 0,
                    "accepted_monotonic_ns": time.monotonic_ns(),
                }
            )
        except BaseException as error:
            self.fail(error)
            raise

    def tick(self) -> None:
        self._owned()
        if self.mode != "running":
            return
        obs = self.observation
        try:
            if obs.step >= self.settings.max_steps:
                self.journal.finish("step_limit", self.outcome)
                self.target = None
                self.mode = "idle"
                self.episode_id = None
                self.observation = None
                self.frame_snapshot = None
                self.revision += 1
                return
            target = self.target if self.target is not None else obs.state
            limited = self.guard.constrain(target, obs.state)
            self.sim.apply(limited.target)
            self.sim.step()
            following, received_ns = self._observe(obs.step + 1)
            self.journal.append(
                {
                    "episode_id": self.episode_id,
                    "step": obs.step,
                    "sim_time": obs.sim_time,
                    "next_sim_time": following.sim_time,
                    "applied_monotonic_ns": time.monotonic_ns(),
                    "state_rad": obs.state,
                    "requested_target_rad": target,
                    "applied_target_rad": limited.target,
                    "next_state_rad": following.state,
                    "task": self.instruction,
                    "command_id": self.target_command,
                    "action_source": "operator_joint_position",
                    "joint_limit_clipped": limited.limit_clipped,
                    "speed_limit_clipped": limited.speed_clipped,
                },
                obs.frame.rgb,
            )
            self.observation = following
            self.observation_received_monotonic_ns = received_ns
            self._freeze_observation()
            self.last_applied_command_id = self.target_command
            self.last_applied_step = obs.step
        except BaseException as error:
            self.fail(error)
            raise

    def fail(self, error: BaseException) -> None:
        self._owned()
        self.mode = "faulted"
        self.fault = type(error).__name__ + ": " + str(error)
        self.target = None
        self.journal.abort()
        self.revision += 1

    def close(self) -> None:
        self._owned()
        if self.mode != "faulted":
            self.journal.finish("shutdown", self.outcome)
        self.mode = "closed"
        self.target = None
