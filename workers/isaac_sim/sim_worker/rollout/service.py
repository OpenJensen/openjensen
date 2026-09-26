import math
import time
from collections.abc import Callable
from dataclasses import replace

from sim_worker.rollout.calibration import CalibrationUse, JointMap
from sim_worker.rollout.config import RolloutSpec
from sim_worker.rollout.contracts import RGB_CHANNELS, Observation, Policy, Simulation
from sim_worker.rollout.experimental import MAX_SPEED_RAD_S, MotionGuard

_TIME_TOLERANCE_SECONDS = 1e-6


class Rollout:
    def __init__(
        self,
        spec: RolloutSpec,
        simulation: Simulation,
        policy: Policy,
        mapping: JointMap,
        guard: MotionGuard | None = None,
    ):
        if mapping.use is CalibrationUse.EXPERIMENTAL and guard is None:
            raise ValueError("Experimental calibration requires a motion guard")
        self._spec = spec
        self._simulation = simulation
        self._policy = policy
        self._mapping = mapping
        self._guard = guard

    def _observe(self, episode_id, step):
        observation = self._simulation.observe()
        sim = self._spec.sim
        if (
            observation.episode_id != episode_id
            or observation.step != step
            or not math.isfinite(observation.sim_time)
            or abs(observation.sim_time - step / sim.fps) > _TIME_TOLERANCE_SECONDS
        ):
            raise ValueError("Simulator observation has an incorrect episode or control timestamp")
        frame = observation.frame
        if (frame.width, frame.height) != (sim.width, sim.height) or len(
            frame.rgb
        ) != sim.width * sim.height * RGB_CHANNELS:
            raise ValueError("Simulator frame differs from capture configuration")
        state = self._mapping.to_policy(observation.state)
        return observation, replace(observation, state=state)

    def run(
        self,
        episode_id: str,
        record: Callable[[dict, Observation], None],
        finish: Callable[[Observation], None] | None = None,
    ) -> dict:
        self._simulation.reset(episode_id)
        self._policy.reset(episode_id)
        step = 0
        requests = 0
        latency_seconds = 0.0
        bounds_report = {"action_extrapolations": 0, "joint_limit_clips": 0, "speed_limit_clips": 0}
        observed, policy_observation = self._observe(episode_id, step)
        while step < self._spec.steps:
            started = time.monotonic()
            chunk = self._policy.predict(policy_observation)
            latency = time.monotonic() - started
            requests += 1
            latency_seconds += latency
            if (
                chunk.episode_id != episode_id
                or chunk.step != step
                or chunk.model_id != self._spec.model_id
                or not chunk.actions
            ):
                raise ValueError("Policy returned an incompatible action chunk")
            count = min(self._spec.execute_steps, len(chunk.actions), self._spec.steps - step)
            # Validate the whole selected prefix before any joint moves.
            targets = tuple(self._mapping.to_sim(action) for action in chunk.actions[:count])
            for index, target in enumerate(targets):
                bounds_trace = {}
                if self._guard is not None:
                    decision = self._guard.constrain(target, observed.state)
                    extrapolated = self._mapping.action_outside(chunk.actions[index])
                    bounds_trace = {
                        "raw_target_rad": target,
                        "action_extrapolated": extrapolated,
                        "joint_limit_clipped": decision.limit_clipped,
                        "speed_limit_clipped": decision.speed_clipped,
                    }
                    bounds_report["action_extrapolations"] += sum(extrapolated)
                    bounds_report["joint_limit_clips"] += sum(decision.limit_clipped)
                    bounds_report["speed_limit_clips"] += sum(decision.speed_clipped)
                    target = decision.target
                self._simulation.apply(target)
                self._simulation.step()
                measured, next_policy = self._observe(episode_id, step + 1)
                record(
                    {
                        "episode_id": episode_id,
                        "step": step,
                        "sim_time": observed.sim_time,
                        "state_rad": observed.state,
                        "policy_state": self._mapping.to_policy(observed.state),
                        "policy_action": chunk.actions[index],
                        "target_rad": target,
                        "next_state_rad": measured.state,
                        "next_sim_time": measured.sim_time,
                        "prediction_step": chunk.step,
                        "inference_seconds": latency,
                        **bounds_trace,
                    },
                    observed,
                )
                step += 1
                observed, policy_observation = measured, next_policy
        if finish is not None:
            finish(observed)
        result = {
            "status": "succeeded",
            "episode_id": episode_id,
            "steps": step,
            "model_id": self._spec.model_id,
            "calibration_sha256": self._mapping.digest,
            "calibration_status": self._mapping.status,
            "experimental": self._mapping.use is CalibrationUse.EXPERIMENTAL,
            "policy_requests": requests,
            "inference_seconds": latency_seconds,
            "sim_seconds": step / self._spec.sim.fps,
            "final_state_rad": observed.state,
        }
        if self._guard is not None:
            result.update(bounds_report | {"max_target_speed_rad_s": MAX_SPEED_RAD_S})
        return result
