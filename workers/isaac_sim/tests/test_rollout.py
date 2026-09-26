import json
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import yaml

from sim_worker.rollout.calibration import JointMap
from sim_worker.rollout.client import RemotePolicy
from sim_worker.rollout.config import RolloutSpec, load
from sim_worker.rollout.contracts import (
    API_VERSION,
    CALIBRATION_VERSION,
    ActionChunk,
    Frame,
    Observation,
    SimSpec,
)
from sim_worker.rollout.service import Rollout

_MODEL = "test-checkpoint"
_JOINTS = ("arm", "gripper")


def _calibration(path):
    data = {
        "api_version": CALIBRATION_VERSION,
        "status": "verified",
        "source": "test fixture",
        "joints": {
            "arm": {"sim_rad": [-2, 0, 2], "policy": [20, 0, -20]},
            "gripper": {"sim_rad": [0, 0.4, 1], "policy": [0, 60, 100]},
        },
    }
    path.write_text(yaml.safe_dump(data))
    return data


def _spec(path):
    return RolloutSpec(
        SimSpec("unused.usda", "/Camera", "/Robot", _JOINTS, 2, 2, 30, 120),
        3,
        1,
        "http://127.0.0.1:8080",
        _MODEL,
        "test",
        1,
        path,
    )


class _Simulation:
    def __init__(self):
        self.events = []
        self.index = 0
        self.observed = 0

    def reset(self, episode):
        self.events.append("reset")
        self.episode = episode
        self.index = 0
        self.state = (0.0, 0.4)

    def observe(self):
        self.observed += 1
        return Observation(
            self.episode,
            self.index,
            self.index / 30,
            self.state,
            Frame(2, 2, bytes([self.index] * 12)),
        )

    def apply(self, targets):
        self.events.append("apply")
        self.target = targets

    def step(self):
        self.events.append("step")
        self.state = self.target
        self.index += 1


class _Policy:
    def __init__(self, simulation):
        self.sim = simulation
        self.observations = []
        self.mode = "valid"

    def reset(self, episode):
        self.episode = episode

    def predict(self, observation):
        self.observations.append(observation)
        if self.sim.index != observation.step:
            raise AssertionError("Simulation advanced during inference")
        if self.mode == "timeout":
            raise TimeoutError("Policy disconnected")
        actions = ((-1.0, 50.0), (-2.0, 40.0))
        episode = observation.episode_id
        if self.mode == "stale":
            episode = "previous-episode"
        if self.mode == "invalid_tail":
            actions = (actions[0], (float("nan"), 40.0))
        return ActionChunk(episode, observation.step, _MODEL, actions)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "calibration.yaml"
        self.data = _calibration(self.path)

    def test_mapping_roundtrip(self):
        mapping = JointMap(self.path, _JOINTS)
        physical = (0.5, 0.7)
        policy = mapping.to_policy(physical)
        self.assertAlmostEqual(policy[0], -5)
        self.assertAlmostEqual(policy[1], 80)
        for actual, expected in zip(mapping.to_sim(policy), physical):
            self.assertAlmostEqual(actual, expected)

    def test_rejects_out_of_range(self):
        mapping = JointMap(self.path, _JOINTS)
        for action in ((21, 50), (0, 101), (0, float("nan")), (False, 50), (0,)):
            with self.subTest(action=action), self.assertRaises(ValueError):
                mapping.to_sim(action)

    def test_requires_verified_map(self):
        self.data["status"] = "unverified"
        self.path.write_text(yaml.safe_dump(self.data))
        with self.assertRaisesRegex(ValueError, "verified"):
            JointMap(self.path, _JOINTS)

    def test_requires_invertible_map(self):
        self.data["joints"]["gripper"]["policy"] = [0, 100, 50]
        self.path.write_text(yaml.safe_dump(self.data))
        with self.assertRaisesRegex(ValueError, "monotonic"):
            JointMap(self.path, _JOINTS)

    def test_rejects_wrong_joints(self):
        with self.assertRaises(ValueError):
            JointMap(self.path, ("other", "gripper"))


class RolloutTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "calibration.yaml"
        _calibration(self.path)
        self.spec = _spec(self.path)
        self.mapping = JointMap(self.path, _JOINTS)
        self.sim = _Simulation()
        self.policy = _Policy(self.sim)

    def test_replans_and_records(self):
        records = []
        result = Rollout(self.spec, self.sim, self.policy, self.mapping).run(
            "episode", lambda record, observation: records.append((record, observation))
        )
        self.assertEqual(result["steps"], 3)
        self.assertEqual([obs.step for obs in self.policy.observations], [0, 1, 2])
        self.assertEqual(
            self.sim.events, ["reset", "apply", "step", "apply", "step", "apply", "step"]
        )
        for actual, expected in zip(records[0][0]["target_rad"], (0.1, 1 / 3)):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(records[0][0]["next_state_rad"], self.sim.state)
        self.assertEqual(records[0][1].frame.rgb, bytes(12))
        self.assertAlmostEqual(result["sim_seconds"], 0.1)

    def test_renders_once_per_tick(self):
        Rollout(self.spec, self.sim, self.policy, self.mapping).run("episode", lambda *_: None)
        self.assertEqual(self.sim.observed, self.spec.steps + 1)

    def test_preserves_original_error(self):
        from sim_worker.rollout.app import execute

        class BrokenIsaac:
            def __enter__(self):
                return self

            def reset(self, episode_id):
                raise RuntimeError("SDK initialization failed")

            def __exit__(self, *args):
                raise SystemExit(1)

        output = Path(self.directory.name) / "output"
        with (
            patch("sim_worker.rollout.app.load", return_value=self.spec),
            patch("sim_worker.rollout.app.RemotePolicy"),
            patch("sim_worker.rollout.app.Trace"),
            patch("sim_worker.rollout.isaac.IsaacSim", return_value=BrokenIsaac()),
            self.assertLogs(level="ERROR"),
            self.assertRaises(SystemExit),
        ):
            execute(self.path, output, 1)
        result = json.loads((output / "result.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "SDK initialization failed")

    def test_failed_reply_stops_time(self):
        for mode, error in (("timeout", TimeoutError), ("stale", ValueError)):
            with self.subTest(mode=mode):
                self.policy.mode = mode
                self.sim.events.clear()
                with self.assertRaises(error):
                    Rollout(self.spec, self.sim, self.policy, self.mapping).run(
                        "episode", lambda *_: None
                    )
                self.assertEqual(self.sim.events, ["reset"])

    def test_checks_chunk_before_step(self):
        self.policy.mode = "invalid_tail"
        spec = replace(self.spec, execute_steps=2)
        with self.assertRaises(ValueError):
            Rollout(spec, self.sim, self.policy, self.mapping).run("episode", lambda *_: None)
        self.assertEqual(self.sim.events, ["reset"])

    def test_reset_clears_state(self):
        rollout = Rollout(self.spec, self.sim, self.policy, self.mapping)
        rollout.run("one", lambda *_: None)
        rollout.run("two", lambda *_: None)
        first = [obs for obs in self.policy.observations if obs.step == 0]
        self.assertEqual([obs.episode_id for obs in first], ["one", "two"])
        self.assertEqual(first[0].state, first[1].state)

    def test_http_hold_policy(self):
        from sim_worker.rollout.backend import MockPolicy
        from sim_worker.rollout.server import create_server

        server = create_server(("127.0.0.1", 0), MockPolicy(), _MODEL, state_dim=2)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            endpoint = f"http://127.0.0.1:{server.server_address[1]}"
            policy = RemotePolicy(endpoint, _MODEL, "hold", 2)
            policy.wait_ready(2)
            result = Rollout(self.spec, self.sim, policy, self.mapping).run("http", lambda *_: None)
            self.assertEqual(self.sim.state, (0.0, 0.4))
            self.assertEqual(result["policy_requests"], 3)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_http_wrong_model(self):
        from sim_worker.rollout.backend import MockPolicy
        from sim_worker.rollout.server import create_server

        server = create_server(("127.0.0.1", 0), MockPolicy(), "different-model", state_dim=2)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            policy = RemotePolicy(f"http://127.0.0.1:{server.server_address[1]}", _MODEL, "hold", 2)
            with self.assertRaisesRegex(ValueError, "identity"):
                policy.wait_ready(2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


class ConfigurationTests(unittest.TestCase):
    def test_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scene.usda").touch()
            _calibration(root / "calibration.yaml")
            data = {
                "api_version": API_VERSION,
                "scene": {
                    "uri": "scene.usda",
                    "camera": "/Camera",
                    "articulation": "/Robot",
                    "joints": list(_JOINTS),
                },
                "capture": {"width": 2, "height": 2},
                "control": {"fps": 30, "physics_hz": 120, "steps": 3, "execute_steps": 1},
                "policy": {
                    "endpoint": "http://127.0.0.1:8080",
                    "model_id": _MODEL,
                    "task": "hold",
                    "timeout_seconds": 2,
                },
                "calibration": "calibration.yaml",
            }
            path = root / "rollout.yaml"
            path.write_text(yaml.safe_dump(data))
            self.assertEqual(load(path).sim.scene, str(root.resolve() / "scene.usda"))
            data["control"]["physics_hz"] = 100
            path.write_text(yaml.safe_dump(data))
            with self.assertRaisesRegex(ValueError, "multiple"):
                load(path)

    def test_rejects_duplicate_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollout.yaml"
            path.write_text("api_version: first\napi_version: second\n")
            with self.assertRaisesRegex(ValueError, "unique"):
                load(path)


if __name__ == "__main__":
    unittest.main()
