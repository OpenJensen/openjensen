"""ACT compatibility, bounded control and Stop races on fake hardware only."""

import copy
import json
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_robot_lab as fixtures
from hardware import JOINTS, HardwareError
from policy import MAX_ACCELERATION, MAX_SPEED, UNITS, admit


class Runtime:
    def __init__(self, *args):
        self.info = {"device": "cpu", "camera_key": "observation.images.front"}
        self.closed = False
        self.action = [0.0] * 5 + [50.0]

    def request(self, message, timeout=2):
        return {"reset": True} if message.get("reset") else {"action": self.action}

    def close(self):
        self.closed = True


class Camera(fixtures.FakeCamera):
    def __init__(self):
        self.stamp = None

    def start_reader(self, index):
        pass

    def latest_timed_jpeg(self, index, max_age=0.15):
        return {"jpeg": b"test-observation", "captured_monotonic": self.stamp or time.monotonic()}


class PolicyTests(unittest.TestCase):
    setUp = fixtures.SafetyTests.setUp
    tearDown = fixtures.SafetyTests.tearDown
    start = fixtures.SafetyTests.start

    def install(self):
        runner = self.lab.policy
        folder = runner.root / "pen"
        checkpoint = folder / "pretrained_model"
        checkpoint.mkdir(parents=True)
        manifest = {
            "schema_version": 1,
            "name": "Pen ACT",
            "policy_type": "act",
            "fps": 13,
            "joint_names": list(JOINTS),
            "camera_key": "observation.images.front",
            "state_units": UNITS,
            "action_units": UNITS,
            "checkpoint": "pretrained_model",
        }
        (folder / "run_policy.json").write_text(json.dumps(manifest))
        config = {
            "type": "act",
            "input_features": {
                "observation.state": {"shape": [6]},
                "observation.images.front": {"shape": [3, 480, 640]},
            },
            "output_features": {"action": {"shape": [6]}},
        }
        (checkpoint / "config.json").write_text(json.dumps(config))
        for name in ("model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
            (checkpoint / name).write_text("{}")
        settings = self.lab.data_dir / "policy-runtime.json"
        settings.write_text(json.dumps({"python": sys.executable, "device": "cpu"}))
        settings.chmod(0o600)
        runner.process_factory = Runtime
        return runner, folder, checkpoint

    def prepare(self):
        runner, folder, checkpoint = self.install()
        runner.load({"policy_id": "pen"})
        runner.check = {"passed": True, "camera_index": 0, "checked_at": time.monotonic()}
        self.start()
        self.lab.camera = Camera()
        return runner

    def run_payload(self):
        return {
            "confirm": "RUN POLICY",
            "operator_present": True,
            "workspace_clear": True,
            "camera_index": 0,
            "duration_seconds": 10,
            "expected_generation": self.lab.generation,
        }

    def test_loading_and_catalog_never_arm_or_touch_serial(self):
        runner, folder, checkpoint = self.install()
        result = runner.load({"policy_id": "pen"})
        self.assertEqual(result["policy"]["phase"], "ready")
        self.assertEqual(self.lab.mode, "idle")
        self.assertEqual(self.buses, [])
        self.assertTrue(runner.catalog()["policies"][0]["compatible"])
        settings = self.lab.data_dir / "policy-runtime.json"
        settings.chmod(0o644)
        self.assertFalse(runner.catalog()["runtime"]["configured"])

    def test_native_files_features_units_and_identity_are_required(self):
        runner, folder, checkpoint = self.install()
        self.assertEqual(admit(runner.root, "pen")["fps"], 13)
        for path in ("../pen", "/tmp/pen"):
            with self.assertRaises(ValueError):
                admit(runner.root, path)
        (checkpoint / "unsafe.pkl").write_bytes(b"not accepted")
        with self.assertRaisesRegex(ValueError, "safetensors"):
            admit(runner.root, "pen")
        (checkpoint / "unsafe.pkl").unlink()
        metadata = json.loads((folder / "run_policy.json").read_text())
        metadata["action_units"][0] = "radians"
        (folder / "run_policy.json").write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "degrees"):
            admit(runner.root, "pen")

    def test_checkpoint_change_requires_reload_and_start_is_explicit(self):
        runner = self.prepare()
        for mutation in (
            {"operator_present": False},
            {"workspace_clear": False},
            {"duration_seconds": 121},
            {"expected_generation": True},
            {"expected_generation": self.lab.generation - 1},
        ):
            with self.assertRaises(ValueError):
                runner.start(self.run_payload() | mutation)
        (runner.model["checkpoint"] / "model.safetensors").write_text("changed")
        with self.assertRaisesRegex(ValueError, "files changed"):
            runner.start(self.run_payload())
        self.assertEqual(self.lab.mode, "manual")

    def test_start_requires_recent_camera_and_full_arm_check(self):
        runner = self.prepare()
        runner.check = None
        with self.assertRaisesRegex(ValueError, "Check this policy"):
            runner.start(self.run_payload())
        runner.check = {"passed": True, "camera_index": 1, "checked_at": time.monotonic()}
        with self.assertRaisesRegex(ValueError, "Check this policy"):
            runner.start(self.run_payload())
        runner.check["camera_index"] = 0
        runner.check["checked_at"] -= 61
        with self.assertRaisesRegex(ValueError, "Check this policy"):
            runner.start(self.run_payload())

    def test_targets_obey_speed_acceleration_limits_and_manual_heartbeat_cannot_extend_authority(
        self,
    ):
        runner = self.prepare()
        with patch.object(runner, "_run"):
            runner.start(self.run_payload())
            runner.thread.join(timeout=1)
        measured = copy.deepcopy(self.lab.positions)
        runner.predicted = {j: measured[j] + 5 for j in JOINTS}
        now = time.monotonic()
        runner.last_apply = now
        previous, velocity = measured, dict.fromkeys(JOINTS, 0.0)
        for step in range(1, 201):
            if step == 80:
                runner.predicted = {j: measured[j] - (3 if j != "gripper" else -3) for j in JOINTS}
            moment = now + step * 0.025
            runner.last_prediction = moment
            goals = runner.apply(previous, moment)
            for j in JOINTS:
                current = (goals[j] - previous[j]) / 0.025
                self.assertLessEqual(abs(current), MAX_SPEED + 1e-8)
                self.assertLessEqual(abs(current - velocity[j]), MAX_ACCELERATION * 0.025 + 1e-8)
                velocity[j] = current
            previous = goals
        with self.assertRaisesRegex(ValueError, "cannot renew"):
            self.lab.control_heartbeat({"expected_generation": self.lab.generation})
        runner.operator_deadline = time.monotonic() - 1
        with self.assertRaisesRegex(HardwareError, "heartbeat"):
            runner.apply(previous, time.monotonic())

    def test_stop_does_not_wait_for_inference_and_old_reply_cannot_stop_new_controller(self):
        runner = self.prepare()
        entered, release = threading.Event(), threading.Event()

        def request(message, timeout=2):
            if message.get("reset"):
                return {"reset": True}
            entered.set()
            release.wait(timeout=3)
            return {"action": [0.0] * 5 + [50.0]}

        runner.runtime.request = request
        runner.start(self.run_payload())
        self.assertTrue(entered.wait(timeout=1))
        before = time.monotonic()
        old_bus, old_thread = self.buses[-1], runner.thread
        self.lab.stop()
        self.assertLess(time.monotonic() - before, 0.5)
        self.assertTrue(all(("Torque_Enable", i, 0) in old_bus.writes for i in range(1, 7)))
        self.start()
        generation = self.lab.generation
        release.set()
        old_thread.join(timeout=1)
        self.assertFalse(old_thread.is_alive())
        self.assertEqual(self.lab.generation, generation)
        self.assertEqual(self.lab.mode, "manual")

    def test_invalid_native_output_stops_and_disables_torque(self):
        runner = self.prepare()
        runner.runtime.action = [float("nan")] * 6
        runner.start(self.run_payload())
        deadline = time.monotonic() + 2
        while self.lab.mode != "idle" and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.lab.mode, "idle")
        self.assertFalse(self.lab.torque)
        self.assertIn("non-finite", self.lab.error)

    def test_preflight_measures_native_camera_path_without_arming(self):
        runner, _, _ = self.install()
        runner.load({"policy_id": "pen"})
        self.lab.camera = Camera()
        runner.preflight({"camera_index": 0})
        self.assertTrue(runner.check["passed"])
        self.assertEqual(self.lab.mode, "idle")
        self.assertEqual(self.buses, [])

    def test_preflight_rejects_repeated_camera_frame(self):
        runner, _, _ = self.install()
        runner.load({"policy_id": "pen"})
        self.lab.camera = Camera()
        self.lab.camera.stamp = time.monotonic()
        runner.preflight({"camera_index": 0})
        self.assertFalse(runner.check["passed"])
        self.assertEqual(runner.check["fresh_frames"], 1)


if __name__ == "__main__":
    unittest.main()
