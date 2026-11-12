"""ACT compatibility, bounded control and Stop races on fake hardware only."""

import base64
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
from policy import MAX_ACCELERATION, MAX_SPEED, PREDICTION_TOLERANCE, UNITS, admit
from motion import control_limits
from policy_contract import PIPELINES

FRONT, SIDE = "observation.images.front", "observation.images.camera_4"


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
        (checkpoint / "model.safetensors").write_text("{}")
        for filename, registries in PIPELINES.items():
            steps = []
            for index, registry in enumerate(registries):
                step = {"registry_name": registry, "config": {}}
                if registry in {"normalizer_processor", "unnormalizer_processor"}:
                    step["state_file"] = f"state-{index}-{registry}.safetensors"
                    (checkpoint / step["state_file"]).write_text("{}")
                steps.append(step)
            (checkpoint / filename).write_text(json.dumps({"name": filename, "steps": steps}))
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

    def test_offline_preview_needs_no_devices_or_control_session(self):
        runner, _, _ = self.install()
        runner.load({"policy_id": "pen"})
        generation = self.lab.generation
        result = runner.preview_prediction({"synthetic": True})
        self.assertEqual(result["policy"]["preview"]["action"], [0.0] * 5 + [50.0])
        self.assertFalse(result["policy"]["preview"]["executed"])
        self.assertEqual(self.lab.generation, generation)
        self.assertEqual(self.lab.mode, "idle")
        self.assertFalse(self.lab.torque)
        self.assertEqual(self.buses, [])

    def test_uploaded_preview_accepts_explicit_null_camera_without_camera_access(self):
        runner, _, _ = self.install()
        runner.load({"policy_id": "pen"})
        with patch.object(
            self.lab.camera,
            "capture_timed",
            side_effect=AssertionError("camera access"),
            create=True,
        ):
            result = runner.preview_prediction(
                {"state": [0.0] * 6, "jpeg": "YQ==", "camera_index": None}
            )
        self.assertEqual(
            result["policy"]["preview"]["source"], "Uploaded image and supplied model state"
        )
        self.assertEqual(self.buses, [])

    def test_unknown_units_and_fps_remain_offline_even_with_valid_weights(self):
        runner, folder, _ = self.install()
        path = folder / "run_policy.json"
        metadata = json.loads(path.read_text()) | {
            "offline_only": True,
            "fps": None,
            "state_units": None,
            "action_units": None,
            "joint_names": None,
        }
        path.write_text(json.dumps(metadata))
        runner.load({"policy_id": "pen"})
        self.assertTrue(runner.model["offline_only"])
        self.assertIsNone(runner.model["fps"])
        runner.preview_prediction({"synthetic": True})
        with self.assertRaisesRegex(ValueError, "Offline inference"):
            runner.preflight({"camera_index": 0})
        self.assertEqual(runner.phase, "ready")
        self.start()
        with self.assertRaisesRegex(ValueError, "offline inference"):
            runner.start(self.run_payload())
        self.assertEqual(self.lab.mode, "manual")

    def test_invalid_task_does_not_leave_preview_checking(self):
        runner, _, _ = self.install()
        runner.load({"policy_id": "pen"})
        runner.model["task_required"] = True
        with self.assertRaisesRegex(ValueError, "task instruction"):
            runner.preview_prediction({"synthetic": True, "task": ""})
        self.assertEqual(runner.phase, "ready")
        with self.assertRaisesRegex(ValueError, "task instruction"):
            runner.preflight({"camera_index": 0, "task": ""})
        self.assertEqual(runner.phase, "ready")

    def test_stop_cancels_preview_without_any_motor_access(self):
        runner, _, _ = self.install()
        runner.load({"policy_id": "pen"})
        entered, release = threading.Event(), threading.Event()
        errors = []

        def request(message, timeout=2):
            if message.get("reset"):
                return {"reset": True}
            entered.set()
            release.wait(timeout=3)
            return {"action": [0.0] * 6}

        runner.runtime.request = request

        def preview():
            try:
                runner.preview_prediction({"synthetic": True})
            except ValueError as error:
                errors.append(str(error))

        thread = threading.Thread(target=preview)
        thread.start()
        self.assertTrue(entered.wait(timeout=1))
        self.lab.stop()
        release.set()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertTrue(any("cancelled" in error for error in errors))
        self.assertIsNone(runner.preview)
        self.assertEqual(self.buses, [])
        self.assertEqual(self.lab.mode, "idle")

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

    def test_processor_classes_artifacts_external_state_and_configs_are_rejected(self):
        runner, _, checkpoint = self.install()
        path = checkpoint / "policy_preprocessor.json"
        original = json.loads(path.read_text())
        for change in (
            {"class": "untrusted.module.Object"},
            {"registry_name": "other_processor"},
            {"artifacts": {"model": "../external"}},
            {"config": []},
            {"config": {"unknown": "argument"}},
            {"state_file": "../external.safetensors"},
            {"state_file": "/tmp/external.safetensors"},
        ):
            document = copy.deepcopy(original)
            document["steps"][0].update(change)
            path.write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                admit(runner.root, "pen")
        document = copy.deepcopy(original)
        document["steps"].append(copy.deepcopy(document["steps"][0]))
        path.write_text(json.dumps(document))
        with self.assertRaises(ValueError):
            admit(runner.root, "pen")

    def test_architecture_allocations_are_bounded_before_native_loading(self):
        runner, _, checkpoint = self.install()
        path = checkpoint / "config.json"
        original = json.loads(path.read_text())
        for key, value in (
            ("dim_model", 8192),
            ("chunk_size", 1000000),
            ("n_encoder_layers", 500),
            ("vision_backbone", "unrecognized"),
            ("n_obs_steps", True),
        ):
            path.write_text(json.dumps(original | {key: value}))
            with self.assertRaises(ValueError):
                admit(runner.root, "pen")
        oversized = copy.deepcopy(original)
        oversized["input_features"]["observation.images.front"]["shape"] = [3, 4096, 4096]
        path.write_text(json.dumps(oversized))
        with self.assertRaisesRegex(ValueError, "image budget"):
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
                self.assertLessEqual(abs(current - velocity[j]), MAX_ACCELERATION * 0.025 + 1e-6)
                velocity[j] = current
            previous = goals
        with self.assertRaisesRegex(ValueError, "cannot renew"):
            self.lab.control_heartbeat({"expected_generation": self.lab.generation})
        runner.operator_deadline = time.monotonic() - 1
        with self.assertRaisesRegex(HardwareError, "heartbeat"):
            runner.apply(previous, time.monotonic())
        runner.operator_deadline = time.monotonic() + 10
        runner.run_deadline = time.monotonic() - 1
        self.assertTrue(runner.expired())

    def test_small_prediction_overshoots_keep_motion_inside_original_bounds(self):
        runner = self.prepare()
        with patch.object(runner, "_run"):
            runner.start(self.run_payload())
            runner.thread.join(timeout=1)
        ranges = control_limits(self.lab.calibration)
        calibration = copy.deepcopy(self.lab.calibration)
        for side, direction in (("min", -1), ("max", 1)):
            with self.subTest(side=side):
                measured = {j: ranges[j][side] - direction * 0.1 for j in JOINTS}
                runner.commanded = dict(measured)
                runner.velocity = dict.fromkeys(JOINTS, 0.0)
                predicted = {j: ranges[j][side] + direction * PREDICTION_TOLERANCE for j in JOINTS}
                runner.predicted = dict(predicted)
                now = time.monotonic()
                runner.last_apply = now
                previous, velocity = measured, dict.fromkeys(JOINTS, 0.0)
                for step in range(1, 201):
                    moment = now + step * 0.025
                    runner.last_prediction = moment
                    goals = runner.apply(previous, moment)
                    for j in JOINTS:
                        self.assertGreaterEqual(goals[j], ranges[j]["min"])
                        self.assertLessEqual(goals[j], ranges[j]["max"])
                        current = (goals[j] - previous[j]) / 0.025
                        self.assertLessEqual(abs(current), MAX_SPEED + 1e-8)
                        self.assertLessEqual(
                            abs(current - velocity[j]), MAX_ACCELERATION * 0.025 + 1e-6
                        )
                        velocity[j] = current
                    previous = goals
                self.assertEqual(runner.predicted, predicted)
                self.assertEqual(self.lab.calibration, calibration)

    def test_large_or_nonfinite_prediction_overshoots_still_reject_without_goals(self):
        runner = self.prepare()
        with patch.object(runner, "_run"):
            runner.start(self.run_payload())
            runner.thread.join(timeout=1)
        measured = dict(self.lab.positions)
        ranges = control_limits(self.lab.calibration)
        for joint in JOINTS:
            for target in (
                ranges[joint]["min"] - PREDICTION_TOLERANCE - 0.001,
                ranges[joint]["max"] + PREDICTION_TOLERANCE + 0.001,
                float("nan"),
                float("inf"),
            ):
                with self.subTest(joint=joint, target=target):
                    runner.commanded = dict(measured)
                    runner.predicted = dict(measured) | {joint: target}
                    runner.last_prediction = time.monotonic()
                    before = list(self.buses[-1].writes)
                    with self.assertRaisesRegex(HardwareError, joint):
                        runner.apply(measured, time.monotonic())
                    self.assertEqual(runner.commanded, measured)
                    self.assertEqual(self.buses[-1].writes, before)

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

    def run_with_slow_predictions(self, slow):
        """Start a run whose inference takes 0.26 s on the calls ``slow`` selects."""
        runner = self.prepare()
        calls = []
        answer = runner.runtime.request

        def request(message, timeout=2):
            if not message.get("reset"):
                calls.append(time.monotonic())
                if slow(len(calls)):
                    time.sleep(0.26)
            return answer(message)

        runner.runtime.request = request
        runner.start(self.run_payload())

        def control_loop():
            # The arm's own loop: applies the latest prediction and enforces its age limit.
            while self.lab.mode == "policy":
                try:
                    runner.apply(dict(self.lab.positions), time.monotonic())
                except HardwareError as error:
                    self.lab.stop(str(error))
                    return
                time.sleep(0.01)

        threading.Thread(target=control_loop, daemon=True).start()
        return runner, calls

    def test_one_slow_step_while_planning_a_chunk_does_not_stop_a_run(self):
        """At 23 FPS with two cameras, planning took 0.2 s and every run stopped at once,
        although the check before it had passed."""
        runner, calls = self.run_with_slow_predictions(lambda number: number == 1)
        deadline = time.monotonic() + 3
        while runner.steps < 8 and self.lab.mode == "policy" and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.lab.mode, "policy", self.lab.error)
        self.assertGreaterEqual(runner.steps, 8)
        # The run continued at its own rate instead of hurrying through missed steps.
        gaps = [later - earlier for earlier, later in zip(calls[1:], calls[2:], strict=False)]
        self.assertGreater(min(gaps), 0.8 / runner.model["fps"])
        self.lab.stop()

    def test_a_policy_that_keeps_falling_behind_is_still_stopped(self):
        runner, calls = self.run_with_slow_predictions(lambda number: True)
        deadline = time.monotonic() + 4
        while self.lab.mode != "idle" and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.lab.mode, "idle")
        self.assertFalse(self.lab.torque)
        self.assertIn("cannot maintain the trained frame rate", self.lab.error)
        self.assertLessEqual(len(calls), 4)

    def planning_takes(self, runner, seconds):
        """Make the first prediction after every reset take ``seconds``, like a policy
        that plans a chunk of actions and then replays it."""
        state = {"fresh": True}
        answer = runner.runtime.request

        def request(message, timeout=2):
            if message.get("reset"):
                state["fresh"] = True
            elif state["fresh"]:
                state["fresh"] = False
                time.sleep(seconds)
            return answer(message)

        runner.runtime.request = request

    def test_slow_planning_fails_the_check_and_says_how_to_run_the_policy(self):
        runner, _, _ = self.install()
        runner.load({"policy_id": "pen"})
        self.lab.camera = Camera()
        self.planning_takes(runner, 0.6)
        runner.preflight({"camera_index": 0})
        check = runner.check
        self.assertFalse(check["passed"])
        self.assertTrue(check["planning_pause_needed"])
        self.assertEqual(check["prediction_limit_ms"], 500)
        self.assertRegex(check["message"], r"takes 6\d\d ms to plan .* 500 ms is allowed")
        self.assertIn("allow planning pauses", check["message"])
        # The slow prediction was measured, not cut off: the policy is still loaded.
        self.assertFalse(runner.runtime.closed)
        self.assertEqual(runner.phase, "ready")

        runner.preflight({"camera_index": 0, "allow_planning_pause": True})
        check = runner.check
        self.assertTrue(check["passed"], check)
        self.assertTrue(check["planning_pause"])
        self.assertFalse(check["planning_pause_needed"])
        self.assertEqual(check["prediction_limit_ms"], 1500)
        self.assertEqual(self.buses, [])
        with self.assertRaisesRegex(ValueError, "Choose a camera"):
            runner.preflight({"camera_index": 0, "allow_planning_pause": "yes"})

    def test_planning_slower_than_the_pause_allows_cannot_be_run_at_all(self):
        runner, _, _ = self.install()
        runner.load({"policy_id": "pen"})
        self.lab.camera = Camera()
        self.planning_takes(runner, 1.6)
        for payload in ({"camera_index": 0}, {"camera_index": 0, "allow_planning_pause": True}):
            runner.preflight(payload)
            self.assertFalse(runner.check["passed"])
            self.assertFalse(runner.check["planning_pause_needed"])
            self.assertIn("cannot sustain", runner.check["message"])

    def test_a_run_pauses_for_planning_only_when_its_check_allowed_it(self):
        runner = self.prepare()
        runner.check = {
            "passed": True,
            "camera_index": 0,
            "planning_pause": True,
            "checked_at": time.monotonic(),
        }
        # The check and the run must agree, in both directions.
        with self.assertRaisesRegex(ValueError, "Check this policy"):
            runner.start(self.run_payload())
        runner.check["planning_pause"] = False
        with self.assertRaisesRegex(ValueError, "Check this policy"):
            runner.start(self.run_payload() | {"allow_planning_pause": True})
        with self.assertRaisesRegex(ValueError, "Run duration"):
            runner.start(self.run_payload() | {"allow_planning_pause": 1})
        self.assertEqual(self.lab.mode, "manual")

    def run_slow_planner(self, pause):
        runner = self.prepare()
        runner.check["planning_pause"] = pause
        self.planning_takes(runner, 0.7)
        runner.start(self.run_payload() | ({"allow_planning_pause": True} if pause else {}))

        def control_loop():
            while self.lab.mode == "policy":
                try:
                    runner.apply(dict(self.lab.positions), time.monotonic())
                except HardwareError as error:
                    self.lab.stop(str(error))
                    return
                time.sleep(0.01)

        threading.Thread(target=control_loop, daemon=True).start()
        return runner

    def test_an_allowed_planning_pause_does_not_stop_the_run(self):
        runner = self.run_slow_planner(True)
        deadline = time.monotonic() + 3
        while runner.steps < 6 and self.lab.mode == "policy" and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.lab.mode, "policy", self.lab.error)
        self.assertGreaterEqual(runner.steps, 6)
        self.lab.stop()

    def test_without_the_allowance_slow_planning_disables_torque_as_before(self):
        self.run_slow_planner(False)
        deadline = time.monotonic() + 3
        while self.lab.mode != "idle" and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.lab.mode, "idle")
        self.assertFalse(self.lab.torque)
        self.assertIn("fell behind", self.lab.error)

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

    def two_cameras(self):
        """The same policy trained on two views, as a two-camera recording produces."""
        runner, _, checkpoint = self.install()
        config = json.loads((checkpoint / "config.json").read_text())
        config["input_features"][SIDE] = {"shape": [3, 480, 640]}
        (checkpoint / "config.json").write_text(json.dumps(config))
        runner.load({"policy_id": "pen"})
        self.lab.camera = Camera()
        self.lab.camera.latest_timed_jpeg = lambda index, max_age=0.15: {
            "jpeg": b"view-%d" % index,
            "captured_monotonic": time.monotonic(),
        }
        sent = []
        answer = runner.runtime.request
        runner.runtime.request = lambda message, timeout=2: (
            sent.append(message),
            answer(message),
        )[1]
        return runner, sent

    def test_a_two_camera_policy_is_checked_with_a_camera_for_each_view(self):
        runner, sent = self.two_cameras()
        self.assertEqual(runner.catalog()["policies"][0]["camera_keys"], [FRONT, SIDE])
        # One camera cannot stand in for two views, and neither can the same camera twice.
        with self.assertRaisesRegex(ValueError, "trained on 2 cameras.*front, camera_4"):
            runner.preflight({"camera_index": 0})
        for cameras in (
            {FRONT: 0, SIDE: 0},
            {FRONT: 0},
            {FRONT: 2, SIDE: 3},
            {FRONT: 0, SIDE: "3"},
        ):
            with self.assertRaisesRegex(ValueError, "different available camera"):
                runner.preflight({"camera_index": 0, "cameras": cameras})
        self.assertEqual(runner.phase, "ready")
        self.assertEqual(sent, [])

        runner.preflight({"camera_index": 2, "cameras": {SIDE: 5, FRONT: 2}})
        self.assertTrue(runner.check["passed"])
        self.assertEqual(runner.check["cameras"], {FRONT: 2, SIDE: 5})
        observations = [message for message in sent if not message.get("reset")]
        self.assertGreaterEqual(len(observations), 16)
        for message in observations:
            self.assertNotIn("jpeg", message)
            self.assertEqual(
                {key: base64.b64decode(value) for key, value in message["images"].items()},
                {FRONT: b"view-2", SIDE: b"view-5"},
            )
        self.assertEqual(self.buses, [])

    def test_a_two_camera_check_fails_when_either_camera_repeats_frames(self):
        runner, _ = self.two_cameras()
        frozen = time.monotonic()
        self.lab.camera.latest_timed_jpeg = lambda index, max_age=0.15: {
            "jpeg": b"view",
            "captured_monotonic": frozen if index == 5 else time.monotonic(),
        }
        runner.preflight({"camera_index": 2, "cameras": {FRONT: 2, SIDE: 5}})
        self.assertFalse(runner.check["passed"])
        self.assertEqual(runner.check["fresh_frames"], 1)

    def test_a_two_camera_run_uses_the_checked_cameras_and_sends_both_views(self):
        runner, sent = self.two_cameras()
        chosen = {FRONT: 2, SIDE: 5}
        runner.check = {
            "passed": True,
            "camera_index": 2,
            "cameras": chosen,
            "checked_at": time.monotonic(),
        }
        camera = self.lab.camera
        self.start()
        self.lab.camera = camera
        payload = self.run_payload() | {"camera_index": 2}
        with self.assertRaisesRegex(ValueError, "trained on 2 cameras"):
            runner.start(payload)
        # Swapping the views after the check would feed each input the wrong picture.
        with self.assertRaisesRegex(ValueError, "Check this policy"):
            runner.start(payload | {"cameras": {FRONT: 2, SIDE: 6}})
        self.assertEqual(self.lab.mode, "manual")
        runner.start(payload | {"cameras": chosen})
        deadline = time.monotonic() + 2
        while runner.steps < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.lab.stop()
        self.assertGreaterEqual(runner.steps, 2)
        observations = [message for message in sent if not message.get("reset")]
        self.assertTrue(observations)
        for message in observations:
            self.assertEqual(
                {key: base64.b64decode(value) for key, value in message["images"].items()},
                {FRONT: b"view-2", SIDE: b"view-5"},
            )

    def test_a_two_camera_preview_says_when_one_image_stood_in_for_both_views(self):
        runner, sent = self.two_cameras()
        result = runner.preview_prediction({"state": [0.0] * 6, "jpeg": "YQ=="})
        self.assertIn(
            "same image shown to all 2 camera inputs", result["policy"]["preview"]["source"]
        )
        self.assertTrue(sent[-1]["repeat"])
        self.lab.camera.capture_timed = lambda index: {"jpeg": b"view-%d" % index}
        result = runner.preview_prediction(
            {"state": [0.0] * 6, "camera_index": 2, "cameras": {FRONT: 2, SIDE: 5}}
        )
        self.assertEqual(
            result["policy"]["preview"]["source"], "Selected cameras and supplied model state"
        )
        self.assertEqual(
            {key: base64.b64decode(value) for key, value in sent[-1]["images"].items()},
            {FRONT: b"view-2", SIDE: b"view-5"},
        )
        self.assertNotIn("repeat", sent[-1])
        self.assertEqual(self.buses, [])

    def test_more_than_three_cameras_are_refused(self):
        runner, _, checkpoint = self.install()
        config = json.loads((checkpoint / "config.json").read_text())
        for name in ("a", "b", "c"):
            config["input_features"]["observation.images." + name] = {"shape": [3, 480, 640]}
        (checkpoint / "config.json").write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "one to three RGB cameras"):
            runner.load({"policy_id": "pen"})


if __name__ == "__main__":
    unittest.main()
