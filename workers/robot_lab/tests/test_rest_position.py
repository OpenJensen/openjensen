"""Rest pose persistence, bounded motion and cancellation without physical hardware."""

import json
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import test_robot_lab as fixtures
from hardware import JOINTS
from rest_position import RestPositions
from server import RequestError


class RestTests(unittest.TestCase):
    setUp = fixtures.SafetyTests.setUp
    tearDown = fixtures.SafetyTests.tearDown
    start = fixtures.SafetyTests.start

    def save(self):
        return self.lab.rest_position({"confirm": "SAVE REST POSITION"}, save=True)

    def go(self):
        return self.lab.rest_position({"confirm": "RETURN TO REST"})

    def test_save_samples_encoders_without_writing_and_survives_restart(self):
        self.start(mode="handguide")
        bus = self.buses[-1]
        bus.raw = dict.fromkeys(JOINTS, 1800)
        before = list(bus.writes)
        self.save()
        self.assertEqual(bus.writes, before)
        self.assertFalse(self.lab.torque)
        loaded = RestPositions(self.lab.data_dir).load("test", self.lab.calibration)
        self.assertEqual(loaded, bus.raw)
        changed = json.loads(json.dumps(self.lab.calibration))
        changed["elbow_flex"]["homing_offset"] = 1
        self.assertIsNone(RestPositions(self.lab.data_dir).load("test", changed))
        self.lab.stop()
        result = self.start()
        self.assertTrue(result["session"]["rest_position"]["saved"])

    def test_return_ramps_all_joints_then_holds_without_releasing(self):
        self.start()
        bus = self.buses[-1]
        bus.raw = dict.fromkeys(JOINTS, 1800)
        self.save()
        bus.raw = dict.fromkeys(JOINTS, 2200)
        before = len(bus.writes)
        self.go()
        rest = self.lab.rest_motion
        previous = dict(bus.raw)
        for i in range(1, 150):
            now = rest.last + 0.05
            self.lab.heartbeat = now
            with patch("motion.time.monotonic", return_value=now):
                self.lab._tick()
            for joint in JOINTS:
                # The ramp rounds raw goals, then upstream normalization truncates
                # on the way back to ticks: allow both quantization steps.
                self.assertLessEqual(
                    abs(bus.raw[joint] - previous[joint]), rest.rates[joint] * 0.05 + 2
                )
            previous = dict(bus.raw)
            if self.lab.rest_motion is None:
                break
        self.assertIsNone(self.lab.rest_motion)
        self.assertEqual(self.lab.rest_phase, "holding")
        self.assertEqual(bus.raw, dict.fromkeys(JOINTS, 1800))
        self.assertTrue(self.lab.torque)
        self.assertEqual(self.lab.mode, "manual")
        self.assertFalse(
            any(
                register == "Torque_Enable" and value == 0
                for register, _, value in bus.writes[before:]
            )
        )

    def test_return_takes_over_leader_and_stop_cancels_further_goals(self):
        self.start()
        self.save()
        self.lab.mode = "leader"
        self.lab.leader = fixtures.FakeBus()
        self.lab.leader.raw = dict.fromkeys(JOINTS, 3000)
        self.lab.leader_calibration = self.lab.calibration
        self.buses[-1].raw = dict.fromkeys(JOINTS, 2200)
        self.go()
        self.assertEqual(self.lab.mode, "manual")
        self.lab._tick()
        self.lab.stop()
        before = list(self.buses[-1].writes)
        self.lab._tick()
        self.assertEqual(self.buses[-1].writes, before)
        self.assertFalse(self.lab.torque)
        self.assertIsNone(self.lab.rest_motion)

    def test_timeout_holds_observed_pose_and_does_not_claim_arrival(self):
        self.start()
        self.save()
        bus = self.buses[-1]
        bus.raw = dict.fromkeys(JOINTS, 2500)
        bus.follow_goals = False
        self.go()
        deadline = self.lab.rest_motion.deadline + 1
        self.lab.heartbeat = deadline
        with patch("motion.time.monotonic", return_value=deadline):
            self.lab._tick()
        self.assertEqual(self.lab.rest_phase, "error")
        self.assertIsNone(self.lab.rest_motion)
        self.assertTrue(self.lab.torque)
        self.assertIn("Holding current position", self.lab.error)
        self.assertEqual(self.lab.commanded_raw, bus.raw)

    def test_rejects_missing_pose_partial_control_unpowered_and_recording(self):
        self.start()
        with self.assertRaisesRegex(RequestError, "Save a rest position"):
            self.go()
        self.save()
        self.lab.writer = SimpleNamespace(episode_active=True)
        with self.assertRaisesRegex(RequestError, "recording episode"):
            self.go()
        with self.assertRaisesRegex(RequestError, "recording episode"):
            self.save()
        self.lab.writer = None
        self.lab.stop()
        self.start(joints=["elbow_flex"])
        with self.assertRaisesRegex(RequestError, "all six"):
            self.go()
        self.lab.stop()
        self.start(mode="handguide")
        with self.assertRaisesRegex(RequestError, "Start joint control"):
            self.go()

    def test_save_rejects_limit_pose_and_return_requires_fresh_heartbeat(self):
        self.start()
        self.buses[-1].raw["elbow_flex"] = 100
        with self.assertRaisesRegex(ValueError, "joint limits"):
            self.save()
        self.buses[-1].raw["elbow_flex"] = 2047
        self.save()
        self.lab.heartbeat = time.monotonic() - 3
        with self.assertRaisesRegex(RequestError, "heartbeat"):
            self.go()

    def test_return_blocks_jog_recording_and_resave(self):
        self.start()
        self.save()
        self.buses[-1].raw = dict.fromkeys(JOINTS, 2300)
        self.go()
        with self.assertRaisesRegex(RequestError, "return to rest"):
            self.lab.jog({"joint": "elbow_flex", "delta": 1})
        with self.assertRaisesRegex(RequestError, "return to rest"):
            self.lab.recording_start({})
        with self.assertRaisesRegex(RequestError, "movement"):
            self.save()
        self.lab.stop()
        self.assertEqual(self.lab.rest_phase, "idle")

    def test_corrupt_pose_is_unavailable_and_confirmation_required(self):
        self.start()
        self.save()
        path = self.lab.rest_positions.root / "test.json"
        path.write_text("[]")
        with self.assertRaisesRegex(RequestError, "Save a rest position"):
            self.go()
        with self.assertRaises(RequestError):
            self.lab.rest_position({})

    def test_stop_during_return_start_prevents_new_goals(self):
        self.start()
        self.save()
        bus = self.buses[-1]
        observed = threading.Event()
        release = threading.Event()
        original = bus.positions
        failures = []

        def blocked_positions():
            observed.set()
            release.wait(3)
            return original()

        bus.positions = blocked_positions

        def attempt():
            try:
                self.go()
            except Exception as error:
                failures.append(error)

        before = len(bus.writes)
        runner = threading.Thread(target=attempt)
        runner.start()
        self.assertTrue(observed.wait(2))
        generation = self.lab.generation
        stopper = threading.Thread(target=self.lab.stop)
        stopper.start()
        deadline = time.monotonic() + 2
        while self.lab.generation == generation and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertNotEqual(self.lab.generation, generation)
        release.set()
        runner.join(3)
        stopper.join(3)
        self.assertFalse(runner.is_alive() or stopper.is_alive())
        self.assertTrue(any(isinstance(error, RequestError) for error in failures))
        self.assertFalse(any(register == "Goal_Position" for register, _, _ in bus.writes[before:]))
        self.assertFalse(self.lab.torque)
