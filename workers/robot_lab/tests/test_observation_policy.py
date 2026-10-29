"""LeRobot observation semantics with bounded OpenJensen control extensions."""

import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_robot_lab as fixtures
from hardware import JOINTS, HardwareError, denormalized, normalized
from server import validate_raw


class NormalizationTests(unittest.TestCase):
    def test_upstream_degree_and_percent_reference_vectors(self):
        # SerialMotorsBus._normalize, upstream ref in LEROBOT-PROVENANCE.json.
        calibration = copy.deepcopy(fixtures.CALIBRATION)
        calibration["wrist_flex"].update(range_min=861, range_max=3185)
        for raw in (0, 860, 861, 2047, 3185, 3186, 4095):
            self.assertAlmostEqual(
                normalized("wrist_flex", raw, calibration), (raw - 2023) * 360 / 4095
            )
        for raw in (0, 99, 100, 101, 2047, 4000, 4001, 4095):
            bounded = min(4000, max(100, raw))
            self.assertEqual(
                normalized("gripper", raw, calibration), ((bounded - 100) / 3900) * 100
            )
        for target in (-1.0, 0.0, 0.1, 12.345, 50.0, 99.9, 100.0, 101.0):
            expected = int((min(100.0, max(0.0, target)) / 100) * 3900 + 100)
            self.assertEqual(denormalized("gripper", target, calibration), expected)

    def test_physical_encoder_domain_and_default_strict_validation_are_preserved(self):
        for value in (-1, 4096, True, 1.5):
            raw = dict.fromkeys(JOINTS, 2047)
            raw["wrist_flex"] = value
            with self.subTest(value=value), self.assertRaises(HardwareError):
                validate_raw(raw, fixtures.CALIBRATION, calibration_bounds=False)
        raw = dict.fromkeys(JOINTS, 2047)
        raw["wrist_flex"] = 4001
        with self.assertRaises(HardwareError):
            validate_raw(raw, fixtures.CALIBRATION)
        self.assertEqual(validate_raw(raw, fixtures.CALIBRATION, calibration_bounds=False), raw)


class HumanControlTests(unittest.TestCase):
    setUp = fixtures.SafetyTests.setUp
    tearDown = fixtures.SafetyTests.tearDown
    start = fixtures.SafetyTests.start

    def test_reported_one_tick_wrist_crossing_continues_and_is_logged_once(self):
        calibration = copy.deepcopy(fixtures.CALIBRATION)
        calibration["wrist_flex"].update(range_min=861, range_max=3185)
        self.lab.calibration_path("test", "follower").write_text(json.dumps(calibration))
        factory = self.lab.bus_factory

        def calibrated(*args):
            bus = factory(*args)
            bus.calibration = copy.deepcopy(calibration)
            return bus

        self.lab.bus_factory = calibrated
        self.start()
        bus = self.buses[-1]
        bus.follow_goals = False
        bus.raw["wrist_flex"] = 3186
        generation = self.lab.generation
        for _ in range(30):
            self.lab._tick()
        self.assertEqual(self.lab.mode, "manual")
        self.assertEqual(self.lab.generation, generation)
        self.assertTrue(self.lab.torque and bus.opened)
        self.assertEqual(self.lab.last_raw["follower"]["wrist_flex"], 3186)
        self.assertAlmostEqual(self.lab.positions["wrist_flex"], (3186 - 2023) * 360 / 4095)
        events = self.lab.diagnostics.snapshot(full=True)["events"]
        self.assertEqual(sum(e["event"] == "calibration_endpoint_observed" for e in events), 1)
        self.assertIsNone(self.lab.diagnostics.snapshot()["last_stop"])
        bus.raw["wrist_flex"] = 3185
        self.lab._tick()
        self.assertEqual(self.lab.range_excursions["follower"], {})

    def test_leader_endpoint_readings_map_to_bounded_follower_goals(self):
        self.lab.port_provider = lambda: [{"device": "/dev/cu.fake"}, {"device": "/dev/cu.leader"}]
        path = self.lab.calibration_path("leader", "leader")
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(fixtures.CALIBRATION))
        self.start(mode="leader", leader_port="/dev/cu.leader", leader_id="leader")
        follower, leader = self.buses[-2:]
        leader.raw.update(wrist_flex=4001, gripper=99)
        self.lab._tick()
        self.assertEqual(self.lab.mode, "leader")
        self.assertTrue(self.lab.torque)
        sent = [
            (motor, value)
            for register, motor, value in follower.writes
            if register == "Goal_Position"
        ]
        self.assertTrue(sent)
        self.assertTrue(all(108 <= value <= 3992 for _, value in sent[-6:]))
        self.assertEqual(self.lab.last_raw["leader"]["gripper"], 99)

    def test_outside_starting_pose_is_still_rejected_before_enable(self):
        factory = self.lab.bus_factory

        def outside(*args):
            bus = factory(*args)
            bus.raw["wrist_flex"] = 4001
            return bus

        self.lab.bus_factory = outside
        with self.assertRaises(HardwareError):
            self.start()
        self.assertFalse(any(r == "Torque_Enable" and v == 1 for r, _, v in self.buses[-1].writes))

    def test_invalid_feedback_still_stops_and_retains_the_offending_sample(self):
        self.start()
        bus = self.buses[-1]
        bus.raw["wrist_flex"] = 4096
        with self.assertRaises(HardwareError):
            self.lab._tick()
        self.lab.stop("Invalid encoder reading")
        self.assertFalse(self.lab.torque or bus.opened)
        self.assertEqual(
            self.lab.diagnostics.snapshot()["last_stop"]["context"]["raw_positions"]["follower"][
                "wrist_flex"
            ],
            4096,
        )


if __name__ == "__main__":
    unittest.main()
