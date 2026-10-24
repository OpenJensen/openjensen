"""Full position goals retain calibration limits and encoder-based middle arrival."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hardware import JOINTS
from motion import MiddleMotion, control_limits, middle_targets, step_target


class MotionTests(unittest.TestCase):
    def setUp(self):
        self.calibration = {j: {"range_min": 100, "range_max": 4000} for j in JOINTS}

    def test_middle_is_recorded_half_turn_not_average_of_range(self):
        self.assertEqual(middle_targets(self.calibration), dict.fromkeys(JOINTS, 2047))

    def test_middle_target_is_full_position_and_arrival_uses_encoders(self):
        motion = MiddleMotion(dict.fromkeys(JOINTS, 3000), self.calibration)
        self.assertEqual(motion.targets, dict.fromkeys(JOINTS, 2047))
        self.assertFalse(motion.arrived(dict.fromkeys(JOINTS, 2500)))
        self.assertGreater(motion.progress, 0)
        self.assertLess(motion.progress, 1)
        self.assertTrue(motion.arrived(dict.fromkeys(JOINTS, 2047)))
        self.assertEqual(motion.progress, 1)

    def test_already_at_middle_is_complete_without_any_trajectory(self):
        motion = MiddleMotion(dict.fromkeys(JOINTS, 2047), self.calibration)
        self.assertTrue(motion.arrived(dict.fromkeys(JOINTS, 2047)))

    def test_step_uses_previous_goal_not_a_lagging_encoder(self):
        commanded = dict.fromkeys(JOINTS, 2400)
        target = step_target("elbow_flex", 10, commanded, self.calibration)
        self.assertGreater(target - commanded["elbow_flex"], 110)
        commanded["elbow_flex"] = target
        self.assertGreater(step_target("elbow_flex", 10, commanded, self.calibration), target + 110)

    def test_targets_keep_eight_tick_calibration_margin(self):
        limits = control_limits(self.calibration)
        self.assertAlmostEqual(limits["gripper"]["min"], 8 / 39)
        self.assertAlmostEqual(limits["gripper"]["max"], 100 - 8 / 39)
        self.assertEqual(
            step_target("elbow_flex", 10, {"elbow_flex": 3991}, self.calibration), 3992
        )
        self.assertEqual(step_target("gripper", -10, {"gripper": 109}, self.calibration), 108)

    def test_middle_arrival_tolerance_does_not_change_the_requested_goal(self):
        motion = MiddleMotion(dict.fromkeys(JOINTS, 2200), self.calibration, 2)
        raw = dict.fromkeys(JOINTS, 2047)
        raw["shoulder_lift"] += 22
        raw["gripper"] += 78
        self.assertTrue(motion.arrived(raw))
        raw["shoulder_lift"] += 1
        self.assertFalse(motion.arrived(raw))
        self.assertEqual(motion.targets["shoulder_lift"], 2047)
        for value in (True, "2", None, float("nan"), float("inf"), 0, 5.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MiddleMotion(dict.fromkeys(JOINTS, 2200), self.calibration, value)
