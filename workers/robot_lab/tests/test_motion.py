"""Trajectory limits are tested without serial devices."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hardware import JOINTS, HardwareError
from motion import JointMotion, MiddleMotion, control_limits, middle_targets


class MotionTests(unittest.TestCase):
    def test_middle_is_recorded_half_turn_not_average_of_range(self):
        calibration = {j: {"range_min": 1000, "range_max": 4000} for j in JOINTS}
        self.assertEqual(middle_targets(calibration), dict.fromkeys(JOINTS, 2047))

    def test_delayed_tick_does_not_catch_up_with_large_jump(self):
        calibration = {j: {"range_min": 1000, "range_max": 4000} for j in JOINTS}
        with patch("motion.time.monotonic", return_value=10) as clock:
            motion = MiddleMotion(dict.fromkeys(JOINTS, 3000), calibration)
            clock.return_value = 12
            targets = motion.advance()
            self.assertTrue(all(2995 <= v <= 3000 for v in targets.values()))
            clock.return_value = motion.deadline + 0.1
            with self.assertRaisesRegex(HardwareError, "timed out"):
                motion.advance()

    def test_already_at_middle_has_no_division_or_target_jump(self):
        calibration = {j: {"range_min": 1000, "range_max": 4000} for j in JOINTS}
        motion = MiddleMotion(dict.fromkeys(JOINTS, 2047), calibration)
        self.assertEqual(motion.advance(), dict.fromkeys(JOINTS, 2047))
        self.assertEqual(motion.progress, 1)

    def test_slider_trajectory_limits_degrees_and_gripper_percent_per_elapsed_time(self):
        calibration = {j: {"range_min": 100, "range_max": 4000} for j in JOINTS}
        for joint, maximum_step in (("elbow_flex", 5), ("gripper", 16)):
            with (
                self.subTest(joint=joint),
                patch("motion.time.monotonic", return_value=10) as clock,
            ):
                motion = JointMotion(joint, 2000, 3900, calibration)
                clock.return_value = 20  # A late tick advances at most 100 ms.
                target = motion.advance()[joint]
                self.assertLessEqual(target - 2000, maximum_step)
                self.assertGreater(target, 2000)

    def test_slider_targets_stay_inside_calibration_with_eight_tick_margin(self):
        calibration = {j: {"range_min": 100, "range_max": 4000} for j in JOINTS}
        limits = control_limits(calibration)
        self.assertAlmostEqual(limits["gripper"]["min"], 8 / 39)
        self.assertAlmostEqual(limits["gripper"]["max"], 100 - 8 / 39)
        self.assertLess(limits["elbow_flex"]["min"], 0)
        self.assertGreater(limits["elbow_flex"]["max"], 0)
