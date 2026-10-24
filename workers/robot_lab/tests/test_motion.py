"""Trajectory limits are tested without serial devices."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hardware import JOINTS, HardwareError
from motion import MiddleMotion, middle_targets


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
