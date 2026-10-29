"""Optional differential checks against installed, offline upstream LeRobot."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hardware import IDS, denormalized, normalized


@unittest.skipUnless(
    os.getenv("OPENJENSEN_VERIFY_LEROBOT") == "1", "Explicit official reference lane required"
)
class OfficialReferenceTests(unittest.TestCase):
    def test_normalization_matches_upstream_without_opening_a_serial_port(self):
        from lerobot.motors import Motor, MotorCalibration, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus

        bus = FeetechMotorsBus.__new__(FeetechMotorsBus)
        bus.motors = {
            j: Motor(
                sid,
                "sts3215",
                MotorNormMode.RANGE_0_100 if j == "gripper" else MotorNormMode.DEGREES,
            )
            for j, sid in IDS.items()
        }
        calibration = {
            j: dict(id=sid, drive_mode=0, homing_offset=0, range_min=861, range_max=3185)
            for j, sid in IDS.items()
        }
        bus.calibration = {j: MotorCalibration(**entry) for j, entry in calibration.items()}
        bus._id_to_name_dict = {sid: joint for joint, sid in IDS.items()}
        bus._id_to_model_dict = dict.fromkeys(IDS.values(), "sts3215")
        for joint, sid in IDS.items():
            for raw in (0, 860, 861, 862, 2047, 3185, 3186, 4095):
                self.assertEqual(
                    normalized(joint, raw, calibration), bus._normalize({sid: raw})[sid]
                )
            for target in (-180.0, -90.0, 0.0, 0.1, 12.345, 50.0, 99.9, 100.0, 180.0):
                upstream = bus._unnormalize({sid: target})[sid]
                # OpenJensen additionally bounds commanded goals to calibration.
                expected = max(861, min(3185, upstream))
                self.assertEqual(denormalized(joint, target, calibration), expected)


if __name__ == "__main__":
    unittest.main()
