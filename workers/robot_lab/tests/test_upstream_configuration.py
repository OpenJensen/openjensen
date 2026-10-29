"""Upstream motor configuration without energizing real hardware."""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_robot_lab as fixtures
from hardware import HardwareError


class ConfigurationTests(unittest.TestCase):
    setUp = fixtures.SafetyTests.setUp
    tearDown = fixtures.SafetyTests.tearDown
    start = fixtures.SafetyTests.start

    def test_feedback_mode_clears_only_phase_bit_four_before_any_enable(self):
        factory = self.lab.bus_factory

        def phase_bus(*args):
            bus = factory(*args)
            bus.writes.extend(("Phase", sid, 0xB4) for sid in range(1, 7))
            return bus

        self.lab.bus_factory = phase_bus
        self.start()
        writes = self.buses[-1].writes
        first_enable = next(
            i for i, (name, _, value) in enumerate(writes) if name == "Torque_Enable" and value == 1
        )
        for sid in range(1, 7):
            self.assertIn(("Phase", sid, 0xA4), writes[:first_enable])
            self.assertEqual(self.buses[-1].read("Lock", sid), 1)
        self.assertTrue(self.lab.torque)

    def test_phase_readback_failure_relocks_and_never_enables_motion(self):
        original = fixtures.FakeBus.read

        def read(bus, register, sid):
            return 0x10 if register == "Phase" else original(bus, register, sid)

        with (
            patch.object(fixtures.FakeBus, "read", read),
            self.assertRaisesRegex(HardwareError, "Phase read-back"),
        ):
            self.start()
        bus = self.buses[-1]
        self.assertIn(("Lock", 1, 1), bus.writes)
        self.assertFalse(any(r == "Torque_Enable" and value == 1 for r, _, value in bus.writes))
        self.assertFalse(bus.opened)

    def test_failed_eeprom_relock_attempts_every_selected_motor_and_prevents_enable(self):
        factory = self.lab.bus_factory
        attempted = []

        def failing(*args):
            bus = factory(*args)

            def on_write(register, sid, value):
                if register == "Lock" and value == 1:
                    attempted.append(sid)
                    if sid == 3:
                        raise OSError("motor 3 lock failed")

            bus.on_write = on_write
            return bus

        self.lab.bus_factory = failing
        with self.assertRaisesRegex(HardwareError, "EEPROM re-lock failed"):
            self.start()
        self.assertEqual(attempted, list(range(1, 7)))
        self.assertFalse(
            any(r == "Torque_Enable" and value == 1 for r, _, value in self.buses[-1].writes)
        )

    def test_passive_leader_gets_upstream_timing_and_position_mode_without_torque(self):
        self.lab.port_provider = lambda: [{"device": "/dev/cu.fake"}, {"device": "/dev/cu.leader"}]
        path = self.lab.calibration_path("leader", "leader")
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(fixtures.CALIBRATION))
        self.start(mode="leader", leader_port="/dev/cu.leader", leader_id="leader")
        leader = self.buses[-1]
        for sid in range(1, 7):
            for setting, value in [
                ("Operating_Mode", 0),
                ("Return_Delay_Time", 0),
                ("Maximum_Acceleration", 254),
                ("Acceleration", 254),
            ]:
                self.assertIn((setting, sid, value), leader.writes)
            self.assertEqual(leader.read("Lock", sid), 1)
        self.assertFalse(any(r == "Torque_Enable" and value == 1 for r, _, value in leader.writes))
        self.assertFalse(
            any(
                r in {"P_Coefficient", "Max_Torque_Limit", "Protection_Current"}
                for r, _, _ in leader.writes
            )
        )

    def test_half_turn_homing_clamps_the_unrepresentable_encoder_endpoint(self):
        self.lab.calibration_start(
            {
                "port": "/dev/cu.fake",
                "robot_id": "test",
                "role": "follower",
                "confirm": "START CALIBRATION",
            }
        )
        bus = self.buses[-1]
        bus.raw["wrist_flex"] = 4095
        self.lab.calibration_center({})
        self.assertEqual(self.lab.calibration_state["offsets"]["wrist_flex"], 2047)
        self.assertIn(("Homing_Offset", 4, 2047), bus.writes)
        self.assertEqual(self.lab.calibration_state["phase"], "ranges")
        self.assertFalse(any(r == "Torque_Enable" and value == 1 for r, _, value in bus.writes))


if __name__ == "__main__":
    unittest.main()
