"""Grip-load compatibility and stop proof use fake buses/packets only."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_hardware_transport as transport
import test_robot_lab as fixtures
from hardware import HardwareError


class GripPacketTests(unittest.TestCase):
    setUp = transport.TransportTests.setUp

    def test_gripper_load_returns_real_position_only_after_configuration(self):
        self.bus.packet.read2ByteTxRx.return_value = (1234, 0, 0x20)
        with self.assertRaises(HardwareError):
            self.bus.read("Present_Position", 6)
        self.bus.gripper_load_tolerance = True
        self.assertEqual(self.bus.read("Present_Position", 6), 1234)
        self.assertEqual(self.bus.motor_status[6], 0x20)
        self.bus.packet.read2ByteTxRx.return_value = (1280, 0, 0)
        self.assertEqual(self.bus.read("Present_Position", 6), 1280)
        self.assertEqual(self.bus.motor_status[6], 0)

    def test_other_joints_and_every_other_fault_bit_still_fail(self):
        self.bus.gripper_load_tolerance = True
        for motor_id, error in [(motor, 0x20) for motor in range(1, 6)] + [
            (6, flag) for flag in (1, 2, 4, 8, 16, 64, 128, 0x21, 0x24, 0x28)
        ]:
            with self.subTest(motor_id=motor_id, error=error):
                self.bus.packet.read2ByteTxRx.return_value = (1234, 0, error)
                with self.assertRaises(HardwareError):
                    self.bus.read("Present_Position", motor_id)

    def test_alarm_never_admits_short_corrupt_wrong_id_or_inconsistent_data(self):
        self.bus.gripper_load_tolerance = True
        good = transport.frame(6, (1234).to_bytes(2, "little"), 0x20)
        for invalid in (
            transport.frame(6, b"", 0x20),
            transport.frame(5, (1234).to_bytes(2, "little"), 0x20),
            [*good[:-1], good[-1] ^ 1],
            transport.frame(6, (1234).to_bytes(2, "little"), 0),
        ):
            with self.subTest(invalid=invalid):
                self.bus.packet.txRxPacket.side_effect = None
                self.bus.packet.txRxPacket.return_value = (invalid, 0, 0x20)
                with self.assertRaises(HardwareError):
                    self.bus.read("Present_Position", 6)

    def test_alarm_never_admits_a_transport_failure(self):
        self.bus.gripper_load_tolerance = True
        for result in (-3001, -3002, -1001):
            self.bus.packet.read2ByteTxRx.return_value = (1234, result, 0x20)
            with self.subTest(result=result), self.assertRaises(HardwareError):
                self.bus.read("Present_Position", 6)

    def test_gripper_torque_off_alarm_requires_exact_zero_readback_without_replay(self):
        self.bus.packet.write1ByteTxRx.return_value = (0, 0x20)
        self.bus.packet.read1ByteTxRx.return_value = (0, 0, 0x20)
        self.bus.write("Torque_Enable", 6, 0)
        self.bus.packet.write1ByteTxRx.assert_called_once_with(self.bus.port, 6, 40, 0)
        self.bus.packet.read1ByteTxRx.assert_called_once_with(self.bus.port, 6, 40)

    def test_torque_off_with_alarm_cannot_claim_a_missing_or_nonzero_readback(self):
        for returned in ((1, 0, 0x20), (0, -3001, 0x20), (0, 0, 0x24)):
            with self.subTest(returned=returned):
                self.bus.packet.reset_mock()
                self.bus.packet.write1ByteTxRx.return_value = (0, 0x20)
                self.bus.packet.read1ByteTxRx.return_value = returned
                with self.assertRaises(HardwareError):
                    self.bus.write("Torque_Enable", 6, 0)
                self.bus.packet.write1ByteTxRx.assert_called_once()

    def test_alarm_never_admits_a_malformed_torque_off_ack(self):
        for reply in (
            transport.frame(6, b"\x00", 0x20),
            transport.frame(5, b"", 0x20),
            transport.frame(6, b"", 0x24),
        ):
            with self.subTest(reply=reply):
                self.bus.packet.txRxPacket.side_effect = None
                self.bus.packet.txRxPacket.return_value = (reply, 0, 0x20)
                with self.assertRaises(HardwareError):
                    self.bus.write("Torque_Enable", 6, 0)

    def test_load_alarm_does_not_bypass_enabling_or_configuration(self):
        self.bus.gripper_load_tolerance = True
        for register, value in (("Torque_Enable", 1), ("Lock", 1), ("Protection_Current", 250)):
            self.bus.packet.write1ByteTxRx.return_value = (0, 0x20)
            self.bus.packet.write2ByteTxRx.return_value = (0, 0x20)
            with self.subTest(register=register), self.assertRaises(HardwareError):
                self.bus.write(register, 6, value)


class GripSessionTests(unittest.TestCase):
    setUp = fixtures.SafetyTests.setUp
    tearDown = fixtures.SafetyTests.tearDown
    start = fixtures.SafetyTests.start

    def test_grip_load_keeps_control_and_logs_transition_with_readings(self):
        self.start()
        bus = self.buses[-1]
        self.assertTrue(bus.gripper_load_tolerance)
        bus.follow_goals = False
        bus.raw["gripper"] = 1600
        bus.motor_status = {6: 0x20}
        generation = self.lab.generation
        self.lab.controller.command({"gripper": 0.0})
        for _ in range(30):
            self.lab._tick()
        self.assertEqual(self.lab.mode, "manual")
        self.assertEqual(self.lab.generation, generation)
        self.assertTrue(self.lab.torque)
        self.assertTrue(bus.opened)
        self.assertEqual(self.lab._diagnostic_context()["motor_alarms"], {"follower": {"6": 0x20}})
        events = self.lab.diagnostics.snapshot(full=True)["events"]
        loads = [event for event in events if event["event"] == "gripper_load_alarm"]
        self.assertEqual(len(loads), 1)
        self.assertEqual(loads[0]["context"]["raw_positions"]["follower"]["gripper"], 1600)
        self.assertIsNone(self.lab.diagnostics.snapshot()["last_stop"])
        bus.motor_status = {6: 0}
        self.lab._tick()
        events = self.lab.diagnostics.snapshot(full=True)["events"]
        self.assertEqual(sum(event["event"] == "gripper_load_cleared" for event in events), 1)
        self.lab.stop()
        self.assertFalse(bus.opened)
        self.assertFalse(self.lab.torque)

    def test_unverified_protection_settings_never_arm_load_tolerance(self):
        original = fixtures.FakeBus.read
        for setting in (
            "Max_Torque_Limit",
            "Protection_Current",
            "Overload_Torque",
            "Torque_Limit",
        ):

            def read(bus, register, motor_id):
                value = original(bus, register, motor_id)
                return value + 1 if register == setting and motor_id == 6 else value

            with self.subTest(setting=setting), patch.object(fixtures.FakeBus, "read", read):
                with self.assertRaises(HardwareError):
                    self.start()
                self.assertFalse(self.buses[-1].gripper_load_tolerance)
                self.assertFalse(
                    any(r == "Torque_Enable" and v == 1 for r, _, v in self.buses[-1].writes)
                )

    def test_fatal_motor_reply_records_source_and_status_for_next_incident(self):
        self.start()
        fault = HardwareError("Motor 6 overheated")
        fault.details = {"code": "motor_status_fault", "motor_id": 6, "status_error": 4}
        with patch.object(self.buses[-1], "positions", side_effect=fault):
            with self.assertRaises(HardwareError):
                self.lab._tick()
        self.lab.stop(str(fault))
        stop = self.lab.diagnostics.snapshot()["last_stop"]
        self.assertEqual(stop["context"]["fault"]["source"], "follower")
        self.assertEqual(stop["context"]["fault"]["status_error"], 4)
        self.assertEqual(stop["context"]["fault"]["motor_id"], 6)


if __name__ == "__main__":
    unittest.main()
