"""Packet loss recovery uses a fake SDK; no serial ports are opened."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hardware import REGISTERS, HardwareError, ServoBus, encode_sign


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.bus = ServoBus.__new__(ServoBus)
        self.bus.sdk = SimpleNamespace(COMM_SUCCESS=0, COMM_RX_TIMEOUT=-3001)
        self.bus.port = object()
        self.bus.packet = Mock()
        self.bus.packet.getTxRxResult.return_value = "No status packet"
        self.bus.packet.getRxPacketError.return_value = "Overload"
        self.sleep = patch("hardware.time.sleep")
        self.sleep.start()
        self.addCleanup(self.sleep.stop)

    def test_missing_read_reply_is_retried_then_decoded(self):
        read = self.bus.packet.read2ByteTxRx
        read.side_effect = [(0, -3001, 0), (encode_sign(-85), 0, 0)]
        self.assertEqual(self.bus.read("Homing_Offset", 1), -85)
        self.assertEqual(read.call_count, 2)
        self.bus.packet.write2ByteTxRx.assert_not_called()

    def test_read_timeout_is_bounded_and_names_motor_and_register(self):
        read = self.bus.packet.read2ByteTxRx
        read.return_value = (0, -3001, 0)
        with self.assertRaisesRegex(HardwareError, "Motor 2 read Present_Position"):
            self.bus.read("Present_Position", 2)
        self.assertEqual(read.call_count, 3)

    def test_corrupt_read_or_motor_fault_is_not_retried(self):
        for result, error in ((-3002, 0), (0, 32), (-3001, 32)):
            with self.subTest(result=result, error=error):
                read = self.bus.packet.read2ByteTxRx
                read.reset_mock()
                read.return_value = (0, result, error)
                with self.assertRaises(HardwareError):
                    self.bus.read("Present_Position", 1)
                read.assert_called_once()

    def test_lost_configuration_ack_requires_exact_readback_and_never_rewrites(self):
        for register, requested in (
            ("Homing_Offset", -85),
            ("Min_Position_Limit", 100),
            ("Max_Position_Limit", 3900),
            ("Lock", 1),
            ("Operating_Mode", 0),
        ):
            with self.subTest(register=register):
                self.bus.packet.reset_mock()
                address, size = REGISTERS[register]
                write = (
                    self.bus.packet.write1ByteTxRx if size == 1 else self.bus.packet.write2ByteTxRx
                )
                read = self.bus.packet.read1ByteTxRx if size == 1 else self.bus.packet.read2ByteTxRx
                raw = encode_sign(requested) if register == "Homing_Offset" else requested
                write.return_value = (-3001, 0)
                read.return_value = (raw, 0, 0)
                self.bus.write(register, 1, requested)
                write.assert_called_once_with(self.bus.port, 1, address, raw)
                read.assert_called_once_with(self.bus.port, 1, address)

    def test_configuration_mismatch_fails_without_write_replay(self):
        self.bus.packet.write2ByteTxRx.return_value = (-3001, 0)
        self.bus.packet.read2ByteTxRx.return_value = (2, 0, 0)
        with self.assertRaisesRegex(HardwareError, "read-back was 2, expected 0"):
            self.bus.write("Homing_Offset", 1, 0)
        self.bus.packet.write2ByteTxRx.assert_called_once()

    def test_configuration_missing_readback_fails_without_write_replay(self):
        self.bus.packet.write2ByteTxRx.return_value = (-3001, 0)
        self.bus.packet.read2ByteTxRx.return_value = (0, -3001, 0)
        with self.assertRaisesRegex(HardwareError, "write Homing_Offset.*read-back failed"):
            self.bus.write("Homing_Offset", 1, 0)
        self.bus.packet.write2ByteTxRx.assert_called_once()
        self.assertEqual(self.bus.packet.read2ByteTxRx.call_count, 3)

    def test_motion_torque_and_id_writes_never_use_ack_recovery(self):
        for register, value in (
            ("Goal_Position", 2047),
            ("Torque_Enable", 1),
            ("Torque_Enable", 0),
            ("ID", 2),
            ("Baud_Rate", 0),
        ):
            with self.subTest(register=register, value=value):
                self.bus.packet.reset_mock()
                write = (
                    self.bus.packet.write1ByteTxRx
                    if REGISTERS[register][1] == 1
                    else self.bus.packet.write2ByteTxRx
                )
                write.return_value = (-3001, 0)
                with self.assertRaises(HardwareError):
                    self.bus.write(register, 1, value)
                write.assert_called_once()
                self.bus.packet.read1ByteTxRx.assert_not_called()
                self.bus.packet.read2ByteTxRx.assert_not_called()

    def test_acknowledged_configuration_write_does_not_need_recovery(self):
        self.bus.packet.write2ByteTxRx.return_value = (0, 0)
        self.bus.write("Homing_Offset", 1, 0)
        self.bus.packet.write2ByteTxRx.assert_called_once()
        self.bus.packet.read2ByteTxRx.assert_not_called()

    def test_write_fault_or_corrupt_reply_never_uses_readback_recovery(self):
        for result, error in ((-3002, 0), (0, 32), (-3001, 32)):
            with self.subTest(result=result, error=error):
                self.bus.packet.reset_mock()
                self.bus.packet.write2ByteTxRx.return_value = (result, error)
                with self.assertRaises(HardwareError):
                    self.bus.write("Homing_Offset", 1, 0)
                self.bus.packet.write2ByteTxRx.assert_called_once()
                self.bus.packet.read2ByteTxRx.assert_not_called()


if __name__ == "__main__":
    unittest.main()
