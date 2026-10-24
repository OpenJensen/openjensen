"""Packet loss recovery uses a fake SDK; no serial ports are opened."""

import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hardware import REGISTERS, HardwareError, ServoBus, encode_sign


def frame(sid, data, error=0):
    packet = [255, 255, sid, len(data) + 2, error, *data]
    return [*packet, ~sum(packet[2:]) & 255]


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.bus = ServoBus.__new__(ServoBus)
        self.bus.sdk = SimpleNamespace(COMM_SUCCESS=0, COMM_RX_TIMEOUT=-3001, COMM_RX_CORRUPT=-3002)
        self.bus.port = SimpleNamespace(ser=Mock())
        self.bus.packet = Mock()

        # Existing tuple-based fault cases feed raw frames into the adapter.
        def exchange(port, request):
            sid, address = request[2], request[5]
            if request[4] == 3:
                size = request[3] - 3
                source = (
                    self.bus.packet.write1ByteTxRx if size == 1 else self.bus.packet.write2ByteTxRx
                )
                result, error = source(port, sid, address, int.from_bytes(request[6:-1], "little"))
                return frame(sid, b"", error), result, error
            size = request[6]
            source = self.bus.packet.read1ByteTxRx if size == 1 else self.bus.packet.read2ByteTxRx
            value, result, error = source(port, sid, address)
            return frame(sid, value.to_bytes(size, "little"), error), result, error

        self.bus.packet.txRxPacket.side_effect = exchange
        self.bus.packet.getTxRxResult.return_value = "No status packet"
        self.bus.packet.getRxPacketError.return_value = "Overload"
        self.sleep = patch("hardware.time.sleep")
        self.sleep.start()
        self.addCleanup(self.sleep.stop)

    def test_selected_bus_reads_and_stop_only_address_selected_ids(self):
        self.bus.joints = ("shoulder_pan", "elbow_flex")
        self.bus.lock = threading.RLock()
        self.bus.read = Mock(return_value=2047)
        self.bus.write = Mock()
        self.assertEqual(set(self.bus.positions()), set(self.bus.joints))
        self.assertEqual(set(self.bus.read_calibration()), set(self.bus.joints))
        self.assertEqual({call.args[1] for call in self.bus.read.call_args_list}, {1, 3})
        self.bus.disable()
        self.assertEqual(self.bus.write.call_args_list, [
            unittest.mock.call("Torque_Enable", 1, 0),
            unittest.mock.call("Torque_Enable", 3, 0),
        ])

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

    def test_transmit_failure_or_motor_fault_is_not_retried(self):
        for result, error in ((-1001, 0), (0, 32), (-3001, 32), (-3002, 32)):
            with self.subTest(result=result, error=error):
                read = self.bus.packet.read2ByteTxRx
                read.reset_mock()
                read.return_value = (0, result, error)
                with self.assertRaises(HardwareError):
                    self.bus.read("Present_Position", 1)
                read.assert_called_once()

    def test_incomplete_or_corrupt_read_discards_value_and_recovers_fresh_sample(self):
        read = self.bus.packet.read2ByteTxRx
        read.side_effect = [(65000, -3002, 0), (4095, -3001, 0), (2047, 0, 0)]
        self.assertEqual(self.bus.read("Present_Position", 2), 2047)
        self.assertEqual(read.call_count, 3)
        self.assertEqual(self.bus.port.ser.reset_input_buffer.call_count, 2)
        self.bus.packet.write1ByteTxRx.assert_not_called()
        self.bus.packet.write2ByteTxRx.assert_not_called()

    def test_persistent_corrupt_reads_are_bounded_and_fail(self):
        read = self.bus.packet.read2ByteTxRx
        read.return_value = (2047, -3002, 0)
        with self.assertRaisesRegex(HardwareError, "Motor 2 read Present_Position"):
            self.bus.read("Present_Position", 2)
        self.assertEqual(read.call_count, 3)
        self.assertEqual(self.bus.port.ser.reset_input_buffer.call_count, 2)

    def test_short_ack_is_never_decoded_as_register_data(self):
        for register, size in (("Lock", 1), ("Present_Position", 2)):
            with self.subTest(register=register):
                exchange = self.bus.packet.txRxPacket
                exchange.reset_mock()
                expected = 1 if size == 1 else 2047
                exchange.side_effect = [
                    (frame(2, b""), 0, 0),
                    (frame(2, expected.to_bytes(size, "little")), 0, 0),
                ]
                self.assertEqual(self.bus.read(register, 2), expected)
                self.assertEqual(exchange.call_count, 2)
                self.assertEqual(
                    exchange.call_args.args[1], [0, 0, 2, 4, 2, REGISTERS[register][0], size, 0]
                )
        self.bus.packet.write1ByteTxRx.assert_not_called()
        self.bus.packet.write2ByteTxRx.assert_not_called()

    def test_persistent_short_ack_is_a_named_hardware_error(self):
        self.bus.packet.txRxPacket.side_effect = None
        self.bus.packet.txRxPacket.return_value = (frame(6, b""), 0, 0)
        with self.assertRaisesRegex(HardwareError, "Motor 6 read Present_Position"):
            self.bus.read("Present_Position", 6)
        self.assertEqual(self.bus.packet.txRxPacket.call_count, 3)

    def test_wrong_id_length_checksum_or_parser_error_is_discarded(self):
        good = frame(1, (2047).to_bytes(2, "little"))
        bad_checksum = [*good[:-1], good[-1] ^ 1]
        bad_length = list(good)
        bad_length[3] = 3
        for invalid in (
            frame(2, b"\xff\x07"),
            frame(1, b"\xff"),
            bad_checksum,
            bad_length,
            IndexError("short packet"),
        ):
            with self.subTest(invalid=invalid):
                exchange = self.bus.packet.txRxPacket
                exchange.reset_mock()
                exchange.side_effect = [
                    invalid if isinstance(invalid, Exception) else (invalid, 0, 0),
                    (good, 0, 0),
                ]
                self.assertEqual(self.bus.read("Present_Position", 1), 2047)
                self.assertEqual(exchange.call_count, 2)

    def test_lost_configuration_ack_requires_exact_readback_and_never_rewrites(self):
        for register, requested in (
            ("Homing_Offset", -85),
            ("Min_Position_Limit", 100),
            ("Max_Position_Limit", 3900),
            ("Lock", 1),
            ("Operating_Mode", 0),
            ("Max_Torque_Limit", 500),
            ("Protection_Current", 250),
            ("Overload_Torque", 25),
            ("Acceleration", 10),
            ("P_Coefficient", 16),
            ("I_Coefficient", 0),
            ("D_Coefficient", 32),
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

    def test_protection_mismatch_fails_without_write_replay(self):
        for register, requested in (
            ("Max_Torque_Limit", 500),
            ("Protection_Current", 250),
            ("Overload_Torque", 25),
        ):
            with self.subTest(register=register):
                self.bus.packet.reset_mock()
                size = REGISTERS[register][1]
                write = (
                    self.bus.packet.write1ByteTxRx if size == 1 else self.bus.packet.write2ByteTxRx
                )
                read = self.bus.packet.read1ByteTxRx if size == 1 else self.bus.packet.read2ByteTxRx
                write.return_value = (-3001, 0)
                read.return_value = (requested + 1, 0, 0)
                with self.assertRaisesRegex(HardwareError, f"write {register}.*read-back was"):
                    self.bus.write(register, 6, requested)
                write.assert_called_once()

    def test_configuration_missing_readback_fails_without_write_replay(self):
        self.bus.packet.write2ByteTxRx.return_value = (-3001, 0)
        self.bus.packet.read2ByteTxRx.return_value = (0, -3001, 0)
        with self.assertRaisesRegex(HardwareError, "write Homing_Offset.*read-back failed"):
            self.bus.write("Homing_Offset", 1, 0)
        self.bus.packet.write2ByteTxRx.assert_called_once()
        self.assertEqual(self.bus.packet.read2ByteTxRx.call_count, 3)

    def test_target_velocity_and_identity_writes_never_use_ack_recovery(self):
        for register, value in (
            ("Goal_Position", 2047),
            ("Goal_Velocity", 100),
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

    def test_torque_ack_loss_or_corruption_requires_exact_fresh_state_without_replay(self):
        for result in (-3001, -3002):
            for requested in (0, 1):
                with self.subTest(result=result, requested=requested):
                    self.bus.packet.reset_mock()
                    self.bus.packet.write1ByteTxRx.return_value = (result, 0)
                    self.bus.packet.read1ByteTxRx.return_value = (requested, 0, 0)
                    self.bus.write("Torque_Enable", 1, requested)
                    self.bus.packet.write1ByteTxRx.assert_called_once_with(
                        self.bus.port, 1, 40, requested
                    )
                    self.bus.packet.read1ByteTxRx.assert_called_once_with(self.bus.port, 1, 40)

    def test_torque_mismatch_missing_read_or_motor_fault_stops_without_replay(self):
        for requested in (0, 1):
            for read_result in ((1 - requested, 0, 0), (0, -3001, 0), (requested, 0, 32)):
                with self.subTest(requested=requested, read_result=read_result):
                    self.bus.packet.reset_mock()
                    self.bus.packet.write1ByteTxRx.return_value = (-3002, 0)
                    self.bus.packet.read1ByteTxRx.return_value = read_result
                    with self.assertRaises(HardwareError):
                        self.bus.write("Torque_Enable", 1, requested)
                    self.bus.packet.write1ByteTxRx.assert_called_once()

    def test_even_valid_torque_ack_requires_matching_motor_state(self):
        self.bus.packet.write1ByteTxRx.return_value = (0, 0)
        self.bus.packet.read1ByteTxRx.return_value = (0, 0, 0)
        with self.assertRaisesRegex(HardwareError, "torque read-back did not match"):
            self.bus.write("Torque_Enable", 1, 1)
        self.bus.packet.write1ByteTxRx.assert_called_once()
        self.bus.packet.read1ByteTxRx.assert_called_once()

    def test_stale_read_response_cannot_acknowledge_a_position_command(self):
        self.bus.packet.txRxPacket.side_effect = None
        self.bus.packet.txRxPacket.return_value = (frame(1, b"\xff\x07"), 0, 0)
        with self.assertRaisesRegex(HardwareError, "write Goal_Position=2047"):
            self.bus.write("Goal_Position", 1, 2047)
        self.bus.packet.txRxPacket.assert_called_once_with(
            self.bus.port, [0, 0, 1, 5, 3, 42, 255, 7, 0]
        )

    def test_corrupt_short_or_wrong_id_ack_does_not_replay_position_command(self):
        good = frame(1, b"")
        for reply in (good[:-1], [*good[:-1], good[-1] ^ 1], frame(2, b"")):
            with self.subTest(reply=reply):
                self.bus.packet.txRxPacket.reset_mock()
                self.bus.packet.txRxPacket.side_effect = None
                self.bus.packet.txRxPacket.return_value = (reply, 0, 0)
                with self.assertRaises(HardwareError):
                    self.bus.write("Goal_Position", 1, 2047)
                self.bus.packet.txRxPacket.assert_called_once()

    def test_acknowledged_configuration_write_does_not_need_recovery(self):
        self.bus.packet.write2ByteTxRx.return_value = (0, 0)
        self.bus.write("Homing_Offset", 1, 0)
        self.bus.packet.write2ByteTxRx.assert_called_once()
        self.bus.packet.read2ByteTxRx.assert_not_called()

    def test_write_motor_fault_or_transmit_error_never_uses_readback_recovery(self):
        for result, error in ((-1001, 0), (-3002, 32), (0, 32), (-3001, 32)):
            with self.subTest(result=result, error=error):
                self.bus.packet.reset_mock()
                self.bus.packet.write2ByteTxRx.return_value = (result, error)
                with self.assertRaises(HardwareError):
                    self.bus.write("Homing_Offset", 1, 0)
                self.bus.packet.write2ByteTxRx.assert_called_once()
                self.bus.packet.read2ByteTxRx.assert_not_called()


if __name__ == "__main__":
    unittest.main()
