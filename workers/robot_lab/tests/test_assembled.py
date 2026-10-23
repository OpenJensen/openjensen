"""No hardware is opened: deterministic protocol fakes for assembled setup."""

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _vendor.servotools import registers as reg
from _vendor.servotools.framing import packet
from assembled import (
    AssembledCancelled,
    AssembledError,
    AssembledSetup,
    SerialTransport,
    detect_movement,
)
from hardware import JOINTS, HardwareError, ServoBus


def frames(message):
    offset = 0
    while offset < len(message):
        end = offset + 4 + message[offset + 3]
        yield message[offset + 2], message[offset + 4], message[offset + 5 : end - 1]
        offset = end


class Clock:
    def __init__(self):
        self.value = 100.0

    def monotonic(self):
        self.value += 0.01
        return self.value


class FakeTransport:
    def __init__(self, ids=None):
        self.motors = [
            dict(id=sid, saved=sid, lock=1, torque=0, pos=1000 + index * 120, model=777)
            for index, sid in enumerate(ids or [1] * 6)
        ]
        self.writes = []
        self.closed = False
        self.reboots = 0
        self.timeout = 0.006
        self.exchanges = []
        self.no_wake = False
        self.status_error = 0
        self.reboot_torque = 0
        self.split_enabled = True
        self.callback = None
        self.fail_after_id = False
        self.fail_lock_readback = False

    def exchange(self, request, wait, length):
        self.exchanges.append((wait, length, self.timeout))
        sid, instruction, params = next(frames(request))
        if self.no_wake and self.reboots:
            return b""
        motors = [motor for motor in self.motors if motor["id"] == sid]
        if not motors:
            return b""
        if len(motors) > 1:
            return b"COLLIDING-STATUS"
        motor = motors[0]
        address, width = params
        values = {
            reg.MODEL: motor["model"],
            reg.ID: motor["id"],
            reg.LOCK: motor["lock"],
            reg.TORQUE_ENABLE: motor["torque"],
            reg.PRESENT_POSITION: motor["pos"],
        }
        value = values[address]
        if self.fail_lock_readback and address == reg.LOCK:
            value = 0
        return packet(sid, self.status_error, value.to_bytes(width, "little"))

    def send(self, message):
        decoded = list(frames(message))
        self.writes.extend(decoded)
        if len(message) > 16:
            # Model different wake windows; no inference about real host timing.
            if self.split_enabled:
                source = decoded[0][0]
                waiting = [motor for motor in self.motors if motor["id"] == source]
                targets = [params[1] for _, _, params in decoded if params[0] == reg.ID]
                for motor, target in zip(waiting, targets):
                    motor.update(id=target, lock=0)
            return
        for sid, instruction, params in decoded:
            if instruction == 0x08:
                self.reboots += 1
                for motor in self.motors:
                    motor.update(id=motor["saved"], torque=self.reboot_torque, lock=1)
                if self.callback:
                    self.callback(sid, instruction, params)
                continue
            address, value = params
            for motor in [entry for entry in self.motors if sid == 254 or entry["id"] == sid]:
                if address == reg.TORQUE_ENABLE:
                    motor["torque"] = value
                elif address == reg.LOCK:
                    if value == 1:
                        motor["saved"] = motor["id"]
                    motor["lock"] = value
                elif address == reg.ID and motor["lock"] == 0:
                    motor["id"] = value
            if self.callback:
                self.callback(sid, instruction, params)
            if self.fail_after_id and address == reg.ID:
                self.fail_after_id = False
                raise OSError("USB disconnected after accepting ID")

    def flush(self):
        pass

    def close(self):
        self.closed = True


class FastSetup(AssembledSetup):
    def _sleep(self, seconds):
        self._check()

    def _wait_for_boot_cleanup(self, seconds):
        self.cleanup_wait = max(0, seconds)


class AssembledTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="assembled-test-")
        self.transport = FakeTransport()
        self.stop = threading.Event()
        self.clock = Clock()
        self.time_patch = patch("assembled.time.monotonic", self.clock.monotonic)
        self.time_patch.start()
        self.open_count = 0

        def factory(port):
            self.open_count += 1
            return self.transport

        self.setup = FastSetup(
            "/dev/cu.fake",
            self.temp.name,
            stop_event=self.stop,
            io_lock=threading.RLock(),
            bus_factory=factory,
            timeout_s=600,
            verified_so101=True,
        )

    def tearDown(self):
        self.setup.close()
        self.time_patch.stop()
        self.temp.cleanup()

    def ready(self):
        self.transport = FakeTransport([11, 12, 13, 14, 15, 16])
        self.setup.bus_factory = lambda port: self.transport
        return self.setup.split()

    def test_constructor_opens_nothing_and_cancel_before_start_writes_nothing(self):
        self.assertEqual(self.open_count, 0)
        self.stop.set()
        with self.assertRaises(AssembledCancelled):
            self.setup.split()
        self.assertEqual(self.transport.writes, [])
        self.assertEqual(self.open_count, 0)

    def test_full_split_maps_six_with_no_motion_packets(self):
        ids = self.setup.split()
        self.assertEqual(len(ids), 6)
        self.assertEqual(len(set(ids)), 6)
        self.assertEqual(self.setup.snapshot()["phase"], "ready")
        self.assertGreaterEqual(self.transport.reboots, 4)
        self.assertTrue(
            all(
                motor["lock"] == 1 and motor["saved"] == motor["id"]
                for motor in self.transport.motors
            )
        )
        for _, instruction, params in self.transport.writes:
            self.assertIn(instruction, (0x03, 0x08))
            if instruction == 0x03:
                self.assertIn(params[0], (reg.ID, reg.LOCK, reg.TORQUE_ENABLE))
                if params[0] == reg.TORQUE_ENABLE:
                    self.assertEqual(params[1], 0)

    def test_already_distinct_chain_only_reads_and_releases_torque(self):
        ids = self.ready()
        self.assertEqual(ids, [11, 12, 13, 14, 15, 16])
        self.assertEqual(self.transport.reboots, 0)
        self.assertTrue(
            all(params == bytes([reg.TORQUE_ENABLE, 0]) for _, _, params in self.transport.writes)
        )

    def test_fast_wake_poll_restores_normal_timeout(self):
        self.setup.split()
        self.assertIn((0, 8, 0.001), self.transport.exchanges)
        self.assertEqual(self.transport.timeout, 0.006)

    def test_unidentified_collision_requires_model_attestation_before_reboot(self):
        self.setup.verified_so101 = False
        with self.assertRaisesRegex(AssembledError, "could not identify"):
            self.setup.split()
        self.assertEqual(self.transport.reboots, 0)
        self.assertEqual(self.transport.writes, [])

    def test_inventory_change_including_id_zero_prevents_commit(self):
        self.ready()
        self.transport.motors.append(dict(id=0, saved=0, lock=1, torque=0, pos=1500, model=777))
        count = len(self.transport.writes)
        with self.assertRaisesRegex(AssembledError, "inventory changed"):
            self.setup.apply(dict(zip(JOINTS, [11, 12, 13, 14, 15, 16], strict=True)))
        self.assertEqual(len(self.transport.writes), count)

    def test_journal_sync_failure_prevents_any_rename_packet(self):
        self.ready()
        count = len(self.transport.writes)
        with patch("assembled.os.fsync", side_effect=OSError("disk sync failed")):
            with self.assertRaisesRegex(OSError, "disk sync failed"):
                self.setup._rename(11, 100)
        self.assertEqual(len(self.transport.writes), count)

    def test_destination_is_checked_before_each_rename(self):
        self.ready()
        self.transport.motors.append(dict(id=100, saved=100, lock=1, torque=0, pos=1500, model=777))
        count = len(self.transport.writes)
        with self.assertRaisesRegex(AssembledError, "occupied"):
            self.setup._rename(11, 100)
        self.assertEqual(len(self.transport.writes), count)

    def test_valid_servo_error_is_not_treated_as_a_collision(self):
        self.transport = FakeTransport([1, 2, 3, 4, 5, 6])
        self.transport.status_error = 4
        self.setup.bus_factory = lambda port: self.transport
        with self.assertRaisesRegex(AssembledError, "status error"):
            self.setup.split()
        self.assertEqual(self.transport.writes, [])

    def test_torque_is_explicitly_disabled_after_wake_and_confirmed(self):
        self.transport.reboot_torque = 1
        self.setup.split()
        self.assertTrue(all(motor["torque"] == 0 for motor in self.transport.motors))

    def test_no_wake_measurement_has_no_blind_timing_or_id_writes(self):
        self.transport.no_wake = True
        with self.assertRaisesRegex(AssembledError, "Cannot measure"):
            self.setup.split()
        self.assertFalse(
            any(
                instruction == 0x03 and params[0] == reg.ID
                for _, instruction, params in self.transport.writes
            )
        )
        self.assertEqual(self.setup.snapshot()["phase"], "error")

    def test_unseparable_firmware_is_bounded_and_journaled(self):
        self.transport.split_enabled = False
        with self.assertRaisesRegex(AssembledError, "bounded attempts"):
            self.setup.split()
        self.assertEqual(self.setup.snapshot()["round"], 12)
        self.assertTrue(self.setup.journal_path.is_file())

    def test_collision_ram_address_is_rescanned_after_reboot_before_next_sweep(self):
        transport = self.transport
        original_send = transport.send
        sources = []

        def send(message):
            decoded = list(frames(message))
            if len(message) > 16:
                source = decoded[0][0]
                sources.append(source)
                if len(sources) == 1:
                    # A late window catches the complete group at the first
                    # temporary address, but never commits its EEPROM IDs.
                    target = decoded[1][2][1]
                    for motor in transport.motors:
                        if motor["id"] == source:
                            motor.update(id=target, lock=0)
                    transport.writes.extend(decoded)
                    return
            original_send(message)

        transport.send = send
        self.assertEqual(len(self.setup.split()), 6)
        self.assertEqual(sources, [1, 1])
        self.assertEqual(self.setup.snapshot()["next_lead_ms"], 10.0)
        self.assertEqual(len(self.setup.snapshot()["boot_ids"]), 6)
        self.assertTrue(all(motor["id"] == motor["saved"] for motor in transport.motors))

    def test_model_mismatch_aborts_before_any_writes(self):
        self.transport = FakeTransport([1, 2, 3, 4, 5, 6])
        self.transport.motors[0]["model"] = 123
        self.setup.bus_factory = lambda port: self.transport
        with self.assertRaisesRegex(AssembledError, "not an STS3215"):
            self.setup.split()
        self.assertEqual(self.transport.writes, [])

    def test_collision_free_reviewed_mapping_persists_after_reboot(self):
        ids = self.ready()
        mapping = dict(zip(JOINTS, reversed(ids), strict=True))
        result = self.setup.apply(mapping)
        self.assertEqual(result, [1, 2, 3, 4, 5, 6])
        self.assertEqual(self.setup.snapshot()["phase"], "complete")
        self.assertEqual([motor["saved"] for motor in self.transport.motors], [6, 5, 4, 3, 2, 1])
        self.assertTrue(
            all(motor["lock"] == 1 and motor["torque"] == 0 for motor in self.transport.motors)
        )
        self.assertEqual(self.transport.reboots, 1)

    def test_duplicate_or_incomplete_mapping_rejected_before_id_writes(self):
        self.ready()
        count = len(self.transport.writes)
        with self.assertRaises(AssembledError):
            self.setup.apply(dict.fromkeys(JOINTS, 11))
        self.assertEqual(len(self.transport.writes), count)

    def test_cancel_mid_rename_only_finishes_lock_cleanup(self):
        self.ready()
        trigger = []

        def callback(sid, instruction, params):
            if instruction == 0x03 and params[0] == reg.ID and not trigger:
                trigger.append(len(self.transport.writes))
                self.stop.set()

        self.transport.callback = callback
        with self.assertRaises(AssembledCancelled):
            self.setup.apply(dict(zip(JOINTS, [11, 12, 13, 14, 15, 16], strict=True)))
        after = self.transport.writes[trigger[0] :]
        self.assertTrue(after)
        self.assertTrue(
            all(
                instruction == 0x03 and params == bytes([reg.LOCK, 1])
                for _, instruction, params in after
            )
        )
        journal = json.loads(self.setup.journal_path.read_text())
        self.assertEqual(journal["phase"], "cancelled")
        self.assertEqual(journal["pending"]["possible_ids"], [11, 100])
        self.assertTrue(self.transport.motors[0]["lock"] == 1)

    def test_uncertain_id_write_relocks_both_addresses_and_keeps_recovery_journal(self):
        self.ready()
        self.transport.fail_after_id = True
        with self.assertRaises(OSError):
            self.setup.apply(dict(zip(JOINTS, [11, 12, 13, 14, 15, 16], strict=True)))
        journal = json.loads(self.setup.journal_path.read_text())
        self.assertEqual(journal["pending"]["possible_ids"], [11, 100])
        locks = [
            sid
            for sid, instruction, params in self.transport.writes
            if instruction == 0x03 and params == bytes([reg.LOCK, 1])
        ]
        self.assertIn(11, locks)
        self.assertIn(100, locks)

    def test_failed_lock_readback_never_claims_success(self):
        self.ready()
        self.transport.fail_lock_readback = True
        with self.assertRaisesRegex(AssembledError, "read-back"):
            self.setup.apply(dict(zip(JOINTS, [11, 12, 13, 14, 15, 16], strict=True)))
        self.assertEqual(self.setup.snapshot()["phase"], "error")

    def test_cancel_during_reboot_waits_then_only_disables_torque_before_close(self):
        self.transport.reboot_torque = 1

        def callback(sid, instruction, params):
            if instruction == 0x08:
                self.stop.set()

        self.transport.callback = callback
        with self.assertRaises(AssembledCancelled):
            self.setup.split()
        self.assertTrue(all(motor["torque"] == 1 for motor in self.transport.motors))
        self.assertIsNotNone(self.setup.reboot_cleanup_at)
        count = len(self.transport.writes)
        waits = []

        def wait(seconds):
            waits.append((self.setup.io_lock._is_owned(), seconds))

        with patch.object(self.setup, "_wait_for_boot_cleanup", side_effect=wait):
            self.setup.close()
        self.assertEqual(len(waits), 1)
        self.assertTrue(waits[0][0])
        self.assertGreater(waits[0][1], 0)
        self.assertLessEqual(waits[0][1], 4)
        self.assertEqual(self.transport.writes[count:], [(254, 3, bytes([reg.TORQUE_ENABLE, 0]))])
        self.assertTrue(all(motor["torque"] == 0 for motor in self.transport.motors))
        self.assertTrue(self.transport.closed)
        cleanup = json.loads(self.setup.journal_path.read_text())["reboot_cleanup"]
        self.assertTrue(cleanup["attempted"])
        self.assertFalse(cleanup["verified_off"])
        self.assertFalse(cleanup["pending"])

    def test_post_boot_cleanup_failure_is_journaled_and_closes_before_error(self):
        self.setup._open()
        self.setup._reboot()
        self.stop.set()
        with patch.object(self.transport, "send", side_effect=OSError("adapter disconnected")):
            with self.assertRaisesRegex(AssembledError, "cleanup failed"):
                self.setup.close()
        self.assertTrue(self.transport.closed)
        self.assertIsNone(self.setup.bus)
        cleanup = json.loads(self.setup.journal_path.read_text())["reboot_cleanup"]
        self.assertTrue(cleanup["pending"])
        self.assertIn("adapter disconnected", cleanup["error"])

    def test_boot_cleanup_wait_ignores_cancellation_for_bounded_final_off(self):
        self.stop.set()
        with patch("assembled.time.sleep") as sleep:
            AssembledSetup._wait_for_boot_cleanup(self.setup, 3.9)
        sleep.assert_called_once_with(3.9)

    def test_uart_burst_holds_stop_lock_for_nominal_wire_drain(self):
        self.setup._open()
        burst = b"".join(
            packet(1, 3, bytes([reg.LOCK, 0])) + packet(1, 3, bytes([reg.ID, sid]))
            for sid in range(2, 86)
        )
        owned = []

        def drain(seconds):
            owned.append((self.setup.io_lock._is_owned(), seconds))

        with patch("assembled.time.sleep", side_effect=drain):
            self.setup._send(burst)
        self.assertEqual(len(owned), 1)
        self.assertTrue(owned[0][0])
        self.assertGreaterEqual(owned[0][1], len(burst) * 10 / 1_000_000 + 0.005)

    def test_strict_packet_allowlist_rejects_goals_torque_on_and_other_settings(self):
        for message in (
            packet(1, 3, bytes([reg.GOAL_POSITION, 1])),
            packet(1, 3, bytes([reg.TORQUE_ENABLE, 1])),
            packet(1, 3, bytes([reg.P_COEFF, 16])),
            packet(1, 0x0A),
            packet(1, 0x08),
        ):
            with self.assertRaises(AssembledError):
                self.setup._validate(message)

    def test_cancellation_cleanup_cannot_bypass_id_write_guard(self):
        self.setup._open()
        self.stop.set()
        for message in (packet(1, 3, bytes([reg.ID, 2])), packet(1, 3, bytes([reg.LOCK, 0]))):
            with self.assertRaisesRegex(AssembledError, "cleanup permits only"):
                self.setup._send(message, cleanup=True)
        self.assertEqual(self.transport.writes, [])

    def test_constructor_resumes_diagnostic_journal_without_opening_or_writing(self):
        self.ready()
        self.setup.close()
        previous_writes = len(self.transport.writes)
        other = FastSetup(
            "/dev/cu.fake",
            self.temp.name,
            bus_factory=lambda port: self.fail("constructor opened serial"),
        )
        self.assertEqual(other.snapshot()["previous"]["ids"], [11, 12, 13, 14, 15, 16])
        self.assertEqual(len(self.transport.writes), previous_writes)

    def test_cancellation_during_open_closes_handle_before_publishing(self):
        def factory(port):
            self.stop.set()
            return self.transport

        self.setup.bus_factory = factory
        with self.assertRaises(AssembledCancelled):
            self.setup.split()
        self.assertTrue(self.transport.closed)
        self.assertIsNone(self.setup.bus)

    def test_close_is_read_only(self):
        self.ready()
        count = len(self.transport.writes)
        self.setup.close()
        self.assertEqual(len(self.transport.writes), count)
        self.assertTrue(self.transport.closed)


class CalibrationPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.bus = object.__new__(ServoBus)
        self.values = {}
        self.saved = {}
        self.writes = []
        self.failure = None

        def write(register, sid, value):
            self.writes.append((register, sid, value))
            if register == self.failure:
                raise HardwareError("simulated calibration transport failure")
            self.values[(register, sid)] = value
            if register == "Lock" and value == 1:
                self.saved.update(self.values)

        self.bus.write = write
        self.calibration = {
            joint: {"homing_offset": index * 10, "range_min": 800, "range_max": 3200}
            for index, joint in enumerate(JOINTS)
        }

    def test_each_joint_calibration_is_persisted_before_next_joint(self):
        self.bus.write_calibration(self.calibration)
        for index, joint in enumerate(JOINTS):
            sid = index + 1
            self.assertEqual(self.writes[index * 5], ("Lock", sid, 0))
            self.assertEqual(self.writes[index * 5 + 4], ("Lock", sid, 1))
            self.assertEqual(
                self.saved[("Homing_Offset", sid)], self.calibration[joint]["homing_offset"]
            )
            self.assertEqual(self.saved[("Min_Position_Limit", sid)], 800)
            self.assertEqual(self.saved[("Max_Position_Limit", sid)], 3200)

    def test_failed_field_or_uncertain_unlock_still_relocks_before_raising(self):
        for register in ("Lock", "Homing_Offset", "Min_Position_Limit", "Max_Position_Limit"):
            with self.subTest(register=register):
                self.writes.clear()
                self.failure = register
                with self.assertRaises(HardwareError):
                    self.bus.write_calibration(self.calibration)
                self.assertEqual(self.writes[-1], ("Lock", 1, 1))
                self.assertFalse(any(sid != 1 for _, sid, _ in self.writes))


class ManualMappingTests(unittest.TestCase):
    def setUp(self):
        self.baseline = dict.fromkeys(range(11, 17), 2000)

    def samples(self, changes, now=10):
        return [
            (
                now - 0.3 + index * 0.1,
                {sid: (self.baseline[sid] + changes.get(sid, 0)) % 4096 for sid in self.baseline},
            )
            for index in range(4)
        ]

    def test_dominant_manual_motion_requires_four_fresh_samples(self):
        samples = self.samples({12: 200, 13: 3})
        self.assertEqual(detect_movement(self.baseline, samples, now=10), 12)
        self.assertIsNone(detect_movement(self.baseline, samples[:3], now=10))

    def test_neighbor_noise_multiple_motion_and_assigned_ids_rejected(self):
        self.assertIsNone(detect_movement(self.baseline, self.samples({12: 30}), now=10))
        self.assertIsNone(detect_movement(self.baseline, self.samples({12: 200, 13: 180}), now=10))
        self.assertIsNone(detect_movement(self.baseline, self.samples({12: 81, 13: 79}), now=10))
        self.assertIsNone(
            detect_movement(self.baseline, self.samples({12: 200}), assigned=[12], now=10)
        )

    def test_stale_unordered_missing_and_invalid_samples_rejected(self):
        self.assertIsNone(detect_movement(self.baseline, self.samples({12: 200}), now=12))
        self.assertIsNone(
            detect_movement(self.baseline, list(reversed(self.samples({12: 200}))), now=10)
        )
        samples = self.samples({12: 200})
        samples[-1][1].pop(11)
        self.assertIsNone(detect_movement(self.baseline, samples, now=10))
        samples = self.samples({12: 200})
        samples[-1][1][11] = float("nan")
        self.assertIsNone(detect_movement(self.baseline, samples, now=10))

    def test_circular_encoder_boundary_is_small_motion(self):
        self.baseline[12] = 4090
        samples = self.samples({12: 150})
        self.assertEqual(detect_movement(self.baseline, samples, now=10), 12)


class TransportTests(unittest.TestCase):
    def test_output_drain_is_bounded_without_tcdrain(self):
        transport = object.__new__(SerialTransport)
        transport.serial = type("HungSerial", (), {"out_waiting": 1})()
        with (
            patch("assembled.time.monotonic", side_effect=[0, 0.1, 0.3]),
            patch("assembled.time.sleep"),
        ):
            with self.assertRaisesRegex(AssembledError, "250ms"):
                transport.flush()


if __name__ == "__main__":
    unittest.main()
