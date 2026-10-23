"""Bounded powered identification is exercised only against a register fake."""

import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _vendor.servotools.framing import packet
from identify import AMPLITUDE, IdentificationUnavailable, wiggle


class Bus:
    def __init__(self):
        self.values = {
            3: 777,
            5: 10,
            9: 0,
            11: 4095,
            33: 0,
            40: 0,
            41: 20,
            42: 2048,
            44: 0,
            46: 500,
            48: 800,
            56: 2048,
        }
        self.writes = []
        self.fail_on = None
        self.callback = lambda: None
        self.moves = True

    def exchange(self, message, wait, length):
        address, size = message[5:7]
        return packet(message[2], 0, self.values[address].to_bytes(size, "little"))

    def send(self, message):
        sid, address = message[2], message[5]
        value = int.from_bytes(message[6:-1], "little")
        self.writes.append((sid, address, value))
        self.values[address] = value
        if self.moves and address == 42 and self.values[40]:
            self.values[56] = value
        self.callback()
        if address == self.fail_on:
            self.fail_on = None
            raise OSError("uncertain serial write")

    def flush(self):
        pass


class NoWait(threading.Event):
    def wait(self, timeout=None):
        return self.is_set()


class IdentifyTests(unittest.TestCase):
    def setUp(self):
        self.bus = Bus()
        self.stop = NoWait()
        self.sleep = patch("identify.time.sleep")
        self.sleep.start()

    def tearDown(self):
        self.sleep.stop()

    def run_wiggle(self):
        return wiggle(self.bus, 10, list(range(10, 16)), self.stop)

    def test_selected_motor_only_fixed_targets_sram_only_and_restoration(self):
        original = dict(self.bus.values)
        self.assertGreaterEqual(self.run_wiggle(), 4)
        self.assertTrue(all(sid == 10 and address >= 40 for sid, address, value in self.bus.writes))
        goals = [v for sid, a, v in self.bus.writes if a == 42]
        self.assertEqual(goals, [2048, 2048 + AMPLITUDE, 2048 - AMPLITUDE, 2048])
        self.assertLessEqual(
            max(v for sid, a, v in self.bus.writes if a == 48 and v != original[48]), 100
        )
        self.assertEqual(self.bus.values[40], 0)
        for a in (41, 44, 46, 48):
            self.assertEqual(self.bus.values[a], original[a])

    def test_missing_addresses_and_bad_motor_id_never_write(self):
        for ids, sid in [([10], 10), ([10] * 6, 10), (range(10, 16), True), (range(10, 16), 25)]:
            with self.assertRaises(ValueError):
                wiggle(self.bus, sid, ids, self.stop)
        self.assertEqual(self.bus.writes, [])

    def test_unready_position_mode_torque_and_limit_never_write(self):
        for address, value in [(33, 1), (40, 1), (56, 2), (48, 0)]:
            bus = Bus()
            bus.values[address] = value
            with self.assertRaises(IdentificationUnavailable):
                wiggle(bus, 10, range(10, 16), self.stop)
            self.assertEqual(bus.writes, [])

    def test_cancellation_after_torque_on_sends_no_goals_and_restores_off(self):
        def callback():
            if self.bus.writes[-1] == (10, 40, 1):
                self.stop.set()

        self.bus.callback = callback
        with self.assertRaisesRegex(ValueError, "stopped"):
            self.run_wiggle()
        on = self.bus.writes.index((10, 40, 1))
        self.assertFalse(any(a == 42 for sid, a, v in self.bus.writes[on + 1 :]))
        self.assertEqual(self.bus.values[40], 0)
        self.assertEqual(self.bus.values[48], 800)

    def test_uncertain_torque_on_write_still_disables(self):
        self.bus.fail_on = 40
        with self.assertRaises(OSError):
            self.run_wiggle()
        self.assertIn((10, 40, 0), self.bus.writes)
        self.assertEqual(self.bus.values[48], 800)

    def test_no_movement_does_not_produce_identification_evidence(self):
        self.bus.moves = False
        with self.assertRaisesRegex(IdentificationUnavailable, "No measurable"):
            self.run_wiggle()
        self.assertEqual(self.bus.values[40], 0)
