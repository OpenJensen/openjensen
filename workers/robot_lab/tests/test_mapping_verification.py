"""Saved-ID checks exercise register fakes, never connected hardware."""

import hashlib
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assembled import AssembledError
from hardware import IDS
from server import RobotLab
from test_assembled_session import until
from test_identify import Bus
from test_robot_lab import FakeCamera
from verification import VerificationEngine, saved_mapping


def save_mapping(directory, port="/dev/fake"):
    path = (
        Path(directory) / "assembled" / (hashlib.sha256(port.encode()).hexdigest()[:20] + ".json")
    )
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "port": port,
                "phase": "complete",
                "ids": list(range(1, 7)),
                "mapping": IDS,
                "pending": None,
                "reboot_cleanup": {"verified_off": True, "pending": False},
            }
        )
    )
    return path


class Chain:
    def __init__(self):
        self.motors = {sid: Bus() for sid in range(1, 7)}
        self.closed = False
        for sid, bus in self.motors.items():
            bus.values[5] = sid

    def exchange(self, message, wait, length):
        bus = self.motors.get(message[2])
        return bus.exchange(message, wait, length) if bus else b""

    def send(self, message):
        self.motors[message[2]].send(message)

    def flush(self):
        pass

    def close(self):
        self.closed = True


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = save_mapping(self.temp.name)
        self.original = self.path.read_bytes()
        self.chain = Chain()

    def tearDown(self):
        self.temp.cleanup()

    def engine(self, *args, **kwargs):
        return VerificationEngine(*args, **kwargs, bus_factory=lambda port: self.chain)

    def test_only_selected_sram_changes_and_saved_mapping_survives(self):
        engine = self.engine("/dev/fake", self.temp.name)
        try:
            self.assertGreaterEqual(engine.check_mapping(6, lambda active: None), 4)
        finally:
            engine.close()
        self.assertTrue(self.chain.closed)
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertTrue(self.chain.motors[6].writes)
        for sid, bus in self.chain.motors.items():
            self.assertEqual(bus.values[40], 0)
            self.assertTrue(
                all(target == 6 and address >= 40 for target, address, value in bus.writes)
            )
            if sid != 6:
                self.assertEqual(bus.writes, [])

    def test_missing_address_prevents_any_motion_or_journal_change(self):
        del self.chain.motors[3]
        engine = self.engine("/dev/fake", self.temp.name)
        with self.assertRaisesRegex(AssembledError, "ID 3 does not have stable"):
            engine.check_mapping(6, lambda active: None)
        engine.close()
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertTrue(all(not bus.writes for bus in self.chain.motors.values()))

    def test_saved_check_avoids_full_bus_scan_under_usb_latency(self):
        clock = [0.0]
        exchanges = []
        original = self.chain.exchange

        def exchange(message, wait, length):
            clock[0] += 0.025
            exchanges.append(message[2])
            return original(message, wait, length)

        self.chain.exchange = exchange
        engine = self.engine("/dev/fake", self.temp.name)
        motion_started = []
        try:
            with patch("verification.time.monotonic", side_effect=lambda: clock[0]):
                result = engine.check_mapping(
                    6, lambda active: motion_started.append(clock[0]) if active else None
                )
            self.assertGreaterEqual(result, 4)
            self.assertEqual(set(exchanges), set(IDS.values()))
            self.assertLess(motion_started[0], 5)
            self.assertEqual(self.path.read_bytes(), self.original)
        finally:
            engine.close()

    def test_wrong_motor_model_or_enabled_torque_blocks_saved_wiggle(self):
        for register, value in ((3, 999), (40, 1)):
            with self.subTest(register=register):
                self.chain = Chain()
                self.chain.motors[2].values[register] = value
                engine = self.engine("/dev/fake", self.temp.name)
                try:
                    with self.assertRaises(AssembledError):
                        engine.check_mapping(6, lambda active: None)
                    self.assertTrue(all(not bus.writes for bus in self.chain.motors.values()))
                finally:
                    engine.close()

    def test_unfinished_or_wrong_port_mapping_is_rejected(self):
        record = json.loads(self.original)
        for change in (
            {"phase": "ready"},
            {"pending": {}},
            {"port": "/dev/other"},
            {"mapping": {}},
            {"reboot_cleanup": {"pending": True}},
        ):
            self.path.write_text(json.dumps({**record, **change}))
            with self.assertRaises(ValueError):
                saved_mapping(self.temp.name, "/dev/fake")

    def lab(self):
        return RobotLab(
            Path(self.temp.name),
            camera=FakeCamera(),
            start_threads=False,
            port_provider=lambda: [{"device": "/dev/fake"}],
            verification_factory=self.engine,
        )

    def test_restart_restores_mapping_without_opening_hardware(self):
        lab = self.lab()
        try:
            state = lab.status()
            self.assertEqual(state["assembled"]["phase"], "complete")
            self.assertEqual(state["assembled"]["mapping"], IDS)
            self.assertIsNone(lab.bus)
            self.assertTrue(all(not bus.writes for bus in self.chain.motors.values()))
        finally:
            lab.close()

    def test_confirmation_and_session_exclusion_then_cleanup(self):
        lab = self.lab()
        request = {
            "port": "/dev/fake",
            "joint": "gripper",
            "supported": True,
            "confirm": "WIGGLE SAVED MOTOR",
        }
        gate, entered = threading.Event(), threading.Event()
        original = VerificationEngine.check_mapping

        def held(engine, sid, callback):
            entered.set()
            gate.wait(2)
            return original(engine, sid, callback)

        try:
            for patch_request in ({"confirm": "wrong"}, {"supported": False}, {"joint": "unknown"}):
                with self.assertRaises(ValueError):
                    lab.mapping_wiggle({**request, **patch_request})
            with patch.object(VerificationEngine, "check_mapping", held):
                lab.mapping_wiggle(request)
                self.assertTrue(entered.wait(1))
                with self.assertRaises(ValueError):
                    lab.mapping_wiggle(request)
                gate.set()
                until(lambda: lab.mode == "idle", timeout=5)
            state = lab.status()
            self.assertEqual(state["verification"]["phase"], "done")
            self.assertFalse(state["session"]["torque_enabled"])
            self.assertEqual(self.path.read_bytes(), self.original)
        finally:
            gate.set()
            lab.close()

    def test_stop_cancels_selected_motion_and_preserves_ids(self):
        lab = self.lab()
        enabled = threading.Event()
        self.chain.motors[6].callback = lambda: (
            enabled.set() if self.chain.motors[6].values[40] else None
        )
        try:
            lab.mapping_wiggle(
                {
                    "port": "/dev/fake",
                    "joint": "gripper",
                    "supported": True,
                    "confirm": "WIGGLE SAVED MOTOR",
                }
            )
            self.assertTrue(enabled.wait(3))
            lab.stop()
            self.assertEqual(lab.status()["verification"]["phase"], "stopped")
            self.assertEqual(self.chain.motors[6].values[40], 0)
            self.assertTrue(self.chain.closed)
            self.assertEqual(self.path.read_bytes(), self.original)
        finally:
            lab.close()
