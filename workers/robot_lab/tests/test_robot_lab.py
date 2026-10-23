"""No motors are touched: stateful fake-bus safety and wire-format tests."""

import http.client
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hardware import IDS, JOINTS, HardwareError, denormalized, normalized
from server import RequestError, RobotLab, create_server, is_usb_serial

CALIBRATION = {
    joint: {
        "id": motor_id,
        "drive_mode": 0,
        "homing_offset": 0,
        "range_min": 100,
        "range_max": 4000,
    }
    for joint, motor_id in IDS.items()
}


class FakeBus:
    def __init__(self, port="fake", baudrate=1_000_000):
        self.lock = threading.RLock()
        self.port = self
        self.opened = True
        self.motors = dict.fromkeys(IDS.values(), 777)
        self.raw = dict.fromkeys(JOINTS, 2047)
        self.calibration = json.loads(json.dumps(CALIBRATION))
        self.writes = []
        self.on_write = None

    def ping(self, motor_id):
        return self.motors.get(motor_id)

    def write(self, register, motor_id, value):
        if self.on_write:
            self.on_write(register, motor_id, value)
        self.writes.append((register, motor_id, value))
        if register == "ID":
            self.motors[value] = self.motors.pop(motor_id)
        if register == "Goal_Position":
            self.raw[JOINTS[motor_id - 1]] = value

    def positions(self):
        return dict(self.raw)

    def read_calibration(self):
        return self.calibration

    def write_calibration(self, value):
        self.calibration = json.loads(json.dumps(value))

    def disable(self):
        for motor_id in IDS.values():
            self.write("Torque_Enable", motor_id, 0)

    def check_six(self):
        if self.motors != dict.fromkeys(IDS.values(), 777):
            raise HardwareError("All six motors required")

    def setBaudRate(self, value):
        return True

    def close(self):
        self.opened = False


class FakeCamera:
    def discover(self):
        return [{"index": 0, "name": "USB Camera", "width": 640, "height": 480}]

    def close(self):
        pass

    def close_idle(self):
        pass


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="robot-lab-test-")
        self.root = Path(self.temp.name)
        self.buses = []

        def factory(*args):
            bus = FakeBus(*args)
            self.buses.append(bus)
            return bus

        self.lab = RobotLab(
            self.root / "data",
            bus_factory=factory,
            camera=FakeCamera(),
            port_provider=lambda: [{"device": "/dev/cu.fake"}],
            start_threads=False,
        )
        self.lab.calibration_root = self.root / "calibrations"
        path = self.lab.calibration_path("test", "follower")
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(CALIBRATION))

    def tearDown(self):
        self.lab.close()
        self.temp.cleanup()

    def start(self, **extras):
        return self.lab.control_start(
            {
                "port": "/dev/cu.fake",
                "robot_id": "test",
                "mode": "manual",
                "confirm": "ENABLE MOTION",
                **extras,
            }
        )

    def test_probe_is_read_only_and_closes(self):
        result = self.lab.probe({"port": "/dev/cu.fake"})
        self.assertTrue(result["ready"])
        self.assertEqual(self.buses[-1].writes, [])
        self.assertFalse(self.buses[-1].opened)

    def test_port_inventory_excludes_bluetooth_and_debug_console(self):
        from types import SimpleNamespace

        for name in ("/dev/cu.Bluetooth-Incoming-Port", "/dev/cu.debug-console", "COM1"):
            self.assertFalse(is_usb_serial(SimpleNamespace(device=name, vid=None)))
        for name in (
            "/dev/cu.usbmodem5B901482131",
            "/dev/cu.wchusbserial123",
            "/dev/ttyUSB0",
            "/dev/ttyACM0",
        ):
            self.assertTrue(is_usb_serial(SimpleNamespace(device=name, vid=None)))
        self.assertTrue(is_usb_serial(SimpleNamespace(device="COM4", vid=0x1A86)))

    def test_real_enumerated_port_required(self):
        with self.assertRaises(RequestError):
            self.lab.probe({"port": "/tmp/arbitrary-file"})
        self.assertEqual(self.buses, [])

    def test_setup_requires_fresh_physical_isolation_confirmation(self):
        with self.assertRaises(RequestError):
            self.lab.setup_motor(
                {"port": "/dev/cu.fake", "joint": "gripper", "confirm": "ASSIGN MOTOR ID"}
            )
        self.assertEqual(self.buses, [])

    def test_setup_rejects_multiple_ids_without_writes(self):
        with self.assertRaises(RequestError):
            self.lab.setup_motor(
                {
                    "port": "/dev/cu.fake",
                    "joint": "gripper",
                    "isolated": True,
                    "confirm": "ASSIGN MOTOR ID",
                }
            )
        self.assertEqual(self.buses[-1].writes, [])

    def test_setup_assigns_only_verified_isolated_motor(self):
        def factory(*args):
            bus = FakeBus(*args)
            bus.motors = {1: 777}
            self.buses.append(bus)
            return bus

        self.lab.bus_factory = factory
        result = self.lab.setup_motor(
            {
                "port": "/dev/cu.fake",
                "joint": "gripper",
                "isolated": True,
                "confirm": "ASSIGN MOTOR ID",
            }
        )
        self.assertEqual(result["assigned"]["id"], 6)
        self.assertIn(("ID", 1, 6), self.buses[-1].writes)
        self.assertFalse(
            any(
                register == "Torque_Enable" and value
                for register, _, value in self.buses[-1].writes
            )
        )

    def test_start_requires_confirmation_and_calibration(self):
        with self.assertRaises(RequestError):
            self.start(confirm="yes")
        with self.assertRaises(RequestError):
            self.start(robot_id="missing")
        self.assertEqual(self.buses, [])

    def test_hold_current_position_before_any_torque_enable(self):
        result = self.start()
        bus = self.buses[-1]
        enables = [
            i
            for i, (register, _, value) in enumerate(bus.writes)
            if register == "Torque_Enable" and value == 1
        ]
        goals = [i for i, (register, _, _) in enumerate(bus.writes) if register == "Goal_Position"]
        self.assertEqual(len(enables), 6)
        self.assertLess(max(goals), min(enables))
        self.assertTrue(result["session"]["torque_enabled"])

    def test_handguide_never_enables_torque(self):
        self.start(mode="handguide")
        self.assertFalse(
            any(
                register == "Torque_Enable" and value
                for register, _, value in self.buses[-1].writes
            )
        )
        self.assertFalse(self.lab.torque)

    def test_jog_bounds_and_rate_limit(self):
        self.start()
        for delta in (3, float("nan"), float("inf"), True):
            with self.assertRaises(RequestError):
                self.lab.jog({"joint": "shoulder_pan", "delta": delta})
        before = self.lab.positions["shoulder_pan"]
        self.lab.jog({"joint": "shoulder_pan", "delta": 2})
        actual = self.lab.action["shoulder_pan"] - before
        self.assertGreater(actual, 1.8)
        self.assertLessEqual(actual, 2)
        with self.assertRaises(RequestError):
            self.lab.jog({"joint": "shoulder_pan", "delta": 2})

    def test_bad_but_plausible_encoder_read_cannot_create_a_large_jog(self):
        self.start()
        bus = self.buses[-1]
        bus.raw["elbow_flex"] = 0
        old_goals = len([entry for entry in bus.writes if entry[0] == "Goal_Position"])
        with self.assertRaises(HardwareError):
            self.lab.jog({"joint": "elbow_flex", "delta": 2})
        self.assertEqual(
            old_goals, len([entry for entry in bus.writes if entry[0] == "Goal_Position"])
        )

    def test_stop_cancels_startup_before_enable(self):
        def factory(*args):
            bus = FakeBus(*args)

            def on_write(register, motor_id, value):
                if register == "Goal_Position":
                    bus.on_write = None
                    self.lab.stop("Test Stop during startup")

            bus.on_write = on_write
            self.buses.append(bus)
            return bus

        self.lab.bus_factory = factory
        with self.assertRaises(RequestError):
            self.start()
        self.assertIsNone(self.lab.bus)
        self.assertEqual(self.lab.mode, "idle")
        self.assertFalse(
            any(
                register == "Torque_Enable" and value
                for register, _, value in self.buses[-1].writes
            )
        )
        self.assertFalse(self.buses[-1].opened)

    def test_stop_cancels_jog_after_position_read(self):
        self.start()
        bus = self.buses[-1]
        original = bus.positions

        def positions():
            self.lab.stop()
            return original()

        bus.positions = positions
        old_count = len([entry for entry in bus.writes if entry[0] == "Goal_Position"])
        with self.assertRaises(RequestError):
            self.lab.jog({"joint": "shoulder_pan", "delta": 1})
        self.assertEqual(
            old_count, len([entry for entry in bus.writes if entry[0] == "Goal_Position"])
        )

    def test_watchdog_disables_and_closes_without_new_command(self):
        self.start()
        self.lab.heartbeat = time.monotonic() - 3
        thread = threading.Thread(target=self.lab._watchdog, daemon=True)
        thread.start()
        time.sleep(0.25)
        self.lab.finished.set()
        thread.join(1)
        self.assertFalse(self.lab.torque)
        self.assertIsNone(self.lab.bus)
        self.assertFalse(self.buses[-1].opened)
        self.assertIn("heartbeat", self.lab.error)

    def test_recording_deadlines_do_not_accumulate_processing_jitter(self):
        from types import SimpleNamespace

        self.start(mode="handguide")
        times = []
        writer = SimpleNamespace(
            episode_active=True,
            fps=10,
            finalized=False,
            id="fake",
            frames=0,
            episodes=[],
            actual_fps=None,
        )
        writer.add_frame = lambda positions, action, image, captured_at: times.append(captured_at)
        self.lab.writer = writer
        self.lab.next_record = 100.0
        with patch("server.time.monotonic", return_value=100.03):
            self.lab._tick()
        self.assertAlmostEqual(self.lab.next_record, 100.1)
        with patch("server.time.monotonic", return_value=100.14):
            self.lab._tick()
        self.assertAlmostEqual(self.lab.next_record, 100.2)
        self.assertEqual(times, [100.03, 100.14])

    def test_invalid_calibration_reading_aborts(self):
        self.lab.calibration_start(
            {
                "port": "/dev/cu.fake",
                "robot_id": "new",
                "role": "follower",
                "confirm": "START CALIBRATION",
            }
        )
        self.lab.calibration_center({})
        self.lab.positions = dict.fromkeys(JOINTS, 2047)
        self.lab.bus.raw["elbow_flex"] = 4095
        with self.assertRaises(HardwareError):
            self.lab._tick()
        original = self.lab.original_calibration
        self.lab.stop("Invalid range")
        self.assertEqual(self.buses[-1].calibration, original)
        self.assertEqual(self.lab.calibration_state["phase"], "idle")

    def test_stop_during_center_capture_cannot_resurrect_calibration_phase(self):
        self.lab.calibration_start(
            {
                "port": "/dev/cu.fake",
                "robot_id": "new",
                "role": "follower",
                "confirm": "START CALIBRATION",
            }
        )
        bus = self.lab.bus
        original_positions = bus.positions
        reads = []

        def positions():
            reads.append(True)
            result = original_positions()
            if len(reads) == 2:
                self.lab.stop("Cancelled while capturing center")
            return result

        bus.positions = positions
        with self.assertRaises(RequestError):
            self.lab.calibration_center({})
        self.assertEqual(self.lab.calibration_state["phase"], "idle")
        self.assertEqual(self.lab.mode, "idle")
        self.assertFalse(self.lab.calibration_path("new", "follower").exists())

    def test_normalization_roundtrip_and_clamped_limits(self):
        for joint in JOINTS:
            value = normalized(joint, 2047, CALIBRATION)
            self.assertLessEqual(abs(denormalized(joint, value, CALIBRATION) - 2047), 1)
            self.assertEqual(denormalized(joint, 1e5, CALIBRATION), 4000)
            self.assertEqual(denormalized(joint, -1e5, CALIBRATION), 100)


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.lab = RobotLab(
            Path(self.temp.name), camera=FakeCamera(), port_provider=lambda: [], start_threads=False
        )
        self.token = "x" * 40
        self.server = create_server(self.lab, self.token, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.lab.close()
        self.temp.cleanup()

    def get(self, path, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        connection.request("GET", path, headers=headers or {})
        response = connection.getresponse()
        value = json.loads(response.read())
        connection.close()
        return response.status, value

    def test_token_required(self):
        self.assertEqual(self.get("/v1/status")[0], 401)
        self.assertEqual(self.get("/v1/status", {"Authorization": "Bearer wrong"})[0], 401)
        self.assertEqual(self.get("/v1/status", {"Authorization": "Bearer " + self.token})[0], 200)

    def test_browser_direct_access_rejected_even_with_token(self):
        status, value = self.get(
            "/v1/status", {"Authorization": "Bearer " + self.token, "Origin": "https://example.com"}
        )
        self.assertEqual(status, 403)

    def test_non_loopback_bind_rejected(self):
        with self.assertRaises(ValueError):
            create_server(self.lab, self.token, host="0.0.0.0", port=0)


if __name__ == "__main__":
    unittest.main()
