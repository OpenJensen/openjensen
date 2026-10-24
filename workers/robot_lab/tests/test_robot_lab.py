"""No motors are touched: stateful fake-bus safety and wire-format tests."""

import http.client
import importlib.util
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import DatasetWriter
from hardware import IDS, JOINTS, HardwareError, denormalized, normalized
from server import RequestError, RobotLab, create_server, is_usb_serial

DATASET_AVAILABLE = all(
    importlib.util.find_spec(name) is not None for name in ("numpy", "PIL", "pyarrow")
)

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

    def read(self, register, motor_id):
        for name, sid, value in reversed(self.writes):
            if name == register and sid == motor_id:
                return value
        return 0

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
        self.assertEqual(self.buses[-1].writes[-1], ("Lock", 6, 1))
        self.assertEqual(self.buses[-1].read("Lock", 6), 1)
        self.assertFalse(
            any(
                register == "Torque_Enable" and value
                for register, _, value in self.buses[-1].writes
            )
        )

    def test_stop_cancels_isolated_scan_before_any_id_write(self):
        def factory(*args):
            bus = FakeBus(*args)
            bus.motors = {1: 777}
            original_ping = bus.ping

            def ping(motor_id):
                if motor_id == 2:
                    self.lab.stop()
                return original_ping(motor_id)

            bus.ping = ping
            self.buses.append(bus)
            return bus

        self.lab.bus_factory = factory
        with self.assertRaisesRegex(RequestError, "cancelled by Stop"):
            self.lab.setup_motor(
                {
                    "port": "/dev/cu.fake",
                    "joint": "gripper",
                    "isolated": True,
                    "confirm": "ASSIGN MOTOR ID",
                }
            )
        self.assertFalse(any(register == "ID" for register, _, _ in self.buses[-1].writes))
        self.assertFalse(self.buses[-1].opened)
        self.assertIsNone(self.lab.pending_bus)

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

    def test_gripper_protections_verified_before_any_torque_enable(self):
        original_read = FakeBus.read
        verified = []

        def read(bus, register, motor_id):
            if register in ("Max_Torque_Limit", "Protection_Current", "Overload_Torque"):
                self.assertFalse(any(r == "Torque_Enable" and v for r, _, v in bus.writes))
                verified.append((register, motor_id))
            return original_read(bus, register, motor_id)

        with patch.object(FakeBus, "read", read):
            self.start()
        self.assertEqual(
            verified,
            [("Max_Torque_Limit", 6), ("Protection_Current", 6), ("Overload_Torque", 6)],
        )

    def test_unverified_gripper_protection_aborts_before_any_torque_enable(self):
        original_read = FakeBus.read
        for failed_register in ("Max_Torque_Limit", "Protection_Current", "Overload_Torque"):
            for missing in (False, True):
                with self.subTest(register=failed_register, missing=missing):

                    def read(bus, register, motor_id):
                        if register == failed_register:
                            if missing:
                                raise HardwareError("No status packet")
                            return original_read(bus, register, motor_id) + 1
                        return original_read(bus, register, motor_id)

                    with patch.object(FakeBus, "read", read), self.assertRaises(HardwareError):
                        self.start()
                    bus = self.buses[-1]
                    self.assertFalse(any(r == "Torque_Enable" and v for r, _, v in bus.writes))
                    self.assertFalse(bus.opened)
                    self.assertIsNone(self.lab.pending_bus)
                    self.assertIsNone(self.lab.bus)
                    self.assertFalse(self.lab.torque)
                    self.assertEqual(self.lab.mode, "idle")
                    self.assertEqual(bus.calibration, CALIBRATION)

    def test_uncertain_torque_enable_never_publishes_control_and_disables_all_joints(self):
        def factory(*args):
            bus = FakeBus(*args)

            def fail_torque(register, motor_id, value):
                if register == "Torque_Enable" and value == 1 and motor_id == 3:
                    raise HardwareError("Torque state could not be verified")

            bus.on_write = fail_torque
            self.buses.append(bus)
            return bus

        self.lab.bus_factory = factory
        with self.assertRaisesRegex(HardwareError, "Torque state could not be verified"):
            self.start()
        bus = self.buses[-1]
        self.assertEqual(bus.writes[-6:], [("Torque_Enable", sid, 0) for sid in IDS.values()])
        self.assertFalse(bus.opened)
        self.assertIsNone(self.lab.bus)
        self.assertIsNone(self.lab.pending_bus)
        self.assertFalse(self.lab.torque)
        self.assertEqual(self.lab.mode, "idle")

    def test_recorded_middle_request_validates_before_opening_hardware(self):
        for extras in ({"move_to_middle": "yes"}, {"move_to_middle": True, "mode": "handguide"}):
            with self.assertRaises(RequestError):
                self.start(**extras)
        self.assertEqual(self.buses, [])
        invalid = json.loads(json.dumps(CALIBRATION))
        invalid["gripper"]["range_min"] = 2100
        self.lab.calibration_path("test", "follower").write_text(json.dumps(invalid))
        with self.assertRaisesRegex(ValueError, "Recorded middle for gripper"):
            self.start(move_to_middle=True)
        self.assertEqual(self.buses, [])

    def start_middle(self):
        def factory(*args):
            bus = FakeBus(*args)
            bus.raw = dict.fromkeys(JOINTS, 2200)
            self.buses.append(bus)
            return bus

        self.lab.bus_factory = factory
        return self.start(move_to_middle=True)

    def test_middle_holds_current_pose_then_moves_slowly_and_finishes_on_encoder_readback(self):
        with patch("motion.time.monotonic", return_value=100) as clock:
            result = self.start_middle()
            bus = self.buses[-1]
            self.assertTrue(result["session"]["centering"]["active"])
            self.assertEqual(bus.raw, dict.fromkeys(JOINTS, 2200))
            for i in range(1, 60):
                before = dict(bus.raw)
                clock.return_value = 100 + i * 0.1
                self.lab._tick()
                self.assertTrue(all(abs(bus.raw[j] - before[j]) <= 5 for j in JOINTS))
                if self.lab.middle_motion is None:
                    break
            self.assertIsNone(self.lab.middle_motion)
            self.assertEqual(bus.raw, dict.fromkeys(JOINTS, 2047))
            self.assertTrue(self.lab.torque)

    def test_middle_rejects_jogs_and_recordings_and_stop_cancels_future_targets(self):
        self.start_middle()
        with self.assertRaisesRegex(RequestError, "recorded middle"):
            self.lab.jog({"joint": "shoulder_pan", "delta": 1})
        with self.assertRaisesRegex(RequestError, "recorded middle"):
            self.lab.recording_start({})
        self.lab.stop()
        count = len(self.buses[-1].writes)
        self.lab._tick()
        self.assertEqual(len(self.buses[-1].writes), count)
        self.assertIsNone(self.lab.middle_motion)
        self.assertFalse(self.lab.torque)

    def test_stop_during_middle_write_does_not_command_another_joint(self):
        with patch("motion.time.monotonic", return_value=100) as clock:
            self.start_middle()
            bus = self.buses[-1]

            def stop_on_target(register, motor_id, value):
                if register == "Goal_Position":
                    bus.on_write = None
                    self.lab.stop("Stop during middle move")

            bus.on_write = stop_on_target
            before = len([w for w in bus.writes if w[0] == "Goal_Position"])
            clock.return_value = 100.1
            self.lab._tick()
            after = len([w for w in bus.writes if w[0] == "Goal_Position"])
            self.assertEqual(after - before, 1)
            self.assertIsNone(self.lab.middle_motion)
            self.assertFalse(self.lab.torque)

    def test_manual_tracking_error_names_joint_and_stops_sending_targets(self):
        self.start_middle()
        bus = self.buses[-1]
        bus.raw["wrist_flex"] += 100
        count = len(bus.writes)
        with self.assertRaisesRegex(HardwareError, "Wrist flex did not hold.*2300.*2200"):
            self.lab._tick()
        self.assertEqual(len(bus.writes), count)

    def test_hold_target_mismatch_prevents_all_torque_enables(self):
        original = FakeBus.read

        def read(bus, register, motor_id):
            return 0 if register == "Goal_Position" else original(bus, register, motor_id)

        with (
            patch.object(FakeBus, "read", read),
            self.assertRaisesRegex(HardwareError, "hold target"),
        ):
            self.start()
        self.assertFalse(any(r == "Torque_Enable" and v for r, _, v in self.buses[-1].writes))

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

    def test_repeated_jogs_advance_target_when_encoder_has_not_moved(self):
        self.start()
        bus = self.buses[-1]
        for joint in ("elbow_flex", "gripper"):
            origin = bus.raw[joint]
            targets = []
            for _ in range(3):
                bus.raw[joint] = origin
                self.lab.last_jog = 0
                result = self.lab.jog({"joint": joint, "delta": -1})
                targets.append(self.lab.commanded_raw[joint])
                self.assertEqual(result["session"]["targets"][joint], self.lab.action[joint])
            self.assertGreater(origin, targets[0])
            self.assertGreater(targets[0], targets[1])
            self.assertGreater(targets[1], targets[2])

    def test_jogs_cannot_accumulate_a_large_target_ahead_of_stalled_motor(self):
        self.start()
        bus = self.buses[-1]
        origin = bus.raw["elbow_flex"]
        for _ in range(2):
            bus.raw["elbow_flex"] = origin
            self.lab.last_jog = 0
            self.lab.jog({"joint": "elbow_flex", "delta": -2})
        bus.raw["elbow_flex"] = origin
        self.lab.last_jog = 0
        count = len(bus.writes)
        with self.assertRaisesRegex(ValueError, "has not reached its target"):
            self.lab.jog({"joint": "elbow_flex", "delta": -2})
        self.assertEqual(len(bus.writes), count)
        # A reversing click reduces the pending distance and is still allowed.
        self.lab.jog({"joint": "elbow_flex", "delta": 2})
        self.assertGreater(self.lab.commanded_raw["elbow_flex"], origin - 44)

    def test_jog_checks_tracking_before_any_new_target(self):
        self.start()
        bus = self.buses[-1]
        bus.raw["elbow_flex"] += 80
        count = len(bus.writes)
        with self.assertRaisesRegex(HardwareError, "Elbow flex did not hold"):
            self.lab.jog({"joint": "elbow_flex", "delta": -1})
        self.assertEqual(len(bus.writes), count)

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
        self.assertEqual(self.lab.calibration_state["phase"], "interrupted")
        self.assertEqual(self.lab.calibration_state["interrupted_stage"], "ranges")
        self.assertEqual(self.lab.calibration_state["error"], "Invalid range")

    def test_center_resets_position_baseline_after_offset_change(self):
        self.lab.calibration_start(
            {
                "port": "/dev/cu.fake",
                "robot_id": "new",
                "role": "follower",
                "confirm": "START CALIBRATION",
            }
        )
        self.lab.positions = dict.fromkeys(JOINTS, 4094)
        bus = self.lab.bus
        with patch.object(
            bus, "positions", side_effect=[dict.fromkeys(JOINTS, 4094), dict.fromkeys(JOINTS, 2047)]
        ):
            self.lab.calibration_center({})
        self.assertEqual(self.lab.positions, dict.fromkeys(JOINTS, 2047))
        self.lab._tick()
        self.assertEqual(self.lab.calibration_state["phase"], "ranges")
        bus.raw["shoulder_pan"] = 4095
        with self.assertRaisesRegex(HardwareError, "implausible jump"):
            self.lab._tick()

    def test_http_center_failure_preserves_interruption_and_allows_restart(self):
        self._check_http_center_failure()

    def test_http_center_decode_exception_restores_and_marks_interrupted(self):
        self._check_http_center_failure(decode=True)

    def _check_http_center_failure(self, decode=False):
        payload = {
            "port": "/dev/cu.fake",
            "robot_id": "new",
            "role": "follower",
            "confirm": "START CALIBRATION",
        }
        self.lab.calibration_start(payload)
        bus = self.lab.bus
        original = self.lab.original_calibration

        def fail_center(register, motor_id, value):
            if register == "Homing_Offset":
                raise HardwareError("Motor 1 write Homing_Offset failed: No status packet")

        def fail_read():
            raise IndexError("list index out of range")

        if decode:
            bus.positions = fail_read
        else:
            bus.on_write = fail_center
        expected = "list index out of range" if decode else "Homing_Offset"
        server = create_server(self.lab, "test-token-" + "a" * 32, port=0)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        try:
            connection.request(
                "POST",
                "/v1/calibration/center",
                "{}",
                {
                    "Authorization": "Bearer test-token-" + "a" * 32,
                    "Content-Type": "application/json",
                },
            )
            response = connection.getresponse()
            self.assertEqual(response.status, 409)
            self.assertIn(expected, json.loads(response.read())["error"])
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join(2)
        self.assertFalse(bus.opened)
        self.assertEqual(bus.calibration, original)
        self.assertFalse(any(r == "Torque_Enable" and v for r, _, v in bus.writes))
        self.assertFalse(self.lab.calibration_path("new", "follower").exists())
        state = self.lab.status()
        self.assertEqual(state["session"]["mode"], "idle")
        self.assertEqual(state["calibration"]["phase"], "interrupted")
        self.assertEqual(state["calibration"]["interrupted_stage"], "center")
        self.assertIn(expected, state["calibration"]["error"])
        self.lab.stop()
        self.assertEqual(self.lab.status()["calibration"]["phase"], "interrupted")
        self.lab.calibration_start(payload)
        self.assertEqual(self.lab.calibration_state["phase"], "center")
        self.assertNotIn("error", self.lab.calibration_state)
        self.assertIsNone(self.lab.error)
        self.lab.stop()
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
                self.lab.stop()
            return result

        bus.positions = positions
        with self.assertRaises(RequestError):
            self.lab.calibration_center({})
        self.assertEqual(self.lab.calibration_state["phase"], "idle")
        self.assertEqual(self.lab.mode, "idle")
        self.assertFalse(self.lab.calibration_path("new", "follower").exists())

    @unittest.skipUnless(DATASET_AVAILABLE, "Dataset dependencies required")
    def test_explicit_close_unsaved_recording_allows_lower_fps_without_deleting_files(self):
        self.start(mode="handguide")
        self.lab.recording_start(
            {"dataset_id": "too-fast", "task": "Reach", "fps": 20, "camera_index": None}
        )
        self.lab.recording_action("start")
        writer = self.lab.writer
        writer.add_frame(dict.fromkeys(JOINTS, 0.0), {}, captured_at=0)
        with self.assertRaises(ValueError):
            writer.add_frame(dict.fromkeys(JOINTS, 0.0), {}, captured_at=0.5)
        self.lab.recording_error = "Capture fell below requested FPS"
        pending = (writer.pending / "frames.jsonl").read_bytes()
        bus_writes = list(self.lab.bus.writes)
        for payload in ({"abandon_empty": True}, {"abandon_empty": "true"}):
            with self.assertRaises(RequestError):
                self.lab.recording_action("finish", payload)
            self.assertIs(self.lab.writer, writer)
        with self.assertRaises(ValueError):
            self.lab.recording_action("finish")
        self.assertIs(self.lab.writer, writer)
        with patch("dataset.atomic_json", side_effect=OSError("metadata unavailable")):
            with self.assertRaises(OSError):
                self.lab.recording_action(
                    "finish", {"abandon_empty": True, "confirm": "ABANDON EMPTY RECORDING"}
                )
        self.assertIs(self.lab.writer, writer)
        self.assertEqual((writer.pending / "frames.jsonl").read_bytes(), pending)
        result = self.lab.recording_action(
            "finish", {"abandon_empty": True, "confirm": "ABANDON EMPTY RECORDING"}
        )
        self.assertFalse(result["recording"]["active"])
        self.assertIsNone(self.lab.writer)
        self.assertEqual(self.lab.mode, "handguide")
        self.assertEqual(self.lab.bus.writes, bus_writes)
        self.assertEqual((writer.pending / "frames.jsonl").read_bytes(), pending)
        self.lab.recording_start(
            {"dataset_id": "lower-fps", "task": "Reach", "fps": 5, "camera_index": None}
        )
        self.assertEqual(self.lab.writer.fps, 5)
        self.assertTrue(writer.root.exists())

    @unittest.skipUnless(DATASET_AVAILABLE, "Dataset dependencies required")
    def test_unsaved_close_cannot_detach_dataset_with_saved_episode(self):
        self.start(mode="handguide")
        self.lab.recording_start(
            {"dataset_id": "keep-saved", "task": "Reach", "fps": 5, "camera_index": None}
        )
        self.lab.recording_action("start")
        writer = self.lab.writer
        for index in range(2):
            writer.add_frame(dict.fromkeys(JOINTS, 0.0), {}, captured_at=index / 5)
        self.lab.recording_action("save")
        with self.assertRaises(ValueError):
            self.lab.recording_action(
                "finish", {"abandon_empty": True, "confirm": "ABANDON EMPTY RECORDING"}
            )
        self.assertIs(self.lab.writer, writer)
        self.lab.recording_action("finish")
        self.assertTrue(writer.finalized)

    @unittest.skipUnless(DATASET_AVAILABLE, "Dataset dependencies required")
    def test_inflight_capture_cannot_append_after_unsaved_writer_is_detached(self):
        import numpy as np

        self.start(mode="handguide")
        writer = DatasetWriter(self.lab.datasets_dir, "in-flight", "Reach", 5, (2, 3, 3))
        self.lab.writer = writer
        self.lab.camera_index = 0
        writer.start_episode()
        entered, release = threading.Event(), threading.Event()

        def read(index):
            entered.set()
            release.wait(2)
            return np.zeros((2, 3, 3), dtype=np.uint8)

        self.lab.camera.read = read
        worker = threading.Thread(target=self.lab._tick)
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            self.lab.recording_action(
                "finish", {"abandon_empty": True, "confirm": "ABANDON EMPTY RECORDING"}
            )
        finally:
            release.set()
            worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(writer.frames, 0)
        self.assertFalse((writer.pending / "frames.jsonl").exists())
        self.assertTrue(writer.abandoned)

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
