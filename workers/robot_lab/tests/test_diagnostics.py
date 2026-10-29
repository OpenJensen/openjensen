"""Persistent incident evidence with fake buses only; never command real hardware."""

import http.client
import json
import stat
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnostics import Diagnostics
from hardware import HardwareError
from server import EncoderRangeError, RequestError, RobotLab, create_server, validate_raw
from test_robot_lab import CALIBRATION, FakeBus, FakeCamera


class JournalTests(unittest.TestCase):
    def test_restart_preserves_last_stop_and_ignores_torn_json(self):
        with tempfile.TemporaryDirectory() as temp:
            journal = Diagnostics(temp)
            stop = journal.emit("session_stopped", reason="range fault", retain_as_last_stop=True)
            journal.close()
            with (journal.root / "events.jsonl").open("a") as stream:
                stream.write('{"interrupted":')
            recovered = Diagnostics(temp)
            self.assertEqual(recovered.snapshot()["last_stop"]["id"], stop["id"])
            self.assertTrue(
                any(e["id"] == stop["id"] for e in recovered.snapshot(full=True)["events"])
            )
            recovered.close()
            again = Diagnostics(temp)
            self.assertTrue(
                any(
                    e["event"] == "companion_started" and e["boot_id"] == recovered.boot_id
                    for e in again.snapshot(full=True)["events"]
                )
            )
            again.close()

    def test_rotation_is_bounded_private_and_latest_stop_survives_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            journal = Diagnostics(temp, max_bytes=800, backups=2)
            stop = journal.emit(
                "session_stopped", reason="saved incident", retain_as_last_stop=True
            )
            for number in range(50):
                journal.emit("position_sample", value=number)
                journal.emit("control_started", value=number)
            journal.close()
            files = list(journal.root.glob("*.jsonl*"))
            self.assertLessEqual(len(files), 6)
            for path in files:
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
                self.assertLessEqual(path.stat().st_size, 800)
            recovered = Diagnostics(temp, max_bytes=800, backups=2)
            self.assertEqual(recovered.snapshot()["last_stop"]["id"], stop["id"])
            recovered.close()

    def test_disk_failure_is_visible_and_does_not_escape_to_motor_loop(self):
        with tempfile.TemporaryDirectory() as temp:
            journal = Diagnostics(temp)
            journal.pending.join()
            with patch.object(journal, "_append", side_effect=OSError("disk full")):
                journal.emit("session_stopped", reason="test", retain_as_last_stop=True)
                journal.pending.join()
            self.assertIn("disk full", journal.snapshot()["error"])
            self.assertEqual(journal.snapshot()["last_stop"]["reason"], "test")
            journal.close()


class IncidentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.buses = []

        def factory(*args):
            bus = FakeBus(*args)
            self.buses.append(bus)
            return bus

        self.lab = RobotLab(
            Path(self.temp.name),
            bus_factory=factory,
            camera=FakeCamera(),
            port_provider=lambda: [{"device": "/dev/cu.fake"}],
            start_threads=False,
        )
        self.lab.calibration_root = Path(self.temp.name) / "calibrations"
        path = self.lab.calibration_path("test", "follower")
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(CALIBRATION))

    def tearDown(self):
        self.lab.close()
        self.temp.cleanup()

    def start(self):
        self.lab.control_start(
            {
                "port": "/dev/cu.fake",
                "robot_id": "test",
                "mode": "manual",
                "confirm": "ENABLE MOTION",
            }
        )

    def run_loop_until_stopped(self, target):
        thread = threading.Thread(target=target)
        thread.start()
        deadline = time.monotonic() + 2
        while self.lab.mode != "idle" and time.monotonic() < deadline:
            time.sleep(0.01)
        self.lab.finished.set()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.lab.mode, "idle")

    def test_agent_range_fault_keeps_bad_reading_limits_and_pre_stop_context(self):
        self.start()
        self.lab.mode = "agent"
        self.buses[-1].raw["gripper"] = 99
        self.run_loop_until_stopped(self.lab._loop)
        stop = self.lab.status()["diagnostics"]["last_stop"]
        self.assertIn("1 ticks below minimum", stop["reason"])
        self.assertEqual(stop["context"]["mode"], "agent")
        self.assertEqual(stop["context"]["raw_positions"]["follower"]["gripper"], 99)
        fault = stop["context"]["fault"]
        self.assertEqual(
            (fault["source"], fault["joint"], fault["motor_id"]), ("follower", "gripper", 6)
        )
        self.assertEqual(
            (fault["min_ticks"], fault["max_ticks"], fault["outside_ticks"]), (100, 4000, 1)
        )
        self.assertTrue(stop["torque_disable_acknowledged"])
        self.assertFalse(self.lab.torque)
        self.assertFalse(self.buses[-1].opened)
        self.lab.error = None
        self.assertEqual(self.lab.status()["diagnostics"]["last_stop"]["id"], stop["id"])

    def test_leader_admission_fault_is_attributed_to_leader_calibration(self):
        self.start()
        self.lab.stop()
        leader = FakeBus()
        leader.raw["shoulder_lift"] = 4001
        with self.assertRaises(EncoderRangeError):
            self.lab._read_positions(leader, CALIBRATION, source="leader")
        self.assertEqual(self.lab.last_fault["source"], "leader")
        self.assertEqual(self.lab.last_fault["outside_ticks"], 1)
        self.assertEqual(self.lab.last_fault["measured_ticks"], 4001)

    def test_watchdog_reason_and_heartbeat_age_are_retained(self):
        self.start()
        self.lab.heartbeat -= 3
        self.run_loop_until_stopped(self.lab._watchdog)
        stop = self.lab.diagnostics.snapshot()["last_stop"]
        self.assertIn("heartbeat timed out", stop["reason"])
        self.assertGreaterEqual(stop["context"]["heartbeat_age_seconds"], 3)

    def test_usb_failure_retains_last_good_reading(self):
        self.start()
        self.lab._tick()
        self.buses[-1].positions = lambda: (_ for _ in ()).throw(HardwareError("USB timeout"))
        self.run_loop_until_stopped(self.lab._loop)
        stop = self.lab.diagnostics.snapshot()["last_stop"]
        self.assertIn("USB timeout", stop["reason"])
        self.assertEqual(stop["context"]["raw_positions"]["follower"]["gripper"], 2047)

    def test_blocked_disk_writer_cannot_delay_torque_off(self):
        self.start()
        self.lab.diagnostics.pending.join()
        entered, release = threading.Event(), threading.Event()

        def blocked(*args):
            entered.set()
            release.wait(3)

        with patch.object(self.lab.diagnostics, "_append", side_effect=blocked):
            self.lab.diagnostics.emit("test")
            self.assertTrue(entered.wait(1))
            try:
                started = time.monotonic()
                self.lab.stop("test stop")
                self.assertLess(time.monotonic() - started, 0.5)
                self.assertFalse(self.buses[-1].opened)
                self.assertFalse(self.lab.torque)
            finally:
                release.set()
                self.lab.diagnostics.pending.join()

    def test_stale_stop_does_not_create_false_stop_incident(self):
        self.start()
        with self.assertRaises(RequestError):
            self.lab.stop(expected_generation=self.lab.generation - 1)
        self.assertIsNone(self.lab.diagnostics.snapshot()["last_stop"])
        self.assertTrue(self.lab.torque)

    def test_diagnostics_requires_private_auth_and_status_is_read_only(self):
        server = create_server(self.lab, "x" * 40, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def request(method, body=None, authenticated=True):
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
            headers = {"Authorization": "Bearer " + "x" * 40} if authenticated else {}
            connection.request(
                method, "/v1/diagnostics" if method == "GET" else "/v1/stop", body, headers
            )
            response = connection.getresponse()
            result = response.status, json.loads(response.read())
            connection.close()
            return result

        try:
            self.assertEqual(request("GET", authenticated=False)[0], 401)
            status, result = request("GET")
            self.assertEqual(status, 200)
            self.assertIn("samples", result)
            self.assertEqual(self.buses, [])
            self.assertNotIn("x" * 40, json.dumps(result))
            self.start()
            status, result = request("POST", json.dumps({"cause": "page_hidden"}))
            self.assertEqual(status, 200)
            self.assertIn("page was hidden", result["diagnostics"]["last_stop"]["reason"])
            self.assertFalse(self.lab.torque)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_error_distance_uses_degrees_for_arm_and_percent_for_gripper(self):
        for joint, unit in (("shoulder_lift", "degrees"), ("gripper", "percent")):
            raw = dict.fromkeys(CALIBRATION, 2047)
            raw[joint] = 99
            with self.assertRaises(EncoderRangeError) as caught:
                validate_raw(raw, CALIBRATION)
            self.assertEqual(caught.exception.details["unit"], unit)
