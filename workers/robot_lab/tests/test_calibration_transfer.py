"""Portable profile transfers must never write a motor register or enable motion."""

import copy
import http.client
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import RequestError, RobotLab, create_server
from test_robot_lab import CALIBRATION, FakeBus, FakeCamera


class CalibrationTransferTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="calibration-transfer-test-")
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
        self.original = self.lab.calibration_path("original", "follower")
        self.original.parent.mkdir(parents=True)
        self.original.write_text(json.dumps(CALIBRATION))

    def tearDown(self):
        self.lab.close()
        self.temp.cleanup()

    def payload(self, **changes):
        return {
            "robot_id": "destination",
            "role": "follower",
            "document": copy.deepcopy(CALIBRATION),
            "confirm": "IMPORT CALIBRATION",
            **changes,
        }

    def test_export_round_trips_under_another_profile_name_without_hardware(self):
        exported = self.lab.calibration_export("original", "follower")
        self.assertEqual(exported["calibration"], CALIBRATION)
        self.assertEqual(exported["robot_id"], "original")
        self.assertNotIn(str(self.root), json.dumps(exported))
        result = self.lab.calibration_import(self.payload(document=exported))
        self.assertFalse(result["calibration_transfer"]["verified"])
        self.assertEqual(self.lab.load_calibration("destination", "follower"), CALIBRATION)
        self.assertEqual(self.buses, [])
        self.assertEqual(self.original.read_text(), json.dumps(CALIBRATION))

    def test_import_accepts_standard_lerobot_files_for_leader_profiles(self):
        self.lab.calibration_import(self.payload(role="leader"))
        self.assertEqual(self.lab.load_calibration("destination", "leader"), CALIBRATION)
        self.assertFalse(self.lab.calibration_path("destination", "follower").exists())
        self.assertEqual(self.buses, [])

    def test_connected_import_verifies_and_acknowledges_setup_without_any_writes(self):
        self.lab.calibration_revisions.mark_dirty("/dev/cu.fake")
        self.lab.error = (
            "Motor calibration differs from the saved profile; recalibrate before motion"
        )
        result = self.lab.calibration_import(self.payload(port="/dev/cu.fake"))
        transfer = result["calibration_transfer"]
        self.assertTrue(transfer["verified"])
        self.assertTrue(
            self.lab.calibration_revisions.fresh(
                "/dev/cu.fake",
                "destination",
                "follower",
                self.lab.calibration_path("destination", "follower"),
            )
        )
        self.assertIsNone(result["session"]["error"])
        self.assertEqual(result["session"]["mode"], "idle")
        self.assertFalse(result["session"]["torque_enabled"])
        self.assertEqual(self.buses[-1].writes, [])
        self.assertFalse(self.buses[-1].opened)

    def test_mismatching_motors_leave_existing_file_and_motor_settings_untouched(self):
        different = copy.deepcopy(CALIBRATION)
        different["elbow_flex"]["homing_offset"] = 123
        before = self.original.read_bytes()
        with self.assertRaisesRegex(RequestError, "does not match.*elbow flex"):
            self.lab.calibration_import(
                self.payload(
                    robot_id="original",
                    port="/dev/cu.fake",
                    replace_existing=True,
                    document=different,
                )
            )
        self.assertEqual(self.original.read_bytes(), before)
        self.assertEqual(list(self.original.parent.glob("*.backup-*.json")), [])
        self.assertEqual(self.buses[-1].writes, [])
        self.assertFalse(self.buses[-1].opened)

    def test_replacement_requires_confirmation_and_keeps_exact_previous_file(self):
        before = self.original.read_bytes()
        with self.assertRaisesRegex(RequestError, "Confirm replacing"):
            self.lab.calibration_import(self.payload(robot_id="original"))
        replacement = copy.deepcopy(CALIBRATION)
        replacement["gripper"]["range_max"] = 3500
        result = self.lab.calibration_import(
            self.payload(robot_id="original", document=replacement, replace_existing=True)
        )
        backups = list(self.original.parent.glob("original.backup-*.json"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), before)
        self.assertTrue(result["calibration_transfer"]["backup_created"])
        self.assertEqual(self.lab.load_calibration("original", "follower"), replacement)

    def test_verified_import_keeps_an_unrelated_motor_stop_failure_visible(self):
        self.lab.error = "Stop attempted but motor acknowledgement failed."
        result = self.lab.calibration_import(self.payload(port="/dev/cu.fake"))
        self.assertEqual(result["session"]["error"], self.lab.error)
        self.assertEqual(self.buses[-1].writes, [])

    def test_failed_publication_restores_previous_file(self):
        before = self.original.read_bytes()
        with patch("server.atomic_json", side_effect=OSError("Disk full")):
            with self.assertRaisesRegex(OSError, "Disk full"):
                self.lab.calibration_import(
                    self.payload(robot_id="original", replace_existing=True)
                )
        self.assertEqual(self.original.read_bytes(), before)

    def test_invalid_documents_roles_and_names_never_touch_hardware_or_save(self):
        cases = [
            self.payload(robot_id="../escape"),
            self.payload(role="invalid"),
            self.payload(document=[]),
            self.payload(document={}),
            self.payload(replace_existing="yes"),
            self.payload(confirm="YES"),
        ]
        exported = self.lab.calibration_export("original", "follower")
        cases.append(self.payload(role="leader", document=exported))
        for field, value in [
            ("id", True),
            ("drive_mode", False),
            ("homing_offset", 2048),
            ("range_min", -1),
            ("range_max", 4096),
            ("homing_offset", 2.5),
        ]:
            document = copy.deepcopy(CALIBRATION)
            document["gripper"][field] = value
            cases.append(self.payload(document=document))
        for document in [dict(CALIBRATION, gripper=[]), dict(CALIBRATION, extra={})]:
            cases.append(self.payload(document=document))
        for payload in cases:
            with self.subTest(payload=payload), self.assertRaises((ValueError, TypeError)):
                self.lab.calibration_import(payload)
        self.assertFalse(self.lab.calibration_path("destination", "follower").exists())
        self.assertEqual(self.buses, [])

    def test_export_and_import_reject_symlinks(self):
        target = self.lab.calibration_path("destination", "follower")
        target.symlink_to(self.original)
        with self.assertRaises(RequestError):
            self.lab.calibration_import(self.payload(replace_existing=True))
        with self.assertRaises(RequestError):
            self.lab.calibration_export("destination", "follower")
        self.assertEqual(self.original.read_text(), json.dumps(CALIBRATION))

    def test_import_is_blocked_during_a_live_session(self):
        self.lab.mode = "manual"
        with self.assertRaisesRegex(RequestError, "Stop the current session"):
            self.lab.calibration_import(self.payload(port="/dev/cu.fake"))
        self.lab.mode = "idle"
        self.assertEqual(self.buses, [])

    def test_offline_import_does_not_bypass_mismatch_check_before_motion(self):
        different = copy.deepcopy(CALIBRATION)
        different["gripper"]["homing_offset"] = 123
        self.lab.calibration_import(self.payload(document=different))
        with self.assertRaisesRegex(RequestError, "Motor calibration differs"):
            self.lab.control_start(
                {
                    "port": "/dev/cu.fake",
                    "robot_id": "destination",
                    "mode": "manual",
                    "confirm": "ENABLE MOTION",
                }
            )
        self.assertFalse(
            any(name == "Torque_Enable" and value == 1 for name, _, value in self.buses[-1].writes)
        )

    def test_wire_routes_require_auth_and_round_trip_profiles(self):
        token = "x" * 40
        server = create_server(self.lab, token, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def request(method, path, payload=None, authorized=True):
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            headers = {"Authorization": "Bearer " + token} if authorized else {}
            connection.request(
                method,
                path,
                body=json.dumps(payload) if payload is not None else None,
                headers=headers,
            )
            response = connection.getresponse()
            status, data = response.status, json.loads(response.read())
            connection.close()
            return status, data

        try:
            path = "/v1/calibration/export?robot_id=original&role=follower"
            self.assertEqual(request("GET", path, authorized=False)[0], 401)
            code, exported = request("GET", path)
            self.assertEqual(code, 200)
            code, result = request(
                "POST", "/v1/calibration/import", self.payload(document=exported)
            )
            self.assertEqual(code, 200)
            self.assertEqual(result["calibration_transfer"]["robot_id"], "destination")
            self.assertEqual(request("GET", path + "&role=leader")[0], 422)
            self.assertEqual(self.buses, [])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
