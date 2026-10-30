"""Device identity is separate from a session's leader/follower assignment."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from device_registry import DeviceRegistry
from server import RequestError, RobotLab
from test_robot_lab import CALIBRATION, FakeCamera


class DeviceRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ports = [dict(device="/dev/ttyACM0", serial_number="serial-A", description="USB arm")]
        self.opens = []
        self.lab = RobotLab(
            self.root / "data",
            camera=FakeCamera(),
            port_provider=lambda: self.ports,
            bus_factory=lambda *a: self.opens.append(a),
            start_threads=False,
        )
        self.lab.calibration_root = self.root / "profiles"

    def tearDown(self):
        self.lab.close()
        self.temp.cleanup()

    def configure(self, **changes):
        device = self.lab.devices()["saved_devices"][0]
        payload = dict(id=device["id"], port=device["port"], name="Desk arm", role="follower")
        payload.update(changes)
        return self.lab.configure_device(payload)["saved_devices"][0]

    def test_discovery_and_configuration_never_open_motors_and_survive_restart(self):
        self.assertIsNone(self.lab.devices()["saved_devices"][0]["role"])
        saved = self.configure()
        self.assertEqual(DeviceRegistry(self.root / "data").read()[0]["name"], "Desk arm")
        self.ports.clear()
        offline = self.lab.devices()["saved_devices"][0]
        self.assertEqual(offline["id"], saved["id"])
        self.assertFalse(offline["connected"])
        self.assertEqual(self.opens, [])

    def test_serial_reenumeration_and_recycled_address_do_not_change_identity(self):
        saved = self.configure()
        self.ports[0]["device"] = "/dev/ttyACM4"
        moved = self.lab.devices()["saved_devices"]
        self.assertEqual(len(moved), 1)
        self.assertEqual(moved[0]["id"], saved["id"])
        self.assertEqual(moved[0]["port"], "/dev/ttyACM4")
        self.ports[0]["serial_number"] = "serial-B"
        replaced = self.lab.devices()["saved_devices"]
        self.assertEqual(len(replaced), 2)
        self.assertIsNone(replaced[0]["role"])
        self.assertFalse(replaced[1]["connected"])

    def test_legacy_profile_is_recovered_and_stays_unique_after_serial_adoption(self):
        path = self.lab.calibration_path("existing-arm", "follower")
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(CALIBRATION))
        self.lab.calibration_revisions.acknowledge("/dev/ttyACM0", "existing-arm", "follower", path)
        self.assertEqual(len(self.lab.devices()["saved_devices"]), 1)
        self.assertEqual(len(self.lab.devices()["saved_devices"]), 1)
        self.ports[0]["device"] = "/dev/ttyACM5"
        devices = self.lab.devices()["saved_devices"]
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0]["name"], "existing-arm")

    def test_rename_and_role_change_preserve_calibration_and_revision(self):
        device = self.configure()
        path = self.lab.calibration_path(device["robot_id"], "follower")
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(CALIBRATION))
        self.lab.calibration_revisions.mark_dirty(device["port"])
        self.lab.calibration_revisions.acknowledge(
            device["port"], device["robot_id"], "follower", path
        )
        before = path.read_bytes()
        changed = self.configure(name="Hand guide", role="leader")
        self.assertEqual(changed["robot_id"], device["robot_id"])
        self.assertEqual(self.lab.calibration_path(device["robot_id"], "leader"), path)
        self.assertEqual(
            self.lab.load_calibration(device["robot_id"], "leader", device["port"]), CALIBRATION
        )
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(
            {p["role"] for p in self.lab.status()["calibrations"]}, {"leader", "follower"}
        )
        self.assertEqual(self.opens, [])

    def test_active_session_and_invalid_configuration_are_rejected(self):
        self.lab.mode = "manual"
        with self.assertRaises(RequestError):
            self.configure()
        self.lab.mode = "idle"
        for changes in [dict(name="bad\nname"), dict(role="other"), dict(port="/tmp/foreign")]:
            with self.assertRaises(ValueError):
                self.configure(**changes)
        self.assertFalse(self.lab.device_registry.path.exists())

    def test_ambiguous_serial_numbers_never_merge_distinct_controllers(self):
        self.ports.append(dict(device="/dev/ttyACM1", serial_number="serial-A"))
        devices = self.lab.devices()["saved_devices"]
        self.assertEqual(len({d["id"] for d in devices}), 2)

    def test_unplugging_one_duplicate_serial_does_not_select_the_other_arm(self):
        first = self.configure(name="First")
        self.ports.append(dict(device="/dev/ttyACM1", serial_number="serial-A"))
        second = next(d for d in self.lab.devices()["saved_devices"] if d["port"] == "/dev/ttyACM1")
        self.lab.configure_device(
            dict(id=second["id"], port=second["port"], name="Second", role="leader")
        )
        self.ports.pop(0)
        entries = self.lab.devices()["saved_devices"]
        connected = [d for d in entries if d["connected"]]
        self.assertEqual(len(connected), 1)
        self.assertEqual(connected[0]["name"], "Second")
        self.assertNotEqual(connected[0]["id"], first["id"])

    def test_new_serial_can_be_configured_at_a_disconnected_devices_old_address(self):
        old = self.configure()
        self.ports[0]["serial_number"] = "serial-B"
        current = next(d for d in self.lab.devices()["saved_devices"] if d["connected"])
        result = self.lab.configure_device(
            dict(id=current["id"], port=current["port"], name="New arm", role="leader")
        )
        self.assertEqual(len(result["saved_devices"]), 2)
        self.assertFalse(
            next(d for d in result["saved_devices"] if d["id"] == old["id"])["connected"]
        )

    def test_same_legacy_name_in_both_roles_keeps_distinct_calibration_files(self):
        for role, port in [("follower", "/dev/ttyACM0"), ("leader", "/dev/ttyACM1")]:
            path = self.lab.calibration_path("same-name", role)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(CALIBRATION))
            self.lab.calibration_revisions.acknowledge(port, "same-name", role, path)
        self.ports.append(dict(device="/dev/ttyACM1", serial_number="serial-B"))
        devices = self.lab.devices()["saved_devices"]
        self.assertEqual(len(devices), 2)
        self.assertEqual(len({d["robot_id"] for d in devices}), 2)
        for d in devices:
            for slot in ("leader", "follower"):
                path = self.lab.calibration_path(d["robot_id"], slot)
                self.assertEqual(
                    path.parent.name, "so_leader" if d["role"] == "leader" else "so_follower"
                )
                self.assertEqual(path.stem, "same-name")
        self.assertEqual(len(self.lab.devices()["saved_devices"]), 2)

    def test_unbound_legacy_profile_binds_only_to_an_explicitly_selected_connection(self):
        path = self.lab.calibration_path("stored-arm", "follower")
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(CALIBRATION))
        entries = self.lab.devices()["saved_devices"]
        saved = next(d for d in entries if d["robot_id"] == "stored-arm")
        self.assertEqual(saved["port"], "")
        result = self.lab.configure_device(
            dict(id=saved["id"], port="/dev/ttyACM0", name="Stored arm", role="follower")
        )
        self.assertEqual(len(result["saved_devices"]), 1)
        device = result["saved_devices"][0]
        self.assertEqual(device["robot_id"], "stored-arm")
        self.assertEqual(device["port"], "/dev/ttyACM0")
        self.assertEqual(self.lab.load_calibration(device["robot_id"], "leader"), CALIBRATION)
        self.assertEqual(self.opens, [])


if __name__ == "__main__":
    unittest.main()
