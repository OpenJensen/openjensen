"""Exercise calibration choices and read-only recovery using fake motor buses."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import RequestError, RobotLab
from test_robot_lab import CALIBRATION, FakeBus, FakeCamera


class CalibrationProfilesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bus = FakeBus()
        self.lab = RobotLab(
            self.root / "data",
            bus_factory=self.open_bus,
            camera=FakeCamera(),
            port_provider=lambda: [{"device": "fake"}],
            start_threads=False,
        )
        self.lab.calibration_root = self.root / "calibrations"
        self.path = self.lab.calibration_path("arm", "follower")
        self.path.parent.mkdir(parents=True)
        self.path.write_text(json.dumps(CALIBRATION))
        self.lab.calibration_revisions.mark_dirty("fake")

    def tearDown(self):
        self.lab.close()
        self.temp.cleanup()

    def payload(self, source="saved"):
        return {
            "robot_id": "arm",
            "role": "follower",
            "port": "fake",
            "source": source,
            "same_arm": True,
            "confirm": "REUSE CALIBRATION",
            **(
                {
                    "profile_key": self.lab.calibration_library("arm", "follower", "fake")[
                        "profiles"
                    ][0]["key"]
                }
                if source == "saved"
                else {}
            ),
        }

    def test_saved_profile_revalidates_after_id_setup_without_motor_writes(self):
        self.assertTrue(self.lab.calibration_library("arm", "follower", "fake")["needs_review"])
        self.lab.error = "Motor IDs changed on this port; save a new calibration before control"
        result = self.lab.calibration_reuse(self.payload())
        self.assertIsNone(result["session"]["error"])
        self.assertEqual(self.lab.load_calibration("arm", "follower", "fake"), CALIBRATION)
        self.assertFalse(self.lab.calibration_library("arm", "follower", "fake")["needs_review"])
        self.assertEqual(self.bus.writes, [])
        self.assertFalse(self.bus.opened)
        history = list(self.lab.calibration_profiles.root.glob("*.json"))
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].stat().st_mode & 0o777, 0o600)

    def test_read_motors_preserves_old_profile_and_reuses_history_after_restart(self):
        old = self.path.read_bytes()
        self.bus.calibration["gripper"]["range_min"] = 200
        self.lab.calibration_reuse(self.payload("motors"))
        self.assertEqual(next(self.path.parent.glob("*.backup-*.json")).read_bytes(), old)
        entries = self.lab.calibration_library("arm", "follower", "fake")["profiles"]
        self.assertEqual(len(entries), 2)
        self.assertEqual(sum(entry["current"] for entry in entries), 1)
        self.assertEqual(self.bus.writes, [])
        self.assertEqual(json.loads(self.path.read_text())["gripper"]["range_min"], 200)
        from calibration_profiles import CalibrationProfiles

        restarted = CalibrationProfiles(self.root / "data", self.lab.calibration_profiles.validate)
        self.assertEqual(len(restarted.entries(self.lab.calibration_root, "follower")), 2)

    def test_mismatched_profile_never_rewrites_motors_or_acknowledges_revision(self):
        self.bus.calibration["gripper"]["homing_offset"] = 80
        before = self.path.read_bytes()
        with self.assertRaisesRegex(RequestError, "differs from the motors: gripper"):
            self.lab.calibration_reuse(self.payload())
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.bus.writes, [])
        self.assertFalse(self.lab.calibration_revisions.fresh("fake", "arm", "follower", self.path))

    def test_stop_during_read_does_not_publish_or_acknowledge(self):
        original = self.bus.read_calibration

        def interrupted():
            self.lab.generation += 1
            return original()

        self.bus.read_calibration = interrupted
        before = self.path.read_bytes()
        with self.assertRaisesRegex(RequestError, "cancelled by Stop"):
            self.lab.calibration_reuse(self.payload("motors"))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(self.lab.calibration_profiles.root.exists())
        self.assertEqual(self.bus.writes, [])

    def test_failed_acknowledgement_restores_previous_file(self):
        before = self.path.read_bytes()
        self.bus.calibration["gripper"]["range_min"] = 200
        with patch.object(
            self.lab.calibration_revisions, "acknowledge", side_effect=OSError("disk")
        ):
            with self.assertRaisesRegex(OSError, "disk"):
                self.lab.calibration_reuse(self.payload("motors"))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.bus.writes, [])

    def test_missing_confirmation_active_torque_and_factory_defaults_are_rejected(self):
        payload = self.payload("motors")
        with self.assertRaisesRegex(RequestError, "same physical arm"):
            self.lab.calibration_reuse({**payload, "same_arm": False})
        self.bus.read = lambda *_: 1
        with self.assertRaisesRegex(RequestError, "torque must be off"):
            self.lab.calibration_reuse(payload)
        self.bus.read = lambda *_: 0
        for entry in self.bus.calibration.values():
            entry.update(range_min=0, range_max=4095, homing_offset=0)
        with self.assertRaisesRegex(RequestError, "default full-range"):
            self.lab.calibration_reuse(payload)
        self.assertEqual(self.bus.writes, [])

    def test_unstable_read_and_unconnected_port_do_not_save(self):
        payload = self.payload("motors")
        with self.assertRaisesRegex(RequestError, "currently connected"):
            self.lab.calibration_reuse({**payload, "port": "unplugged"})
        other = copy.deepcopy(CALIBRATION)
        other["gripper"]["range_min"] = 120
        with patch.object(self.bus, "read_calibration", side_effect=[CALIBRATION, other]):
            with self.assertRaisesRegex(RequestError, "changed while reading"):
                self.lab.calibration_reuse(payload)
        self.assertFalse(self.lab.calibration_profiles.root.exists())

    def test_unrelated_safety_error_is_retained_and_pending_capture_blocks_reuse(self):
        self.lab.error = "Motor safety check failed: encoder outside its saved calibration range"
        self.lab.calibration_reuse(self.payload())
        self.assertIsNotNone(self.lab.error)
        self.lab.calibration_state["recoverable"] = True
        with self.assertRaisesRegex(RequestError, "Finish or cancel"):
            self.lab.calibration_reuse(self.payload())

    def test_history_ignores_invalid_files_symlinks_and_other_role(self):
        self.path.with_name("arm.backup-1.json").write_text(json.dumps(CALIBRATION))
        self.path.with_name("bad.json").write_text("broken")
        self.path.with_name("list.json").write_text("[]")
        self.path.with_name("linked.json").symlink_to(self.path)
        entries = self.lab.calibration_library("arm", "follower", "fake")["profiles"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(self.lab.calibration_library("arm", "leader")["profiles"], [])
        payload = self.payload()
        with self.assertRaisesRegex(RequestError, "changed or is unavailable"):
            self.lab.calibration_reuse({**payload, "profile_key": "../../private"})

    def open_bus(self, *_):
        self.bus.opened = True
        return self.bus

    def test_connection_offers_sources_and_remembers_the_arm_after_restart(self):
        self.lab.port_provider = lambda: [{"device": "fake", "serial_number": "arm-1"}]
        self.lab.calibration_revisions.acknowledge("fake", "arm", "follower", self.path)
        self.bus.calibration["gripper"]["homing_offset"] = 80
        start = dict(port="fake", robot_id="arm", mode="handguide", confirm="ENABLE MOTION")
        with self.assertRaisesRegex(RequestError, "Choose calibration for arm"):
            self.lab.control_start(start)
        options = self.lab.calibration_library("arm", "follower", "fake")
        self.assertEqual(options["conflict"]["joints"], ["gripper"])
        self.assertEqual(options["conflict"]["recommended"], "motors")
        self.assertTrue(options["needs_review"])
        self.lab.calibration_reuse({**self.payload("motors"), "remember": True})
        self.assertFalse(self.lab.calibration_library("arm", "follower", "fake")["needs_review"])
        self.lab.close()
        self.lab = RobotLab(
            self.root / "data",
            bus_factory=self.open_bus,
            camera=FakeCamera(),
            port_provider=lambda: [{"device": "fake", "serial_number": "arm-1"}],
            start_threads=False,
        )
        self.lab.calibration_root = self.root / "calibrations"
        self.bus.calibration["gripper"]["homing_offset"] = 95
        probe = self.lab.probe({"robot_id": "arm", "role": "follower", "port": "fake"})
        self.assertEqual(probe["profile"]["state"], "remembered")
        self.assertFalse(self.lab.calibration_library("arm", "follower", "fake")["needs_review"])
        result = self.lab.control_start(start)
        self.assertEqual(result["session"]["mode"], "handguide")
        self.assertEqual(self.lab.calibration["gripper"]["homing_offset"], 95)
        self.lab.stop()
        self.lab.port_provider = lambda: [{"device": "fake", "serial_number": "other-arm"}]
        self.bus.calibration["gripper"]["homing_offset"] = 113
        with self.assertRaisesRegex(RequestError, "Choose calibration for arm"):
            self.lab.control_start(start)

    def test_machine_choice_restores_a_profile_and_preserves_the_arm_copy(self):
        self.bus.calibration["gripper"]["homing_offset"] = 80
        before = copy.deepcopy(self.bus.calibration)
        self.lab.calibration_reuse({**self.payload(), "apply_to_arm": True, "remember": True})
        self.assertEqual(self.bus.calibration, CALIBRATION)
        self.assertFalse(any(reg == "Torque_Enable" and v for reg, _, v in self.bus.writes))
        entries = self.lab.calibration_profiles.entries(self.lab.calibration_root, "follower")
        self.assertIn(before, [entry["calibration"] for entry in entries])

    def test_factory_arm_recommends_machine_and_unchecked_choice_is_not_replayed(self):
        start = dict(port="fake", robot_id="arm", mode="handguide", confirm="ENABLE MOTION")
        self.lab.calibration_revisions.acknowledge("fake", "arm", "follower", self.path)
        for entry in self.bus.calibration.values():
            entry.update(range_min=0, range_max=4095, homing_offset=0)
        with self.assertRaisesRegex(RequestError, "Choose calibration"):
            self.lab.control_start(start)
        conflict = self.lab.calibration_library("arm", "follower", "fake")["conflict"]
        self.assertFalse(conflict["motors_available"])
        self.assertEqual(conflict["recommended"], "saved")
        self.lab.calibration_reuse({**self.payload(), "apply_to_arm": True, "remember": False})
        self.bus.calibration["gripper"]["range_min"] = 11
        with self.assertRaisesRegex(RequestError, "Choose calibration"):
            self.lab.control_start(start)

    def test_partially_written_machine_choice_restores_previous_arm_and_file(self):
        self.bus.calibration["gripper"]["homing_offset"] = 80
        before = copy.deepcopy(self.bus.calibration)
        old_file = self.path.read_bytes()
        original_write = self.bus.write_calibration

        def interrupted(calibration):
            original_write(calibration)
            if calibration != before:
                raise OSError("disconnected")

        with patch.object(self.bus, "write_calibration", side_effect=interrupted):
            with self.assertRaisesRegex(OSError, "disconnected"):
                self.lab.calibration_reuse(
                    {**self.payload(), "apply_to_arm": True, "remember": True}
                )
        self.assertEqual(self.bus.calibration, before)
        self.assertEqual(self.path.read_bytes(), old_file)
        self.assertIsNone(
            self.lab.calibration_library("arm", "follower", "fake")["remembered_source"]
        )

    def test_status_and_catalog_use_current_port_and_keep_arms_independent(self):
        leader_path = self.lab.calibration_path("leader", "leader")
        leader_path.parent.mkdir(parents=True)
        leader_path.write_text(json.dumps(CALIBRATION))
        self.lab.calibration_revisions.acknowledge("old-port", "arm", "follower", self.path)
        self.lab.calibration_revisions.mark_dirty("leader-port")
        devices = [
            dict(
                id="arm",
                robot_id="arm",
                name="Arm",
                role="follower",
                profile_role="follower",
                port="fake",
                serial_number=None,
                connected=True,
            ),
            dict(
                id="leader",
                robot_id="leader",
                name="Leader",
                role="leader",
                profile_role="leader",
                port="leader-port",
                serial_number=None,
                connected=True,
            ),
        ]
        with (
            patch.object(self.lab.device_registry, "read", return_value=devices),
            patch.object(self.lab.device_registry, "discover", return_value=devices),
        ):
            profiles = self.lab.status()["calibrations"]
            follower = next(p for p in profiles if p["robot_id"] == "arm")
            self.assertEqual(follower["setup_port"], "fake")
            self.assertFalse(follower["setup_fresh"])
            self.assertEqual(
                self.lab.devices()["saved_devices"][0]["calibration"], "needs_attention"
            )
            self.lab.calibration_reuse(self.payload())
            profiles = self.lab.status()["calibrations"]
            self.assertTrue(next(p for p in profiles if p["robot_id"] == "arm")["setup_fresh"])
            self.assertFalse(next(p for p in profiles if p["robot_id"] == "leader")["setup_fresh"])
            self.assertEqual(self.lab.devices()["saved_devices"][0]["calibration"], "saved")
            self.assertEqual(self.bus.writes, [])

    def test_role_alias_uses_original_profile_role(self):
        with patch.object(
            self.lab.device_registry,
            "read",
            return_value=[
                {"robot_id": "renamed", "calibration_id": "arm", "profile_role": "follower"}
            ],
        ):
            self.assertEqual(
                self.lab.calibration_library("renamed", "leader")["profiles"][0]["role"], "follower"
            )
            self.lab.calibration_reuse({**self.payload(), "robot_id": "renamed", "role": "leader"})
            self.assertEqual(self.lab.load_calibration("renamed", "leader", "fake"), CALIBRATION)


if __name__ == "__main__":
    unittest.main()
