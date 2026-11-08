"""Calibration recovery uses fake buses; it must never write to motors."""

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
            bus_factory=lambda *_: self.bus,
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
