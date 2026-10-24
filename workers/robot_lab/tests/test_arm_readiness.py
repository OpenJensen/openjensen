"""Read-only readiness and durable operator observations; no real hardware."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import RobotLab
from test_mapping_verification import save_mapping
from test_robot_lab import CALIBRATION, FakeBus, FakeCamera


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.buses = []

        def factory(*args):
            bus = FakeBus(*args)
            self.buses.append(bus)
            return bus

        self.factory = factory
        self.lab = self.new_lab()
        self.profile = self.lab.calibration_path("test-arm", "follower")
        self.profile.parent.mkdir(parents=True, exist_ok=True)
        self.profile.write_text(json.dumps(CALIBRATION))
        self.mapping = save_mapping(self.root / "data")

    def new_lab(self):
        lab = RobotLab(
            self.root / "data",
            bus_factory=self.factory,
            start_threads=False,
            camera=FakeCamera(),
            port_provider=lambda: [{"device": "/dev/fake"}],
        )
        lab.calibration_root = self.root / "calibration"
        return lab

    def tearDown(self):
        self.lab.close()
        self.temp.cleanup()

    def test_passive_status_validates_selected_profiles_and_ignores_backup_names(self):
        self.profile.with_name("test-arm.backup-123.json").write_text(self.profile.read_text())
        self.profile.with_name("բարև.json").write_text(self.profile.read_text())
        original = self.profile.read_bytes()
        state = self.lab.status()
        self.assertEqual([c["robot_id"] for c in state["calibrations"]], ["test-arm"])
        self.assertTrue(state["calibrations"][0]["valid"])
        self.assertEqual(self.buses, [])
        self.assertEqual(original, self.profile.read_bytes())
        self.profile.write_text('{"shoulder_pan":null}')
        self.assertFalse(self.lab.status()["calibrations"][0]["valid"])
        self.assertEqual(self.buses, [])

    def test_explicit_scan_compares_calibration_without_motor_writes(self):
        original = self.profile.read_bytes()
        result = self.lab.probe({"port": "/dev/fake", "robot_id": "test-arm", "role": "follower"})
        self.assertTrue(result["ready"])
        self.assertEqual(result["profile"]["state"], "matched")
        self.assertEqual(len(result["profile"]["joints"]), 6)
        self.assertEqual(self.buses[-1].writes, [])
        self.assertFalse(self.buses[-1].opened)
        self.assertEqual(original, self.profile.read_bytes())
        changed = copy.deepcopy(CALIBRATION)
        changed["elbow_flex"]["homing_offset"] += 1
        self.profile.write_text(json.dumps(changed))
        self.assertEqual(
            self.lab.probe({"port": "/dev/fake", "robot_id": "test-arm"})["profile"]["state"],
            "mismatch",
        )
        self.lab.calibration_revisions.mark_dirty("/dev/fake")
        self.assertEqual(
            self.lab.probe({"port": "/dev/fake", "robot_id": "test-arm"})["profile"]["state"],
            "stale",
        )
        self.assertTrue(all(not b.writes for b in self.buses))

    def test_missing_ids_never_claim_calibration_matches(self):
        def factory(*args):
            bus = self.factory(*args)
            del bus.motors[4]
            return bus

        self.lab.bus_factory = factory
        result = self.lab.probe({"port": "/dev/fake", "robot_id": "test-arm"})
        self.assertFalse(result["ready"])
        self.assertEqual(result["profile"]["state"], "unavailable")
        self.assertEqual(result["profile"]["joints"], {})
        self.assertEqual(self.buses[0].writes, [])

    def test_observation_is_bound_to_completed_run_and_mapping_revision(self):
        observed = {
            "phase": "done",
            "port": "/dev/fake",
            "joint": "gripper",
            "run_id": "run-one",
            "motion_active": False,
            "error": None,
        }
        self.lab.verification_state = observed
        self.lab.verification_revision = self.lab.mapping_checks.revision("/dev/fake")
        payload = {
            "port": "/dev/fake",
            "joint": "gripper",
            "run_id": "run-one",
            "correct": True,
            "confirm": "CONFIRM PHYSICAL JOINT",
        }
        for patch in ({"run_id": "stale"}, {"joint": "elbow_flex"}, {"correct": 1}):
            with self.assertRaises(ValueError):
                self.lab.mapping_confirm({**payload, **patch})
        self.lab.verification_state["phase"] = "stopped"
        with self.assertRaises(ValueError):
            self.lab.mapping_confirm(payload)
        self.lab.verification_state["phase"] = "done"
        original = self.mapping.read_bytes()
        self.lab.mapping_confirm(payload)
        self.lab.mapping_confirm(payload)  # idempotent lost-reply retry
        restored = self.new_lab()
        try:
            check = restored.status()["mapping_checks"]["checks"]["gripper"]
            self.assertTrue(check["correct"])
            self.assertEqual(check["run_id"], "run-one")
        finally:
            restored.close()
        with self.assertRaises(ValueError):
            self.lab.mapping_confirm({**payload, "correct": False})
        self.assertEqual(self.mapping.read_bytes(), original)
        record = json.loads(original)
        record["new_setup_revision"] = "changed"
        self.mapping.write_text(json.dumps(record))
        self.assertEqual(self.lab.mapping_checks.snapshot("/dev/fake")["checks"], {})
        with self.assertRaises(ValueError):
            self.lab.mapping_confirm(payload)
        self.assertEqual(self.buses, [])

    def test_invalid_identity_is_rejected_before_serial_open(self):
        for patch in ({"robot_id": "../bad"}, {"role": "bad"}):
            with self.assertRaises(ValueError):
                self.lab.probe({"port": "/dev/fake", "robot_id": "test-arm", **patch})
        self.assertEqual(self.buses, [])

    def test_new_completed_observation_can_report_wrong_joint(self):
        self.lab.verification_state = {
            "phase": "done",
            "port": "/dev/fake",
            "joint": "elbow_flex",
            "run_id": "run-two",
            "motion_active": False,
            "error": None,
        }
        self.lab.verification_revision = self.lab.mapping_checks.revision("/dev/fake")
        self.lab.mapping_confirm(
            {
                "port": "/dev/fake",
                "joint": "elbow_flex",
                "run_id": "run-two",
                "correct": False,
                "confirm": "CONFIRM PHYSICAL JOINT",
            }
        )
        restored = self.new_lab()
        try:
            self.assertFalse(restored.status()["mapping_checks"]["checks"]["elbow_flex"]["correct"])
        finally:
            restored.close()
        self.assertEqual(self.buses, [])
