"""Assembled setup lifecycle checks use a fake bus and no real serial adapter."""

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assembled_session import AssembledSession, CalibrationRevisions
from hardware import JOINTS
from server import RequestError, RobotLab
from test_robot_lab import CALIBRATION, FakeBus, FakeCamera


def until(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Background operation did not reach the expected state")


class Engine:
    instances = []

    def __init__(self, port, directory, *, stop_event, io_lock, progress, verified_so101):
        assert verified_so101 is True
        self.stop_event, self.io_lock, self.progress = stop_event, io_lock, progress
        self.opened = False
        self.ids = list(range(10, 16))
        self.raw = dict.fromkeys(self.ids, 2048)
        self.actions = []
        self.split_gate = None
        self.apply_entered = threading.Event()
        self.apply_gate = None
        self.baseline_read = threading.Event()
        self.instances.append(self)

    def split(self):
        if self.split_gate:
            self.split_gate.wait(2)
        with self.io_lock:
            if self.stop_event.is_set():
                raise ValueError("cancelled")
            self.opened = True
            self.actions.append("split")
        return self.ids

    def positions(self):
        with self.io_lock:
            if self.stop_event.is_set():
                raise ValueError("cancelled")
            positions = dict(self.raw)
            self.baseline_read.set()
            return positions

    def identify(self, motor_id, on_motion):
        self.actions.append(("identify", motor_id))
        on_motion(True)
        on_motion(False)
        return 24

    def apply(self, mapping):
        with self.io_lock:
            if self.stop_event.is_set():
                raise ValueError("cancelled")
            self.actions.append("begin-id-transaction")
            self.apply_entered.set()
            if self.apply_gate:
                self.apply_gate.wait(2)
            # Essential EEPROM re-lock is part of the already-started transaction.
            self.actions.append("relock")
        if self.stop_event.is_set():
            raise ValueError("cancelled")
        return list(range(1, 7))

    def close(self):
        self.opened = False


def detector(baseline, samples, *, assigned, now):
    from assembled import detect_movement

    return detect_movement(baseline, samples, assigned=assigned, now=now)


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.controllers = []

    def tearDown(self):
        for session in self.controllers:
            session.cancel()
        self.temp.cleanup()

    def session(self, **kwargs):
        session = AssembledSession(
            "/dev/fake",
            self.temp.name,
            lambda *args: None,
            engine_factory=Engine,
            detector=detector,
            verified_so101=True,
            **kwargs,
        )
        self.controllers.append(session)
        session.start()
        until(lambda: session.snapshot()["phase"] == "ready")
        return session

    def test_observation_requires_stable_motion_and_accepts_only_selected_fresh_candidate(self):
        session = self.session()
        session.observe(JOINTS[0])
        self.assertTrue(session.engine.baseline_read.wait(1))
        session.engine.raw[10] += 150
        until(lambda: session.snapshot()["detected_id"] == 10)
        with self.assertRaisesRegex(ValueError, "selected joint"):
            session.accept(JOINTS[1], 10)
        with self.assertRaisesRegex(ValueError, "current detected"):
            session.accept(JOINTS[0], 11)
        with session.lock:
            session.detected_at = time.monotonic() - 2
            with self.assertRaisesRegex(ValueError, "stale"):
                session.accept(JOINTS[0], 10)
            session.detected_at = time.monotonic()
            session.accept(JOINTS[0], 10)
        self.assertEqual(session.snapshot()["mapping"], {JOINTS[0]: 10})
        with self.assertRaisesRegex(ValueError, "already confirmed"):
            session.observe(JOINTS[0])
        with self.assertRaisesRegex(ValueError, "all six"):
            session.finish()
        self.assertEqual(session.engine.actions, ["split"])
        receipt = list((Path(self.temp.name) / "assembled").glob("*.mapping.json"))[0]
        self.assertEqual(json.loads(receipt.read_text())["mapping"], {JOINTS[0]: 10})

    def test_wiggle_requires_a_fresh_completed_action_and_maps_only_after_confirmation(self):
        session = self.session()
        with self.assertRaisesRegex(ValueError, "fresh wiggle"):
            session.accept_wiggle(JOINTS[5], 10)
        session.wiggle(10)
        until(lambda: session.snapshot()["wiggle_id"] == 10)
        self.assertEqual(session.snapshot()["mapping"], {})
        self.assertFalse(session.snapshot()["motion_active"])
        with self.assertRaisesRegex(ValueError, "fresh wiggle"):
            session.accept_wiggle(JOINTS[5], 11)
        session.wiggle_at = time.monotonic() - 61
        with self.assertRaisesRegex(ValueError, "fresh wiggle"):
            session.accept_wiggle(JOINTS[5], 10)
        session.wiggle_at = time.monotonic()
        session.accept_wiggle(JOINTS[5], 10)
        self.assertEqual(session.snapshot()["mapping"], {JOINTS[5]: 10})
        self.assertIsNone(session.snapshot()["wiggle_id"])
        with self.assertRaisesRegex(ValueError, "already"):
            session.wiggle(10)

    def test_no_measurable_wiggle_preserves_hand_identification(self):
        from identify import IdentificationUnavailable

        session = self.session()

        def unavailable(*args):
            raise IdentificationUnavailable("No measurable wiggle")

        session.engine.identify = unavailable
        session.wiggle(10)
        until(lambda: session.snapshot()["phase"] == "ready")
        self.assertIsNone(session.snapshot()["wiggle_id"])
        session.observe(JOINTS[0])
        self.assertEqual(session.snapshot()["phase"], "observing")

    def test_observation_timeout_is_retryable_and_clears_evidence(self):
        session = self.session(observe_timeout=0.1)
        session.observe(JOINTS[0])
        until(lambda: session.snapshot()["phase"] == "ready")
        self.assertIsNone(session.snapshot()["detected_id"])
        session.observe(JOINTS[1])
        self.assertEqual(session.snapshot()["selected_joint"], JOINTS[1])

    def test_stop_waits_for_current_transaction_relock_and_prevents_future_writes(self):
        session = self.session()
        with session.lock:
            session.state["mapping"] = dict(zip(JOINTS, session.engine.ids, strict=True))
        session.engine.apply_gate = threading.Event()
        session.finish()
        self.assertTrue(session.engine.apply_entered.wait(1))
        stopped = threading.Event()
        thread = threading.Thread(target=lambda: (session.cancel(), stopped.set()))
        thread.start()
        self.assertFalse(stopped.wait(0.05))
        session.engine.apply_gate.set()
        self.assertTrue(stopped.wait(1))
        thread.join(1)
        self.assertEqual(session.engine.actions, ["split", "begin-id-transaction", "relock"])
        self.assertEqual(session.snapshot()["phase"], "stopped")
        self.assertFalse(session.engine.opened)
        with self.assertRaises(ValueError):
            session.finish()
        time.sleep(0.05)
        self.assertEqual(session.engine.actions[-1], "relock")

    def test_finish_is_background_and_only_completes_after_canonical_verification(self):
        session = self.session()
        with session.lock:
            session.state["mapping"] = dict(zip(JOINTS, session.engine.ids, strict=True))
        session.engine.apply_gate = threading.Event()
        session.finish()
        self.assertTrue(session.engine.apply_entered.wait(1))
        self.assertEqual(session.snapshot()["phase"], "applying")
        session.engine.apply_gate.set()
        until(lambda: session.snapshot()["phase"] == "complete")
        self.assertEqual(session.snapshot()["ids"], list(range(1, 7)))
        self.assertFalse(session.engine.opened)


class ServerSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.lab = RobotLab(
            Path(self.temp.name),
            bus_factory=FakeBus,
            camera=FakeCamera(),
            port_provider=lambda: [{"device": "/dev/fake"}],
            start_threads=False,
            assembled_factory=Engine,
            movement_detector=detector,
        )
        self.lab.calibration_root = Path(self.temp.name) / "calibration"

    def tearDown(self):
        self.lab.close()
        self.temp.cleanup()

    def test_wiggle_route_requires_explicit_motion_and_observation_confirmations(self):
        self.start()
        until(lambda: self.lab.assembled_state["phase"] == "ready")
        for body in (
            {"motor_id": 10},
            {"motor_id": 10, "confirm": "WIGGLE MOTOR", "supported": False},
        ):
            with self.assertRaises(RequestError):
                self.lab.assembled_action("wiggle", body)
        self.assertEqual(self.lab.assembled_session.engine.actions, ["split"])
        self.lab.assembled_action(
            "wiggle", {"motor_id": 10, "confirm": "WIGGLE MOTOR", "supported": True}
        )
        until(lambda: self.lab.assembled_state["wiggle_id"] == 10)
        with self.assertRaises(RequestError):
            self.lab.assembled_action(
                "accept", {"joint": JOINTS[5], "motor_id": 10, "method": "wiggle"}
            )
        self.lab.assembled_action(
            "accept",
            {
                "joint": JOINTS[5],
                "motor_id": 10,
                "method": "wiggle",
                "confirm": "CONFIRM OBSERVED JOINT",
            },
        )
        self.assertEqual(self.lab.assembled_state["mapping"], {JOINTS[5]: 10})

    def start(self, **overrides):
        return self.lab.assembled_start(
            {
                "port": "/dev/fake",
                "supported": True,
                "confirm": "START ASSEMBLED SETUP",
                **overrides,
            }
        )

    def test_explicit_attestation_required_and_active_setup_excludes_other_hardware(self):
        for overrides in ({"supported": False}, {"supported": 1}, {"confirm": "yes"}):
            with self.assertRaises(RequestError):
                self.start(**overrides)
        self.start()
        self.assertEqual(self.lab.status()["session"]["mode"], "setup")
        self.assertEqual(self.lab.status()["session"]["watchdog_seconds"], 10)
        for action in (
            lambda: self.start(),
            lambda: self.lab.probe({"port": "/dev/fake"}),
            lambda: self.lab.calibration_start(
                {
                    "port": "/dev/fake",
                    "robot_id": "arm",
                    "role": "follower",
                    "confirm": "START CALIBRATION",
                }
            ),
        ):
            with self.assertRaisesRegex(RequestError, "Stop the current"):
                action()
        result = self.lab.stop()
        self.assertEqual(result["assembled"]["phase"], "stopped")
        self.assertEqual(result["session"]["mode"], "idle")
        self.assertTrue(result["assembled"]["calibration_required"])

    def test_old_calibration_preserved_but_refused_after_cancel_until_new_save(self):
        path = self.lab.calibration_path("arm", "follower")
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(CALIBRATION))
        before = path.read_bytes()
        self.assertEqual(self.lab.load_calibration("arm", "follower", "/dev/fake"), CALIBRATION)
        self.start()
        self.lab.stop()
        self.assertEqual(path.read_bytes(), before)
        with self.assertRaisesRegex(RequestError, "Motor IDs changed"):
            self.lab.control_start(
                {
                    "port": "/dev/fake",
                    "robot_id": "arm",
                    "mode": "manual",
                    "confirm": "ENABLE MOTION",
                }
            )
        # The actual successful calibration-finish path acknowledges the saved file.
        self.lab.calibration_start(
            {
                "port": "/dev/fake",
                "robot_id": "arm",
                "role": "follower",
                "confirm": "START CALIBRATION",
            }
        )
        self.lab.calibration_center({})
        self.lab.calibration_state["mins"] = dict.fromkeys(JOINTS, 100)
        self.lab.calibration_state["maxes"] = dict.fromkeys(JOINTS, 4000)
        result = self.lab.calibration_finish({"confirm": "SAVE CALIBRATION"})
        self.assertFalse(result["assembled"]["calibration_required"])
        self.assertTrue(self.lab.calibration_revisions.fresh("/dev/fake", "arm", "follower", path))
        persisted_bus = FakeBus()
        persisted_bus.calibration = json.loads(path.read_text())
        self.lab.bus_factory = lambda *args: persisted_bus
        self.lab.control_start(
            {"port": "/dev/fake", "robot_id": "arm", "mode": "manual", "confirm": "ENABLE MOTION"}
        )

    def test_revision_fsync_failure_blocks_setup_before_engine_construction_or_writes(self):
        before = len(Engine.instances)
        with patch("assembled_session.os.fsync", side_effect=OSError("disk sync failed")):
            with self.assertRaisesRegex(OSError, "disk sync failed"):
                self.start()
        self.assertEqual(len(Engine.instances), before)
        self.assertEqual(self.lab.mode, "idle")
        self.assertIsNone(self.lab.assembled_session)

    def test_stop_reports_post_reboot_cleanup_failure_and_closes_session(self):
        self.start()
        controller = self.lab.assembled_session
        until(lambda: controller.snapshot()["phase"] == "ready")

        def failed_cleanup():
            controller.engine.opened = False
            raise RuntimeError("USB serial cleanup failed")

        controller.engine.close = failed_cleanup
        result = self.lab.stop()
        self.assertEqual(result["session"]["mode"], "idle")
        self.assertEqual(result["assembled"]["phase"], "stopped")
        self.assertIn("cleanup failed", result["assembled"]["error"])
        self.assertIn("Stop attempted", result["session"]["error"])
        self.assertFalse(controller.engine.opened)

    def test_revisions_are_durable_and_profile_specific_and_fail_closed(self):
        path = Path(self.temp.name) / "profile.json"
        path.write_text("{}")
        revisions = CalibrationRevisions(self.temp.name)
        revisions.mark_dirty("/dev/fake")
        revisions.acknowledge("/dev/fake", "arm", "follower", path)
        reloaded = CalibrationRevisions(self.temp.name)
        self.assertTrue(reloaded.fresh("/dev/fake", "arm", "follower", path))
        self.assertFalse(reloaded.fresh("/dev/fake", "other", "follower", path))
        path.write_text('{"changed":true}')
        self.assertFalse(reloaded.fresh("/dev/fake", "arm", "follower", path))
        revisions.path.write_text('{"version":1,"ports":{"bad":null}}')
        with self.assertRaises(ValueError):
            reloaded.fresh("/dev/fake", "arm", "follower", path)

    def test_remembering_a_name_does_not_invalidate_other_unmodified_profiles(self):
        path = Path(self.temp.name) / "profile.json"
        path.write_text("{}")
        revisions = CalibrationRevisions(self.temp.name)
        revisions.acknowledge("/dev/fake", "arm", "follower", path)
        restored = CalibrationRevisions(self.temp.name)
        self.assertEqual(restored.connection("arm", "follower", path), "/dev/fake")
        self.assertTrue(restored.fresh("/dev/fake", "other-arm", "follower", path))
        self.assertFalse(restored.required("/dev/fake"))
        revisions.mark_dirty("/dev/fake")
        self.assertEqual(restored.connection("arm", "follower", path), "/dev/fake")
        self.assertFalse(restored.fresh("/dev/fake", "arm", "follower", path))
        path.write_text('{"changed":true}')
        self.assertIsNone(restored.connection("arm", "follower", path))
