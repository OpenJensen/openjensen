import json
import tempfile
import unittest
from pathlib import Path

from sim_worker.rollout.calibration import CalibrationUse, JointMap
from sim_worker.rollout.config import RolloutSpec
from sim_worker.rollout.contracts import (
    CALIBRATION_VERSION,
    ActionChunk,
    Frame,
    Observation,
    SimSpec,
)
from sim_worker.rollout.experimental import MotionGuard
from sim_worker.rollout.service import Rollout

_JOINTS = ("arm", "gripper")
_MODEL = "experimental-checkpoint"
_FPS = 10


class _Simulation:
    def reset(self, episode):
        self.episode = episode
        self.index = 0
        self.state = (0.0, 0.4)
        self.targets = []

    def observe(self):
        return Observation(
            self.episode, self.index, self.index / _FPS, self.state, Frame(2, 2, bytes(12))
        )

    def apply(self, target):
        self.targets.append(target)
        self.state = target

    def step(self):
        self.index += 1


class _Policy:
    def reset(self, episode):
        pass

    def predict(self, observation):
        return ActionChunk(observation.episode_id, observation.step, _MODEL, ((4.0, -4.0),))


class ExperimentalTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "calibration.json"
        self.data = {
            "api_version": CALIBRATION_VERSION,
            "status": "unverified",
            "source": "visual candidate, not physically validated",
            "joints": {name: {"sim_rad": [-3, 3], "policy": [-30, 30]} for name in _JOINTS},
        }
        self._write()

    def _write(self):
        self.path.write_text(json.dumps(self.data))

    def test_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, "verified"):
            JointMap(self.path, _JOINTS)
        mapping = JointMap(self.path, _JOINTS, CalibrationUse.EXPERIMENTAL)
        self.assertEqual(mapping.status, "unverified")
        self.assertEqual(mapping.to_sim((10, 0)), (1.0, 0.0))

    def test_extrapolation_is_explicit(self):
        mapping = JointMap(self.path, _JOINTS, CalibrationUse.EXPERIMENTAL)
        self.assertEqual(mapping.to_sim((40, -40)), (4.0, -4.0))
        self.assertEqual(mapping.action_outside((40, 0)), (True, False))
        self.assertEqual(mapping.to_policy((4, -4)), (40.0, -40.0))
        self.data["status"] = "verified"
        self._write()
        with self.assertRaisesRegex(ValueError, "outside"):
            JointMap(self.path, _JOINTS).to_sim((40, 0))

    def test_invalid_data_stays_rejected(self):
        for status in ("invalid", True, None):
            self.data["status"] = status
            self._write()
            with self.subTest(status=status), self.assertRaises(ValueError):
                JointMap(self.path, _JOINTS, CalibrationUse.EXPERIMENTAL)
        self.data["status"] = "unverified"
        self.data["joints"]["arm"]["policy"] = [0, 0]
        self._write()
        with self.assertRaisesRegex(ValueError, "monotonic"):
            JointMap(self.path, _JOINTS, CalibrationUse.EXPERIMENTAL)

    def test_guards_limit_and_speed(self):
        guard = MotionGuard(lambda: ((-1.0, 1.0), (0.0, 1.0)), _FPS)
        decision = guard.constrain((4.0, -4.0), (0.0, 0.4))
        self.assertEqual(decision.target, (0.2, 0.2))
        self.assertEqual(decision.limit_clipped, (True, True))
        self.assertEqual(decision.speed_clipped, (True, True))
        for target in ((float("nan"), 0), (1,)):
            with self.subTest(target=target), self.assertRaises(ValueError):
                guard.constrain(target, (0.0, 0.4))

    def test_rollout_records_all_changes(self):
        self.data["joints"] = {name: {"sim_rad": [-3, 3], "policy": [-3, 3]} for name in _JOINTS}
        self._write()
        mapping = JointMap(self.path, _JOINTS, CalibrationUse.EXPERIMENTAL)
        spec = RolloutSpec(
            SimSpec("unused.usda", "/Camera", "/Robot", _JOINTS, 2, 2, _FPS, 120),
            2,
            1,
            "http://127.0.0.1:8080",
            _MODEL,
            "motion test",
            1,
            self.path,
        )
        sim = _Simulation()
        guard = MotionGuard(lambda: ((-1.0, 1.0), (0.0, 1.0)), _FPS)
        records = []
        result = Rollout(spec, sim, _Policy(), mapping, guard).run(
            "episode", lambda record, observation: records.append(record)
        )
        self.assertEqual(sim.targets, [(0.2, 0.2), (0.4, 0.0)])
        self.assertTrue(result["experimental"])
        self.assertEqual(result["calibration_status"], "unverified")
        self.assertEqual(result["action_extrapolations"], 4)
        self.assertEqual(result["joint_limit_clips"], 4)
        self.assertEqual(result["speed_limit_clips"], 3)
        self.assertEqual(records[0]["raw_target_rad"], (4.0, -4.0))
        self.assertEqual(records[0]["target_rad"], (0.2, 0.2))
        self.assertEqual(records[0]["action_extrapolated"], (True, True))

    def test_requires_guard_for_experiment(self):
        mapping = JointMap(self.path, _JOINTS, CalibrationUse.EXPERIMENTAL)
        with self.assertRaisesRegex(ValueError, "guard"):
            Rollout(None, _Simulation(), _Policy(), mapping)


if __name__ == "__main__":
    unittest.main()
