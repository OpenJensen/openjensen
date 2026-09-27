import json
import tempfile
import unittest
from pathlib import Path

from sim_worker.calibration_fit.recorded_policy import RecordedActions
from sim_worker.rollout.calibration import CalibrationUse, JointMap
from sim_worker.rollout.config import RolloutSpec
from sim_worker.rollout.contracts import CALIBRATION_VERSION, Frame, Observation, SimSpec
from sim_worker.rollout.experimental import MotionGuard
from sim_worker.rollout.service import Rollout

_SOURCE = "recorded:source-sha256:episode-1"
_FPS = 10
_JOINTS = ("arm",)


def _row(episode, frame, action):
    return {
        "episode_index": episode,
        "frame_index": frame,
        "action": [action],
        "observation.state": [99],
    }


def _observation(episode, step):
    return Observation(episode, step, step / _FPS, (99,), Frame(1, 1, bytes(3)))


class _Simulation:
    def reset(self, episode):
        self.episode = episode
        self.index = 0
        self.targets = []
        self.measured = (0.5, 0.8, 0.9)

    def observe(self):
        return Observation(
            self.episode,
            self.index,
            self.index / _FPS,
            (self.measured[self.index],),
            Frame(1, 1, bytes(3)),
        )

    def apply(self, target):
        self.targets.append(target)

    def step(self):
        self.index += 1


class RecordedPolicyTests(unittest.TestCase):
    def test_bounded_source_and_reset(self):
        rows = [_row(0, 0, 99), _row(1, 0, 1), _row(1, 1, 2), _row(1, 2, 3), _row(2, 0, 88)]
        policy = RecordedActions(rows, 1, _SOURCE, 2)
        rows[1]["action"][0] = 100
        with self.assertRaisesRegex(ValueError, "Reset"):
            policy.predict(_observation("first", 0))
        policy.reset("first")
        self.assertEqual(policy.predict(_observation("first", 0)).actions, ((1.0,), (2.0,)))
        last = policy.predict(_observation("first", 2))
        self.assertEqual(last.actions, ((3.0,),))
        self.assertEqual(last.model_id, _SOURCE)
        with self.assertRaisesRegex(ValueError, "exhausted"):
            policy.predict(_observation("first", 3))
        with self.assertRaisesRegex(ValueError, "Reset"):
            policy.predict(_observation("second", 0))
        with self.assertRaisesRegex(ValueError, "backwards"):
            policy.predict(_observation("first", 0))
        policy.reset("second")
        with self.assertRaisesRegex(ValueError, "zero"):
            policy.predict(_observation("second", 1))
        self.assertEqual(policy.predict(_observation("second", 0)).actions[0], (1.0,))

    def test_guard_uses_live_state(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "calibration.json"
            path.write_text(
                json.dumps(
                    {
                        "api_version": CALIBRATION_VERSION,
                        "status": "unverified",
                        "source": "test",
                        "joints": {"arm": {"sim_rad": [-3, 3], "policy": [-3, 3]}},
                    }
                )
            )
            mapping = JointMap(path, _JOINTS, CalibrationUse.EXPERIMENTAL)
            policy = RecordedActions([_row(1, 0, 1), _row(1, 1, 1)], 1, _SOURCE, 2)
            spec = RolloutSpec(
                SimSpec("unused", "/Camera", "/Robot", _JOINTS, 1, 1, _FPS, 120),
                2,
                2,
                "unused",
                _SOURCE,
                "recorded test",
                1,
                path,
            )
            simulation = _Simulation()
            guard = MotionGuard(lambda: ((-1.0, 1.0),), _FPS)
            records = []
            result = Rollout(spec, simulation, policy, mapping, guard).run(
                "run", lambda row, observation: records.append(row)
            )
            self.assertEqual(result["steps"], 2)
            self.assertAlmostEqual(simulation.targets[0][0], 0.7)
            self.assertAlmostEqual(simulation.targets[1][0], 1.0)
            self.assertEqual(records[0]["state_rad"], (0.5,))
            self.assertEqual(records[1]["state_rad"], (0.8,))
            self.assertEqual(records[0]["policy_action"], (1.0,))

    def test_reject_invalid_episode(self):
        for rows in (
            [],
            [_row(2, 0, 1)],
            [_row(1, 1, 1)],
            [_row(True, 0, 1)],
            [_row(1, False, 1)],
            [_row(1, 0, float("nan"))],
            [_row(1, 0, 1), _row(1, 0, 2)],
        ):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                RecordedActions(rows, 1, _SOURCE, 2)


if __name__ == "__main__":
    unittest.main()
