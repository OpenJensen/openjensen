import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sim_worker.calibration_fit.action_replay import _recorded_source, audit_actions
from sim_worker.rollout.calibration import CalibrationUse, JointMap
from sim_worker.rollout.contracts import CALIBRATION_VERSION

_JOINTS = ("arm", "gripper")
_LIMITS = ((-1.0, 1.0), (0.0, 1.0))
_FPS = 10


def _row(episode, frame, source, state, action):
    return {
        "episode_index": episode,
        "frame_index": frame,
        "index": source,
        "timestamp": frame / _FPS,
        "observation.state": state,
        "action": action,
    }


class ActionReplayTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "calibration.json"
        path.write_text(
            json.dumps(
                {
                    "api_version": CALIBRATION_VERSION,
                    "status": "unverified",
                    "source": "test",
                    "joints": {name: {"sim_rad": [-3, 3], "policy": [-30, 30]} for name in _JOINTS},
                }
            )
        )
        self.mapping = JointMap(path, _JOINTS, CalibrationUse.EXPERIMENTAL)

    def _audit(self, rows):
        return audit_actions(rows, self.mapping, _JOINTS, _LIMITS, _FPS)

    def test_actual_map_and_guard(self):
        report, trace = self._audit([_row(1, 0, 150, [0, 4], [40, -40])])
        self.assertEqual(trace[0]["raw_target_rad"], (4, -4))
        for value in trace[0]["guarded_target_rad"]:
            self.assertAlmostEqual(value, 0.2)
        self.assertEqual(trace[0]["joint_limit_clipped"], (True, True))
        self.assertEqual(trace[0]["guard_offset_clipped"], (True, True))
        self.assertEqual(report["aggregate"]["counts"]["action_extrapolated"], 2)
        self.assertEqual(report["calibration_sha256"], self.mapping.digest)

    def test_recorded_state_each_tick(self):
        _, trace = self._audit(
            [
                _row(0, 0, 0, [0, 0], [10, 10]),
                _row(0, 1, 1, [8, 8], [10, 10]),
            ]
        )
        for value in trace[0]["guarded_target_rad"]:
            self.assertAlmostEqual(value, 0.2)
        for value in trace[1]["guarded_target_rad"]:
            self.assertAlmostEqual(value, 1.0)
        self.assertEqual(trace[1]["policy_action"], (10, 10))
        self.assertEqual(trace[1]["recorded_policy_state"], (8, 8))

    def test_reset_keeps_pairs_local(self):
        report, trace = self._audit(
            [
                _row(0, 0, 0, [0, 0], [0, 0]),
                _row(0, 1, 1, [1, 1], [1, 1]),
                _row(1, 0, 2, [9, 9], [9, 9]),
                _row(1, 1, 3, [8, 8], [8, 8]),
            ]
        )
        self.assertEqual([row["episode_reset"] for row in trace], [True, False, True, False])
        self.assertIsNone(trace[1]["recorded_next_state_rad"])
        self.assertIsNone(trace[3]["recorded_next_state_rad"])
        profile = report["aggregate"]["action_state_alignment"]["profile"]
        self.assertEqual(
            next(row["pairs"] for row in profile if row["state_offset_frames"] == 1), 2
        )
        self.assertEqual(trace[2]["control_timestamp_s"], 0)

    def test_state_limit_violation(self):
        report, trace = self._audit([_row(0, 0, 0, [-30, 0], [-10, 0])])
        self.assertEqual(trace[0]["state_outside_limits"], (True, False))
        self.assertEqual(trace[0]["guarded_target_rad"], (-1, 0))
        self.assertGreater(trace[0]["guarded_minus_state_rad"][0], report["guard_max_offset_rad"])

    def test_preserve_source_rows(self):
        rows = [_row(1, 0, 150, [0, 0], [10, 5]), _row(1, 1, 151, [2, 3], [4, 5])]
        original = copy.deepcopy(rows)
        _, trace = self._audit(rows)
        self.assertEqual(trace[0]["source_index"], 150)
        self.assertEqual(trace[0]["policy_action"], (10, 5))
        self.assertAlmostEqual(trace[0]["requested_minus_next_state_rad"][0], 0.8)
        self.assertEqual(rows, original)

    def test_reject_wrong_timestamps(self):
        for timestamp in (0.01, float("nan"), True):
            row = _row(0, 0, 0, [0, 0], [0, 0])
            row["timestamp"] = timestamp
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                self._audit([row])

    def test_reject_bad_boundaries(self):
        cases = [
            [_row(0, 1, 0, [0, 0], [0, 0])],
            [_row(0, 0, 0, [0, 0], [0, 0]), _row(0, 2, 1, [0, 0], [0, 0])],
            [
                _row(0, 0, 0, [0, 0], [0, 0]),
                _row(1, 0, 1, [0, 0], [0, 0]),
                _row(0, 0, 2, [0, 0], [0, 0]),
            ],
            [_row(0, 0, 0, [0, 0], [0, 0]), _row(0, 1, 0, [0, 0], [0, 0])],
        ]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self._audit(rows)

    def test_pinned_recorded_source(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder) / "recorded.parquet"
            dataset.write_bytes(b"pinned recorded bytes")
            provenance = Path(folder) / "dataset.json"
            rows = [_row(1, 0, 150, [2, 3], [5, 7])]
            metadata = {
                "dataset": {"repo_id": "test/recorded", "revision": "pinned"},
                "source_files": [
                    {
                        "path": "data/file.parquet",
                        "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                    }
                ],
                "confirmed_metadata": {
                    "joint_order": ["arm.pos", "gripper.pos"],
                    "frames": 1,
                    "fps": _FPS,
                },
                "episodes": [{"episode": 1, "frames": 1, "data_rows_half_open": [150, 151]}],
            }
            provenance.write_text(json.dumps(metadata))
            parquet = MagicMock()
            parquet.read_table.return_value.to_pylist.return_value = rows
            modules = {"pyarrow": SimpleNamespace(parquet=parquet), "pyarrow.parquet": parquet}
            with patch.dict("sys.modules", modules):
                actual, joints, fps, evidence = _recorded_source(dataset, provenance)
                self.assertEqual(actual[0]["action"], [5, 7])
                self.assertEqual(actual[0]["observation.state"], [2, 3])
                self.assertEqual(joints, _JOINTS)
                self.assertEqual(fps, _FPS)
                self.assertTrue(evidence["episode_boundaries_verified"])
                dataset.write_bytes(b"different source")
                parquet.read_table.reset_mock()
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    _recorded_source(dataset, provenance)
                parquet.read_table.assert_not_called()

    def test_reject_bad_data(self):
        for key, value in (
            ("action", [0]),
            ("observation.state", [0, float("inf")]),
            ("episode_index", True),
        ):
            row = _row(0, 0, 0, [0, 0], [0, 0])
            row[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self._audit([row])
        with self.assertRaisesRegex(ValueError, "empty"):
            self._audit([])


if __name__ == "__main__":
    unittest.main()
