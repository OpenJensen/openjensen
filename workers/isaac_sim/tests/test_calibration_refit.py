import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from sim_worker.calibration_fit import refit
from sim_worker.rollout.calibration import CalibrationUse, JointMap

_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
_SIGNS = (1, 1, 1, 1, -1)
_SUPPORT = np.array([[-20, 20]] * 5 + [[0, 50]], dtype=float)
_LIMITS = np.array([[-np.pi, np.pi]] * 5 + [[-0.2, 1.7]])
_PARAMETERS = np.array([0, 0, 0, 0, 0, 1, 1000, 0.1, -0.1, 0.2, 1, 0.02, 0.02])


class CalibrationRefitTests(unittest.TestCase):
    def test_negative_sign_roundtrip(self):
        result = {"parameters": _PARAMETERS, "signs": _SIGNS}
        document = refit._candidate_map({"support": _SUPPORT}, result)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.yaml"
            path.write_text(yaml.safe_dump(document))
            mapping = JointMap(path, _JOINTS, CalibrationUse.EXPERIMENTAL)
            states = np.random.default_rng(21).uniform(*_SUPPORT.T, size=(40, len(_JOINTS)))
            expected = refit._angles(states, _PARAMETERS, _SIGNS)
            actual = np.array([mapping.to_sim(tuple(row)) for row in states.tolist()])
            np.testing.assert_allclose(actual, expected, atol=1e-12)
            returned = [mapping.to_policy(tuple(row)) for row in actual.tolist()]
            np.testing.assert_allclose(returned, states, atol=1e-12)
            with self.assertRaises(ValueError):
                JointMap(path, _JOINTS)

    def test_gripper_support_bounds(self):
        lower, upper = refit._bounds(_SUPPORT, _LIMITS, _SIGNS)
        for seed in (lower, upper, (lower + upper) / 2):
            parameters = refit._physical(seed, _SUPPORT, _LIMITS)
            angles = refit._angles(_SUPPORT.T, parameters, _SIGNS)
            self.assertTrue(np.all(angles >= _LIMITS[:, 0] - 1e-12))
            self.assertTrue(np.all(angles <= _LIMITS[:, 1] + 1e-12))
            self.assertGreater(parameters[-1], 0)

    def test_lag_stays_in_episode(self):
        source = {
            (0, 0): {"observation.state": [0] * 6},
            (0, 1): {"observation.state": [2] * 6},
            (1, 2): {"observation.state": [50] * 6},
        }
        actual = refit._state({"episode": 0, "frame": 1}, source, -0.5)
        np.testing.assert_allclose(actual, [1] * 6)
        with self.assertRaisesRegex(ValueError, "episode boundary"):
            refit._state({"episode": 1, "frame": 2}, source, -0.5)

    def test_fit_excludes_holdout(self):
        train = [{"episode": 0}]
        truth = refit._encode(_PARAMETERS, _SUPPORT, _LIMITS, -2.5)
        context = {
            "train": train,
            "heldout": object(),
            "support": _SUPPORT,
            "limits": _LIMITS,
            "camera": truth[:7],
        }

        def residual(context, rows, values, signs, mode):
            self.assertIs(rows, train)
            return values - truth

        def metrics(context, rows, *args):
            self.assertIs(rows, train)
            return {"rms_px": 0.0}

        with (
            patch.object(refit, "_residual", side_effect=residual),
            patch.object(refit, "_metrics", side_effect=metrics),
        ):
            first = refit._fit(context, refit._Mode.FIXED, _SIGNS, [truth])
            context["heldout"] = {"malformed": "must never be read"}
            second = refit._fit(context, refit._Mode.FIXED, _SIGNS, [truth])
        np.testing.assert_allclose(first["parameters"], second["parameters"])

    def _source(self, directory):
        rows = []
        for episode in range(2):
            for frame in range(3):
                rows.append(
                    {
                        "episode_index": episode,
                        "frame_index": frame,
                        "index": episode * 3 + frame,
                        "timestamp": frame / 30,
                        "observation.state": [0.0] * 6,
                        "action": [1.0] * 6,
                    }
                )
        path = Path(directory) / "data.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        metadata = {
            "source_files": [
                {
                    "path": "data/chunk-000/file-000.parquet",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            ],
            "dataset": {"repo_id": "test/fixture", "revision": "pinned"},
            "confirmed_metadata": {
                "fps": 30,
                "frames": 6,
                "episodes": 2,
                "joint_order": [name + ".pos" for name in _JOINTS],
            },
            "episodes": [
                {
                    "episode": episode,
                    "frames": 3,
                    "data_rows_half_open": [episode * 3, episode * 3 + 3],
                }
                for episode in range(2)
            ],
        }
        provenance = Path(directory) / "provenance.json"
        provenance.write_text(json.dumps(metadata))
        return path, provenance, metadata

    def test_source_pin_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path, provenance, metadata = self._source(directory)
            source, audit = refit._recorded_source(path, provenance)
            self.assertEqual(len(source), 6)
            self.assertTrue(audit["episode_boundaries_verified"])
            metadata["episodes"][0]["data_rows_half_open"] = [0, 2]
            provenance.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "boundaries"):
                refit._recorded_source(path, provenance)
            metadata["source_files"][0]["sha256"] = "0" * 64
            provenance.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "SHA256"):
                refit._recorded_source(path, provenance)

    def test_joint_order_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, provenance, metadata = self._source(directory)
            metadata["confirmed_metadata"]["joint_order"].reverse()
            provenance.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "joint order"):
                refit._recorded_source(path, provenance)


if __name__ == "__main__":
    unittest.main()
