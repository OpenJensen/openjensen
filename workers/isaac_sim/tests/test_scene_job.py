"""Recorded scene jobs preserve every row and never claim calibrated task success."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import scene_job
from sim_worker.rollout.contracts import Frame

_RAW = {
    "episode_index": 3,
    "fps": 30,
    "joint_names": [name + ".pos" for name in scene_job.JOINTS],
    "mapping_status": scene_job.MAPPING,
    "frames": [
        {"timestamp": index / 30, "positions": [float(index), -104.0, 3, 4, 5, 1.5]}
        for index in range(6)
    ],
}


class TrajectoryTests(unittest.TestCase):
    def test_records_named_complete_rows_and_preserves_exceeding_nominal_values(self):
        result = scene_job.trajectory(copy.deepcopy(_RAW))
        self.assertEqual(len(result["frames"]), 6)
        self.assertEqual(result["frames"][0]["positions"][1], -104)
        self.assertEqual(result["joint_names"], scene_job.JOINTS)

    def test_reorders_named_joints_without_mixing_the_gripper(self):
        raw = copy.deepcopy(_RAW)
        raw["joint_names"].reverse()
        for frame in raw["frames"]:
            frame["positions"].reverse()
        self.assertEqual(scene_job.trajectory(raw), scene_job.trajectory(_RAW))

    def test_rejects_missing_samples_bad_cadence_and_unsafe_numbers(self):
        invalid = []
        for count in (0, 1):
            raw = copy.deepcopy(_RAW)
            raw["frames"] = raw["frames"][:count]
            invalid.append(raw)
        for stamp in (-1, 0.3, float("nan"), True):
            raw = copy.deepcopy(_RAW)
            raw["frames"][1]["timestamp"] = stamp
            invalid.append(raw)
        for value in (201, float("nan"), float("inf"), True):
            raw = copy.deepcopy(_RAW)
            raw["frames"][0]["positions"][0] = value
            invalid.append(raw)
        raw = copy.deepcopy(_RAW)
        raw["frames"].pop(2)
        invalid.append(raw)
        raw = copy.deepcopy(_RAW)
        raw["joint_names"][-1] = "shoulder_pan"
        invalid.append(raw)
        raw = copy.deepcopy(_RAW)
        raw["mapping_status"] = "calibrated"
        invalid.append(raw)
        raw = copy.deepcopy(_RAW)
        raw["fps"] = True
        invalid.append(raw)
        raw = copy.deepcopy(_RAW)
        raw["unexpected"] = "dataset instructions"
        invalid.append(raw)
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                scene_job.trajectory(raw)

    def test_bounded_rows_and_episode_duration(self):
        for count, fps in ((18_001, 60), (602, 1)):
            raw = copy.deepcopy(_RAW)
            raw["fps"] = fps
            raw["frames"] = [
                {"timestamp": index / fps, "positions": [0] * 6} for index in range(count)
            ]
            with self.subTest(count=count), self.assertRaises(ValueError):
                scene_job.trajectory(raw)

    def test_explicit_limit_mapping_reports_clamps(self):
        limits = tuple((-1.0, 2.0) for _ in range(6))
        values, clipped = scene_job.targets((-104, -100, 0, 100, 105, 0), limits)
        self.assertEqual(values, (-1, -1, 0.5, 2, 2, -1))
        self.assertEqual(clipped, (True, False, False, False, True, False))
        values, clipped = scene_job.targets((0, 0, 0, 0, 0, 102), limits)
        self.assertEqual(values[-1], 2)
        self.assertTrue(clipped[-1])
        for bad in (([0] * 5, limits), ([0] * 6, [(1, 1)] * 6)):
            with self.assertRaises(ValueError):
                scene_job.targets(*bad)


class _Simulation:
    instances = []

    def __init__(self, spec):
        self.spec = spec
        self.time = 0
        self.applied = []
        self.joint_limits = ((-1, 1),) * 6
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def reset(self, episode):
        self.episode = episode

    def environment_views(self, _):
        for name in ("front", "overview", "side"):
            yield name, Frame(2, 2, bytes([1, 25, 80]) * 4)

    def apply(self, targets):
        self.applied.append(targets)

    def step(self):
        self.time += 1 / self.spec.fps

    def observe(self):
        return SimpleNamespace(
            sim_time=self.time, state=self.applied[-1], frame=Frame(2, 2, bytes([1, 25, 80]) * 4)
        )


class _Recorder:
    instances = []

    def __init__(self, path, spec):
        self.path, self.spec, self.frames = path, spec, []
        self.instances.append(self)

    def __enter__(self):
        return self

    def append(self, frame):
        self.frames.append(frame)

    def __exit__(self, *args):
        self.path.write_bytes(b"synthetic test fixture, not an Isaac recording")


class SceneJobTests(unittest.TestCase):
    def setUp(self):
        self.temporary = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(self.temporary)
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        (self.bundle / "scene.usda").write_text("synthetic test scene")
        (self.bundle / "scene_config.json").write_text("{}")
        self.trajectory = self.root / "trajectory.json"
        self.trajectory.write_text(json.dumps(_RAW))
        self.output = self.root / "output"
        _Simulation.instances.clear()
        _Recorder.instances.clear()
        self.enterContext(patch.object(scene_job, "Recorder", _Recorder))
        self.enterContext(patch.object(scene_job, "_visible", lambda frame: frame))
        self.enterContext(
            patch.object(
                scene_job, "_image", lambda path, frame: path.write_bytes(b"synthetic image")
            )
        )

    def test_output_is_bound_to_source_rows_and_media_hashes(self):
        result = scene_job.run(
            self.bundle, self.trajectory, self.output, width=2, height=2, simulation=_Simulation
        )
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["source_frames"], 6)
        self.assertEqual(result["rendered_frames"], 6)
        self.assertEqual(len(_Recorder.instances[0].frames), 6)
        self.assertFalse(result["calibrated_replay"])
        self.assertFalse(result["calibration_verified"])
        self.assertIsNone(result["task_success"])
        self.assertEqual(result["mapping"]["clipping_counts"]["shoulder_lift"], 6)
        self.assertIn("warning", result)
        rows = [
            json.loads(row) for row in (self.output / "replay-trace.jsonl").read_text().splitlines()
        ]
        self.assertEqual([row["source_frame"] for row in rows], list(range(6)))
        self.assertEqual(
            [row["source_timestamp"] for row in rows], [row["timestamp"] for row in _RAW["frames"]]
        )
        self.assertEqual(len(result["media"]), 4)
        for media in result["media"]:
            self.assertEqual(media["sha256"], scene_job._sha(self.output / media["path"]))
        self.assertEqual(result["trajectory_sha256"], scene_job._sha(self.trajectory))
        self.assertEqual(
            json.loads((self.output / "result.json").read_text())["status"], "succeeded"
        )

    def test_existing_evidence_is_not_overwritten(self):
        self.output.mkdir()
        (self.output / "replay.mp4").write_bytes(b"previous recording")
        with self.assertRaisesRegex(ValueError, "existing evidence"):
            scene_job.run(
                self.bundle, self.trajectory, self.output, width=2, height=2, simulation=_Simulation
            )
        self.assertEqual((self.output / "replay.mp4").read_bytes(), b"previous recording")
        self.assertEqual(_Simulation.instances, [])

    def test_failure_publishes_no_success_claim(self):
        class BrokenClock(_Simulation):
            def step(self):
                self.time = 0

        with self.assertRaisesRegex(RuntimeError, "cadence"):
            scene_job.run(
                self.bundle, self.trajectory, self.output, width=2, height=2, simulation=BrokenClock
            )
        result = json.loads((self.output / "result.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("isaac_runtime", result)
        self.assertFalse(result["calibration_verified"])


if __name__ == "__main__":
    unittest.main()
