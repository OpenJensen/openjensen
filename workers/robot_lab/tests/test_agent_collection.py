"""Recorder failures must stop actuation and preserve honest partial evidence."""

import json
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_agent_control
import test_robot_lab as fixtures
from collection_mock import SyntheticCamera


class CollectionTests(unittest.TestCase):
    setUp = fixtures.SafetyTests.setUp
    tearDown = fixtures.SafetyTests.tearDown
    start = fixtures.SafetyTests.start
    command = test_agent_control.AgentTests.command

    def prepare(self):
        self.start()
        self.lab.camera = SyntheticCamera()
        self.command(
            "acquire",
            {
                "confirm": "ARM AI CONTROL",
                "expected_generation": self.lab.generation,
                "camera_index": 0,
                "operator_present": True,
                "workspace_clear": True,
            },
        )
        self.lab.agent_collection.start(
            {
                "run_id": "agent_test",
                "camera_indices": [0, 1],
                "fps": 10,
                "task": "Synthetic test",
                "environment_kind": "synthetic",
            }
        )

    def ticks(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and self.lab.mode == "agent":
            self.lab._tick()
            time.sleep(0.02)

    def test_missing_camera_mid_episode_stops_and_retains_partial(self):
        self.prepare()
        self.ticks(0.4)

        def missing(*args, **kwargs):
            raise ValueError("Camera disconnected")

        self.lab.camera.read_latest_timed = missing
        self.ticks(0.3)
        result = self.lab.agent_collection.finish({"run_id": "agent_test"})
        self.assertEqual(result["status"], "failed", result)
        self.assertIn("Camera disconnected", result["error"])
        self.assertEqual(self.lab.mode, "idle")
        self.assertFalse(self.lab.torque)
        self.assertGreaterEqual(result["frames"], 2)
        root = self.lab.agent_collection.root
        self.assertTrue((root / "raw/frames.jsonl").exists())
        self.assertTrue((root / "data/chunk-000/file-000.parquet").exists())

    def test_duplicate_command_is_not_reexecuted_and_evidence_matches(self):
        self.prepare()
        original = self.command("observe")["observation"]
        args = {
            "observation_id": original["observation_id"],
            "observation_sha256": original["observation_sha256"],
            "targets": {"shoulder_pan": original["joints"]["shoulder_pan"] + 1},
            "units": {"shoulder_pan": "degrees"},
            "duration_seconds": 1.0,
        }
        first = self.command("execute", args, ident="same-command")
        writes = len(self.buses[-1].writes)
        self.assertEqual(self.command("execute", args, ident="same-command"), first)
        self.assertEqual(len(self.buses[-1].writes), writes)
        self.ticks(1.2)
        self.lab.agent_collection.stop({"run_id": "agent_test"})
        state = self.lab.agent_collection.finish({"run_id": "agent_test"})
        self.assertEqual(state["status"], "saved", state)
        root = self.lab.agent_collection.root
        frames = [json.loads(x) for x in (root / "evidence/frames.jsonl").read_text().splitlines()]
        samples = {
            x["sample_id"]: x
            for x in map(json.loads, (root / "evidence/controller.jsonl").read_text().splitlines())
        }
        rows = list(map(json.loads, (root / "raw/frames.jsonl").read_text().splitlines()))
        self.assertEqual(len(frames), len(rows))
        for frame, row in zip(frames, rows):
            sample = samples[frame["controller_sample_id"]]
            self.assertLessEqual(sample["state_read_monotonic_ns"], frame["capture_monotonic_ns"])
            self.assertEqual(row["action"], list(sample["sent"].values()))
        self.assertTrue(any(f["command_id"] == "same-command" for f in frames))


if __name__ == "__main__":
    unittest.main()
