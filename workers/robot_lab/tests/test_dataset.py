import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import DatasetWriter, dataset_file, list_datasets, manifest, statistics
from hardware import JOINTS

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("numpy", "PIL", "pyarrow"))


@unittest.skipUnless(AVAILABLE, "NumPy, Pillow and PyArrow needed for real dataset format tests")
class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="robot-lab-format-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def record(self, images=True):
        import numpy as np

        writer = DatasetWriter(
            self.root, "pick-block", "Pick up the red block", 5, (24, 32, 3) if images else None
        )
        for episode in range(2):
            writer.start_episode()
            for index in range(5):
                positions = {
                    joint: float(index + axis + episode) for axis, joint in enumerate(JOINTS)
                }
                action = {joint: value + 0.5 for joint, value in positions.items()}
                image = (
                    np.full((24, 32, 3), [index * 20, episode * 40, 120], dtype=np.uint8)
                    if images
                    else None
                )
                writer.add_frame(positions, action, image, captured_at=100 + index / 5)
            writer.save_episode()
        writer.finalize()
        return writer

    def test_two_episodes_camera_data_metadata_and_hashes(self):
        import io

        import pyarrow.parquet as pq
        from PIL import Image

        writer = self.record()
        info = json.loads((writer.root / "meta/info.json").read_text())
        self.assertEqual(info["codebase_version"], "v3.0")
        self.assertEqual((info["total_frames"], info["total_episodes"]), (10, 2))
        rows = pq.read_table(writer.root / "data/chunk-000/file-000.parquet").to_pylist()
        self.assertEqual(rows[4]["timestamp"], float(__import__("numpy").float32(0.8)))
        self.assertEqual(rows[0]["observation.state"], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(rows[0]["action"], [0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        image = Image.open(io.BytesIO(rows[3]["observation.images.front"]["bytes"]))
        self.assertEqual(image.getpixel((0, 0)), (60, 0, 120))
        episodes = pq.read_table(
            writer.root / "meta/episodes/chunk-000/file-000.parquet"
        ).to_pylist()
        self.assertEqual(
            [(row["dataset_from_index"], row["dataset_to_index"]) for row in episodes],
            [(0, 5), (5, 10)],
        )
        description = manifest(self.root, writer.id)
        self.assertGreater(description["total_bytes"], 0)
        for entry in description["files"]:
            file = dataset_file(self.root, writer.id, entry["path"])
            self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(), entry["sha256"])
            self.assertEqual(file.stat().st_size, entry["size"])

    def test_image_stat_count_is_frames_and_sustained_cadence_checked(self):
        import numpy as np

        stats = statistics([np.zeros((64, 3)), np.ones((64, 3))], image=True)
        self.assertEqual(stats["count"], [2])
        self.assertEqual(stats["mean"], [[[0.5]], [[0.5]], [[0.5]]])
        writer = DatasetWriter(self.root, "slow-cadence", "Reach", 10)
        writer.start_episode()
        with self.assertRaisesRegex(ValueError, "Sustained"):
            for index in range(30):
                writer.add_frame(dict.fromkeys(JOINTS, 0.0), {}, captured_at=index * 0.12)

    def test_pending_episode_save_discard_and_finalization_guards(self):
        writer = DatasetWriter(self.root, "pending", "Reach", 5)
        writer.start_episode()
        writer.add_frame(dict.fromkeys(JOINTS, 0.0), {}, captured_at=0)
        with self.assertRaises(ValueError):
            writer.finalize()
        with self.assertRaises(ValueError):
            manifest(self.root, "pending")
        writer.discard_episode()
        self.assertEqual(writer.frames, 0)
        with self.assertRaises(ValueError):
            writer.finalize()

    def test_abandon_unsaved_recording_preserves_pending_camera_data(self):
        import numpy as np

        writer = DatasetWriter(self.root, "unsaved", "Reach", 20, (2, 3, 3))
        writer.start_episode()
        writer.add_frame(
            dict.fromkeys(JOINTS, 1.0), {}, np.ones((2, 3, 3), dtype=np.uint8), captured_at=0
        )
        before = {
            path.relative_to(writer.root): path.read_bytes()
            for path in writer.root.rglob("*")
            if path.is_file() and path.name != "openjensen.json"
        }
        writer.abandon_empty()
        self.assertTrue(writer.abandoned)
        self.assertFalse(writer.finalized)
        self.assertFalse(writer.episode_active)
        self.assertEqual(writer.frames, 1)
        for relative, contents in before.items():
            self.assertEqual((writer.root / relative).read_bytes(), contents)
        summary = next(item for item in list_datasets(self.root) if item["id"] == "unsaved")
        self.assertTrue(summary["abandoned"])
        self.assertEqual(summary["pending_frames"], 1)
        for action in (
            writer.start_episode,
            writer.save_episode,
            writer.discard_episode,
            writer.finalize,
        ):
            with self.assertRaises(ValueError):
                action()
        with self.assertRaises(ValueError):
            manifest(self.root, writer.id)
        self.assertEqual(
            (writer.pending / "frames.jsonl").read_bytes(), before[Path(".pending/frames.jsonl")]
        )

    def test_abandon_rejects_saved_episodes_without_changing_files_or_capture(self):
        writer = DatasetWriter(self.root, "saved", "Reach", 5)
        writer.start_episode()
        for index in range(2):
            writer.add_frame(dict.fromkeys(JOINTS, 1.0), {}, captured_at=index / 5)
        writer.save_episode()
        writer.start_episode()
        before = {path: path.read_bytes() for path in writer.root.rglob("*") if path.is_file()}
        with self.assertRaisesRegex(ValueError, "no saved episodes"):
            writer.abandon_empty()
        self.assertTrue(writer.episode_active)
        self.assertFalse(writer.abandoned)
        for path, contents in before.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_abandon_metadata_failure_pauses_and_preserves_retryable_writer(self):
        writer = DatasetWriter(self.root, "retry-abandon", "Reach", 5)
        writer.start_episode()
        writer.add_frame(dict.fromkeys(JOINTS, 1.0), {}, captured_at=0)
        pending = (writer.pending / "frames.jsonl").read_bytes()
        with patch("dataset.atomic_json", side_effect=OSError("disk unavailable")):
            with self.assertRaisesRegex(OSError, "disk unavailable"):
                writer.abandon_empty()
        self.assertFalse(writer.abandoned)
        self.assertFalse(writer.episode_active)
        self.assertEqual((writer.pending / "frames.jsonl").read_bytes(), pending)
        writer.abandon_empty()
        self.assertTrue(writer.abandoned)

    def test_background_image_error_keeps_episode_unsaved_and_discard_recovers(self):
        import numpy as np

        writer = DatasetWriter(self.root, "failed-image", "Reach", 5, (2, 3, 3))
        self.addCleanup(lambda: writer.image_writer.stop(ignore_error=True))
        writer.start_episode()
        with patch.object(writer.image_writer, "write", side_effect=OSError("disk full")):
            writer.add_frame(
                dict.fromkeys(JOINTS, 1.0), {}, np.zeros((2, 3, 3), dtype=np.uint8), captured_at=0
            )
            writer.image_writer.wait_until_done(ignore_error=True)
            with self.assertRaisesRegex(ValueError, "disk full"):
                writer.save_episode()
        self.assertEqual(writer.frames, 1)
        self.assertEqual(writer.episodes, [])
        self.assertTrue((writer.pending / "frames.jsonl").is_file())
        writer.discard_episode()
        writer.start_episode()
        for index in range(2):
            writer.add_frame(
                dict.fromkeys(JOINTS, 2.0),
                {},
                np.ones((2, 3, 3), dtype=np.uint8),
                captured_at=index / 5,
            )
        writer.save_episode()
        writer.finalize()
        self.assertEqual(writer.total_frames, 2)

    def test_save_retries_failed_images_and_keeps_every_recorded_frame(self):
        import threading

        import numpy as np
        import pyarrow.parquet as pq

        writer = DatasetWriter(self.root, "retry-image", "Reach", 5, (2, 3, 3))
        entered, release = threading.Event(), threading.Event()
        original = writer.image_writer.write
        first = [True]

        def temporary_failure(image, path):
            if first[0]:
                entered.set()
                release.wait(2)
                first[0] = False
                raise OSError("temporary image failure")
            original(image, path)

        writer.image_writer.write = temporary_failure
        try:
            writer.start_episode()
            writer.add_frame(
                dict.fromkeys(JOINTS, 1.0), {}, np.zeros((2, 3, 3), dtype=np.uint8), captured_at=0
            )
            self.assertTrue(entered.wait(1))
            writer.add_frame(
                dict.fromkeys(JOINTS, 2.0), {}, np.ones((2, 3, 3), dtype=np.uint8), captured_at=0.2
            )
            release.set()
            writer.image_writer.wait_until_done(ignore_error=True)
            writer.save_episode()
            writer.finalize()
            table = pq.read_table(writer.root / "data/chunk-000/file-000.parquet")
            self.assertEqual(len(table), 2)
            self.assertEqual(writer.total_frames, 2)
        finally:
            release.set()
            writer.image_writer.stop(ignore_error=True)

    def test_save_metadata_failure_preserves_a_retryable_pending_episode(self):
        writer = DatasetWriter(self.root, "retry-save", "Reach", 5)
        writer.start_episode()
        for index in range(2):
            writer.add_frame(dict.fromkeys(JOINTS, 1.0), {}, captured_at=index / 5)
        with patch.object(writer, "write_metadata", side_effect=OSError("metadata unavailable")):
            with self.assertRaisesRegex(OSError, "metadata unavailable"):
                writer.save_episode()
        self.assertEqual((writer.frames, writer.total_frames, writer.episodes), (2, 0, []))
        self.assertTrue((writer.pending / "frames.jsonl").is_file())
        writer.save_episode()
        writer.finalize()
        self.assertEqual(writer.total_frames, 2)
        self.assertEqual(len(writer.episodes), 1)

    def test_finalize_metadata_failure_does_not_publish_a_false_finalized_state(self):
        writer = DatasetWriter(self.root, "retry-finish", "Reach", 5)
        writer.start_episode()
        for index in range(2):
            writer.add_frame(dict.fromkeys(JOINTS, 1.0), {}, captured_at=index / 5)
        writer.save_episode()
        with patch.object(writer, "write_metadata", side_effect=OSError("metadata unavailable")):
            with self.assertRaises(OSError):
                writer.finalize()
        self.assertFalse(writer.finalized)
        self.assertTrue(writer.pending.is_dir())
        writer.finalize()
        self.assertTrue(writer.finalized)

    def test_unsafe_export_paths_and_symlinks_rejected(self):
        writer = self.record(False)
        for relative in (
            "../secret",
            "/etc/passwd",
            ".pending/frames.jsonl",
            "meta/../../secret",
            "meta\\info.json",
        ):
            with self.assertRaises(ValueError):
                dataset_file(self.root, writer.id, relative)
        (writer.root / "meta/link").symlink_to("/etc/passwd")
        with self.assertRaises(ValueError):
            dataset_file(self.root, writer.id, "meta/link")

    def test_short_scheduling_jitter_is_retained_with_measured_timing(self):
        writer = DatasetWriter(self.root, "jitter", "Reach", 10)
        writer.start_episode()
        for now in (0, 0.1, 0.31, 0.4, 0.5):
            writer.add_frame(dict.fromkeys(JOINTS, 0.0), {}, captured_at=now)
        writer.save_episode()
        summary = json.loads((writer.root / "openjensen.json").read_text())
        self.assertEqual(summary["fps"], 10)
        self.assertEqual(summary["capture_timing"][0]["samples"], 5)
        self.assertAlmostEqual(summary["capture_timing"][0]["elapsed_s"], 0.5)
        self.assertAlmostEqual(summary["capture_timing"][0]["largest_gap_s"], 0.21)
        writer.close()

    def test_capture_stalls_and_disk_exhaustion_stop_collection(self):
        writer = DatasetWriter(self.root, "stalled", "Reach", 10)
        writer.start_episode()
        writer.add_frame(dict.fromkeys(JOINTS, 0.0), {}, captured_at=0)
        with self.assertRaisesRegex(ValueError, "below requested FPS"):
            writer.add_frame(dict.fromkeys(JOINTS, 1.0), {}, captured_at=0.5)
        with patch("dataset.shutil.disk_usage") as usage:
            usage.return_value.free = 100
            with self.assertRaisesRegex(ValueError, "disk space"):
                writer.add_frame(dict.fromkeys(JOINTS, 1.0), {}, captured_at=0.1)

    @unittest.skipUnless(
        os.getenv("OPENJENSEN_VERIFY_LEROBOT") == "1", "Explicit official loader lane required"
    )
    def test_official_lerobot_044_loader_reads_camera_actions_and_delta_timestamps(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        writer = self.record()
        dataset = LeRobotDataset(
            repo_id="local/pick-block",
            root=writer.root,
            delta_timestamps={"action": [0.0, 0.2]},
            download_videos=False,
        )
        self.assertEqual(len(dataset), 10)
        self.assertEqual(dataset.num_episodes, 2)
        frame = dataset[0]
        self.assertEqual(tuple(frame["observation.images.front"].shape), (3, 24, 32))
        self.assertEqual(tuple(frame["action"].shape), (2, 6))
        self.assertEqual(frame["task"], "Pick up the red block")
        self.assertAlmostEqual(float(frame["action"][0, 0]), 0.5)
        self.assertAlmostEqual(float(dataset[5]["observation.state"][0]), 1.0)
        self.assertEqual(dataset.meta.info["codebase_version"], "v3.0")
        self.assertEqual(list(dataset.meta.tasks.index), ["Pick up the red block"])


if __name__ == "__main__":
    unittest.main()
