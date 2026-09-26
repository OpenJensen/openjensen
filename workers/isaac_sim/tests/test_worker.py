import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import yaml

from sim_worker import service
from sim_worker.adapters import isaac
from sim_worker.adapters.artifacts import Artifacts
from sim_worker.adapters.manifest import load
from sim_worker.adapters.video import Recorder


_ROOT = Path(__file__).resolve().parents[1]
_DEMO = _ROOT / "demo.yaml"
_HAS_VIDEO_TOOLS = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
if os.environ.get("SIM_REQUIRE_VIDEO_TOOLS") == "1" and not _HAS_VIDEO_TOOLS:
    raise RuntimeError("The worker image must include ffmpeg and ffprobe")


class ManifestTests(unittest.TestCase):
    def test_demo(self):
        spec = load(_DEMO)
        self.assertEqual(spec.frames / spec.fps, 6)

    def test_invalid_inputs(self):
        cases = [
            ("capture", "width", 1279),
            ("capture", "height", 0),
            ("capture", "fps", True),
            ("capture", "frames", 18001),
            ("scene", "uri", "https://example.com/scene.usd"),
            ("scene", "camera", "/World/../Camera"),
            ("outputs", "uri", "gs://bucket/results?token=value"),
            ("capture", "unknown", 1),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            for section, field, value in cases:
                with self.subTest(field=field, value=value):
                    data = yaml.safe_load(_DEMO.read_text())
                    data[section][field] = value
                    path.write_text(yaml.safe_dump(data))
                    with self.assertRaises(ValueError):
                        load(path)

    def test_duplicate_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.yaml"
            path.write_text(_DEMO.read_text() + "api_version: unexpected\n")
            with self.assertRaisesRegex(ValueError, "unique"):
                load(path)


class WorkflowTests(unittest.TestCase):
    def test_publish_before_shutdown(self):
        events = []
        app = MagicMock()

        def shutdown(**kwargs):
            events.append("shutdown")
            raise SystemExit(kwargs.get("exit_code", 0))

        app.close.side_effect = shutdown
        sdk = SimpleNamespace(SimulationApp=MagicMock(return_value=app))
        with (
            patch.dict(sys.modules, {"isaacsim": sdk}),
            patch.dict(os.environ, {"ACCEPT_EULA": "Y"}),
            patch.object(isaac, "_capture", return_value=(frame for frame in [b"frame"])),
            patch.object(service, "Artifacts") as store,
            patch.object(service, "Recorder") as encoder,
        ):
            encoder.return_value.__exit__.side_effect = lambda *args: events.append("encoded")

            def complete():
                events.append("uploaded")
                return {"result_uri": "gs://test/run/result.json"}

            store.return_value.complete.side_effect = complete
            with self.assertRaises(SystemExit) as stopped:
                service.execute(_DEMO, Path("unused"))

        self.assertEqual(stopped.exception.code, 0)
        self.assertEqual(events, ["encoded", "uploaded", "shutdown"])

    def test_report_before_shutdown(self):
        app = MagicMock()

        def shutdown(**kwargs):
            raise SystemExit(kwargs.get("exit_code", 0))

        def broken_capture(spec, app):
            yield b"frame"
            raise RuntimeError("render failed")

        app.close.side_effect = shutdown
        sdk = SimpleNamespace(SimulationApp=MagicMock(return_value=app))
        with (
            patch.dict(sys.modules, {"isaacsim": sdk}),
            patch.dict(os.environ, {"ACCEPT_EULA": "Y"}),
            patch.object(isaac, "_capture", broken_capture),
            patch.object(service, "Artifacts") as store,
            patch.object(service, "Recorder"),
        ):
            with patch.object(service.logging, "exception"), self.assertRaises(SystemExit) as stopped:
                service.execute(_DEMO, Path("unused"))
            store.return_value.fail.assert_called_once_with("render failed")
            store.return_value.complete.assert_not_called()

        self.assertEqual(stopped.exception.code, 1)

    def test_failure_closes_simulator(self):
        closed = []

        def capture():
            yield b"frame"
            raise RuntimeError("render failed")

        @contextmanager
        def failing_frames(spec):
            try:
                yield capture()
            finally:
                closed.append(True)

        with patch.object(service, "Artifacts") as store, patch.object(service, "Recorder"):
            with patch.object(service.isaac, "frames", failing_frames):
                with self.assertLogs(level="ERROR"):
                    with self.assertRaisesRegex(RuntimeError, "render failed"):
                        service.execute(_DEMO, Path("unused"))
            store.return_value.complete.assert_not_called()
            store.return_value.fail.assert_called_once_with("render failed")
        self.assertEqual(closed, [True])

    def test_upload_before_gpu(self):
        with patch.object(service, "Artifacts") as store, patch.object(service.isaac, "frames") as frames:
            store.return_value.start.side_effect = PermissionError("denied")
            with self.assertLogs(level="ERROR"):
                with self.assertRaises(PermissionError):
                    service.execute(_DEMO, Path("unused"))
            frames.assert_not_called()

    def test_encoder_closes_simulator(self):
        closed = []

        @contextmanager
        def live_frames(spec):
            try:
                yield iter([b"frame", b"another"])
            finally:
                closed.append(True)

        with patch.object(service, "Artifacts"), patch.object(service, "Recorder") as recorder:
            recorder.return_value.__enter__.return_value.append.side_effect = RuntimeError("encoder failed")
            with patch.object(service.isaac, "frames", live_frames), self.assertLogs(level="ERROR"):
                with self.assertRaisesRegex(RuntimeError, "encoder failed"):
                    service.execute(_DEMO, Path("unused"))
        self.assertEqual(closed, [True])


class _PlaybackStarted(Exception):
    pass


class TimelineTests(unittest.TestCase):
    def test_demo_has_time_range(self):
        header = (_ROOT / "scenes/falling-cube.usda").read_text().split(")", 1)[0]
        metadata = dict(re.findall(r"(\w+)\s*=\s*([\d.]+)", header))
        start = float(metadata.get("startTimeCode", 0))
        end = float(metadata.get("endTimeCode", 0))
        self.assertGreater(end, start, "The demo has a single-frame playback range")

    def test_range_covers_capture(self):
        names = ("carb", "carb.settings", "omni", "omni.replicator", "omni.replicator.core",
                 "omni.timeline", "omni.usd", "pxr")
        for fps, frames, time_codes in ((30, 180, 30), (60, 1200, 24), (60, 1, 24)):
            with self.subTest(fps=fps, frames=frames, time_codes=time_codes):
                modules = {name: MagicMock() for name in names}
                for name in names:
                    if "." in name:
                        parent, child = name.rsplit(".", 1)
                        setattr(modules[parent], child, modules[name])
                stage = modules["omni.usd"].get_context.return_value.get_stage.return_value
                root = SimpleNamespace(startTimeCode=0.0, endTimeCode=0.0)
                stage.GetRootLayer.return_value = root
                stage.GetTimeCodesPerSecond.return_value = time_codes
                timeline = modules["omni.timeline"].get_timeline_interface.return_value
                spec = replace(load(_DEMO), fps=fps, frames=frames)

                def check_playback():
                    # Validate timing before playback, without loading a GPU SDK.
                    self.assertEqual(root.startTimeCode, 0.0)
                    self.assertGreater(root.endTimeCode / time_codes, frames / fps)
                    timeline.set_looping.assert_called_once_with(False)
                    stage.SetTimeCodesPerSecond.assert_not_called()
                    raise _PlaybackStarted

                timeline.play.side_effect = check_playback
                with patch.dict(sys.modules, modules), self.assertRaises(_PlaybackStarted):
                    next(isaac._capture(spec, MagicMock()))


class PublicationTests(unittest.TestCase):
    def test_publish_order(self):
        uploaded = []
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Artifacts(Path(directory), load(_DEMO))
            with patch.object(artifacts, "_upload", side_effect=lambda path, kind: uploaded.append(path.name)):
                artifacts.start()
                artifacts.video_path.write_bytes(b"video-fixture")
                result = artifacts.complete()
            self.assertEqual(uploaded, ["manifest.json", "video.mp4", "result.json"])
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(len(result["sha256"]), 64)
            saved = json.loads((artifacts.video_path.parent / "result.json").read_text())
            self.assertEqual(saved, result)

    def test_failed_upload_no_success(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Artifacts(Path(directory), load(_DEMO))
            artifacts.video_path.write_bytes(b"video-fixture")
            with patch.object(artifacts, "_upload", side_effect=PermissionError("denied")):
                with self.assertRaises(PermissionError):
                    artifacts.complete()
            self.assertFalse((artifacts.video_path.parent / "result.json").exists())


@unittest.skipUnless(_HAS_VIDEO_TOOLS, "ffmpeg/ffprobe unavailable locally; required in Cloud Build")
class VideoTests(unittest.TestCase):
    def test_video_round_trip(self):
        spec = replace(load(_DEMO), width=32, height=32, frames=12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.mp4"
            with Recorder(path, spec) as recorder:
                for index in range(spec.frames):
                    frame = bytes((index * 20, 80, 160)) * (spec.width * spec.height)
                    recorder.append(frame)
            self.assertGreater(path.stat().st_size, 0)

    def test_partial_video_removed(self):
        spec = replace(load(_DEMO), width=32, height=32, frames=12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.mp4"
            with self.assertRaisesRegex(RuntimeError, "Incomplete"):
                with Recorder(path, spec) as recorder:
                    recorder.append(bytes(spec.frame_bytes))
            self.assertFalse(path.exists())

    def test_bad_frame_removed(self):
        spec = replace(load(_DEMO), width=32, height=32, frames=12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.mp4"
            with self.assertRaises(ValueError):
                with Recorder(path, spec) as recorder:
                    recorder.append(b"truncated")
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
