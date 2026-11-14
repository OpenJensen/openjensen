"""Mode discovery, persistent settings and atomic multi-camera admission."""

import errno
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera import CameraPool
from camera_modes import camera_identity, linux_modes

MODE = {"fourcc": "MJPG", "width": 1280, "height": 720, "fps": 30}
OLD = {"fourcc": "YUYV", "width": 800, "height": 600, "fps": 10}


class CameraModeTests(unittest.TestCase):
    def test_enumeration_is_read_only_and_preserves_fractional_rates(self):
        requests = []

        def query(fd, request, value, mutate):
            requests.append(request)
            index = struct.unpack_from("=I", value)[0]
            if index > 0:
                raise OSError(errno.EINVAL, "end")
            if request == 0xC0405602:
                struct.pack_into("=I", value, 44, int.from_bytes(b"MJPG", "little"))
            elif request == 0xC02C564A:
                struct.pack_into("=III", value, 8, 1, 1280, 720)
            elif request == 0xC034564B:
                struct.pack_into("=III", value, 16, 1, 1001, 30000)
            else:
                self.fail("Inventory must never start streaming")

        with (
            patch("os.open", return_value=7) as opened,
            patch("os.close") as closed,
            patch("fcntl.ioctl", side_effect=query),
        ):
            modes = linux_modes(4)
        self.assertEqual(modes, [{**MODE, "fps": 29.97003}])
        self.assertTrue(opened.call_args.args[1] & os.O_NONBLOCK)
        self.assertEqual(opened.call_args.args[1] & os.O_RDWR, 0)
        closed.assert_called_once_with(7)

    def test_identical_cameras_have_separate_port_settings_after_video_renumbering(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for port, index in [("1-3.1", 2), ("1-3.4", 4)]:
                device = root / port
                device.mkdir()
                (device / "idVendor").write_text("1234")
                (device / "idProduct").write_text("abcd")
                interface = device / "interface"
                interface.mkdir()
                node = root / f"video{index}"
                node.mkdir()
                (node / "device").symlink_to(interface)
            first = camera_identity(2, root)
            self.assertNotEqual(first, camera_identity(4, root))
            (root / "video2").rename(root / "video8")
            self.assertEqual(first, camera_identity(8, root))

    @patch("camera.platform.system", return_value="Linux")
    @patch("camera.camera_identity", side_effect=lambda index: f"camera-{index}")
    @patch("camera.linux_modes", return_value=[MODE, OLD])
    def test_group_settings_commit_only_after_every_camera_opens(self, *_):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "camera-settings.json"
            pool = CameraPool()
            pool.set_settings_path(path)
            with patch.object(pool, "read", return_value=object()) as read:
                pool.configure([{"index": 2, "mode": MODE}, {"index": 4, "mode": OLD}])
            self.assertEqual(read.call_count, 2)
            restored = CameraPool()
            restored.set_settings_path(path)
            self.assertEqual(restored.configured_modes, {"camera-2": MODE, "camera-4": OLD})
            before = path.read_bytes()

            def reject(index, **kwargs):
                if index == 4:
                    raise RuntimeError("USB bandwidth unavailable")

            with patch.object(pool, "read", side_effect=reject):
                with self.assertRaisesRegex(RuntimeError, "previous settings kept.*bandwidth"):
                    pool.configure([{"index": 2, "mode": OLD}, {"index": 4, "mode": MODE}])
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(pool.configured_modes, restored.configured_modes)
            self.assertFalse(pool.handles)

    @patch("camera.platform.system", return_value="Linux")
    @patch("camera.linux_modes", return_value=[MODE])
    def test_unadvertised_and_invalid_modes_never_open_a_camera(self, *_):
        pool = CameraPool()
        with patch.object(pool, "read") as read:
            for payload in [
                [{"index": True, "mode": MODE}],
                [{"index": 2, "mode": {**MODE, "fps": True}}],
                [{"index": 2, "mode": OLD}],
                [{"index": 2, "mode": MODE}, {"index": 2, "mode": MODE}],
            ]:
                with self.assertRaises(ValueError):
                    pool.configure(payload)
            read.assert_not_called()

    @patch("camera.platform.system", return_value="Linux")
    @patch("camera.camera_identity", return_value="camera-2")
    @patch("camera.linux_modes", return_value=[MODE])
    def test_camera_modes_can_change_while_preview_is_running(self, *_):
        import threading
        from unittest.mock import Mock

        pool = CameraPool()
        pool.readers[2] = (Mock(is_alive=lambda: True), threading.Event())
        pool.preview_readers.add(2)
        with patch.object(pool, "read", return_value=object()):
            pool.configure([{"index": 2, "mode": MODE}])
        self.assertEqual(pool.configured_modes, {"camera-2": MODE})
        self.assertFalse(pool.readers or pool.preview_readers)

    def test_camera_settings_are_not_changed_during_recording_capture(self):
        pool = CameraPool()
        pool.readers[2] = (object(), object())
        with (
            patch("camera.platform.system", return_value="Linux"),
            patch("camera.linux_modes", return_value=[MODE]),
            patch.object(pool, "close") as close,
        ):
            with self.assertRaisesRegex(RuntimeError, "Finish recording"):
                pool.configure([{"index": 2, "mode": MODE}])
            close.assert_not_called()


class CameraModeAdmissionTests(unittest.TestCase):
    def test_camera_changes_require_idle_and_a_finished_dataset(self):
        from types import SimpleNamespace
        from unittest.mock import Mock

        from server import RobotLab

        with tempfile.TemporaryDirectory() as directory:
            camera = SimpleNamespace(configure=Mock(return_value={"cameras": []}), close=Mock())
            lab = RobotLab(
                Path(directory), camera=camera, port_provider=lambda: [], start_threads=False
            )
            try:
                lab.mode = "manual"
                with self.assertRaisesRegex(ValueError, "Stop the current session"):
                    lab.configure_cameras({"cameras": [{"index": 2, "mode": MODE}]})
                lab.mode = "idle"
                lab.writer = SimpleNamespace(finalized=False)
                with self.assertRaisesRegex(ValueError, "Finish the current dataset"):
                    lab.configure_cameras({"cameras": [{"index": 2, "mode": MODE}]})
                camera.configure.assert_not_called()
                lab.writer = None
                self.assertEqual(
                    lab.configure_cameras({"cameras": [{"index": 2, "mode": MODE}]}),
                    {"cameras": []},
                )
                self.assertFalse(lab.camera_configuring)
                camera.close.assert_called_once()
                camera.configure.side_effect = RuntimeError("Mode refused")
                with self.assertRaisesRegex(RuntimeError, "Mode refused"):
                    lab.configure_cameras({"cameras": [{"index": 2, "mode": MODE}]})
                self.assertFalse(lab.camera_configuring)
            finally:
                lab.close()
