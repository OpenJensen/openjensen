"""Linux camera discovery never starts streaming; preview opens only the chosen node."""

import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera import CameraPool, linux_cameras


@unittest.skipUnless(sys.platform == "linux", "V4L2 is a Linux API")
class LinuxCameraTests(unittest.TestCase):
    def test_inventory_filters_metadata_sorts_indices_and_releases_descriptors(self):
        with tempfile.TemporaryDirectory() as root:
            sysfs = Path(root)
            for index in (12, 2, 3, 64):
                directory = sysfs / f"video{index}"
                directory.mkdir()
                (directory / "name").write_text(f"Camera {index}")
            (sysfs / "video-invalid").mkdir()
            closed = []

            def query(fd, request, capability):
                self.assertEqual(request, 0x80685600)
                # Per-device flags override aggregate capabilities: video3 is metadata only.
                struct.pack_into("=II", capability, 84, 0x80000001, 0x00800000 if fd == 3 else 1)

            with (
                patch("os.open", side_effect=lambda path, flags: int(path.name[5:])),
                patch("fcntl.ioctl", side_effect=query),
                patch("os.close", side_effect=closed.append),
            ):
                cameras = linux_cameras(sysfs)
            self.assertEqual([camera["index"] for camera in cameras], [2, 12])
            self.assertEqual(sorted(closed), [2, 3, 12])

    def test_unreadable_device_is_visible_and_disconnected_device_is_skipped(self):
        with tempfile.TemporaryDirectory() as root:
            sysfs = Path(root)
            (sysfs / "video0").mkdir()
            (sysfs / "video1").mkdir()

            def open_device(path, flags):
                if path.name == "video0":
                    raise PermissionError()
                raise FileNotFoundError()

            with patch("os.open", side_effect=open_device):
                self.assertEqual([entry["index"] for entry in linux_cameras(sysfs)], [0])

    def test_query_failure_closes_device_and_omits_it(self):
        with tempfile.TemporaryDirectory() as root:
            sysfs = Path(root)
            (sysfs / "video0").mkdir()
            with (
                patch("os.open", return_value=70),
                patch("fcntl.ioctl", side_effect=OSError()),
                patch("os.close") as close,
            ):
                self.assertEqual(linux_cameras(sysfs), [])
                close.assert_called_once_with(70)

    def test_high_index_preview_uses_v4l2_and_permission_error_releases_handle(self):
        calls, released = [], []
        capture = SimpleNamespace(isOpened=lambda: False, release=lambda: released.append(True))
        cv2 = SimpleNamespace(
            CAP_V4L2=200, VideoCapture=lambda *args: calls.append(args) or capture
        )
        with patch.dict(sys.modules, cv2=cv2), patch("platform.system", return_value="Linux"):
            with self.assertRaisesRegex(RuntimeError, "/dev/video12.*video group"):
                CameraPool().read(12)
        self.assertEqual(calls, [(12, 200)])
        self.assertEqual(released, [True])

    def test_invalid_camera_indices_are_rejected_before_open(self):
        with patch.dict(sys.modules, cv2=SimpleNamespace()):
            for value in (-1, 64, True, "0"):
                with self.assertRaises(ValueError):
                    CameraPool().read(value)
