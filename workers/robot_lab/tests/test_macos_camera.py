"""Camera choices use the capture backend's inventory, never another tool's indices."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera import CameraPool, macos_cameras


class MacOSCameraTests(unittest.TestCase):
    def test_native_camera_indices_preserve_backend_order_and_identity(self):
        # FFmpeg reports the laptop first, while OpenCV sorts the USB unique ID first.
        devices = [("USB Camera", "0x1200000"), ("Laptop Camera", "6C707041")]
        with patch("camera._avfoundation_devices", return_value=devices):
            cameras = macos_cameras()
        assert [(c["index"], c["name"], c["unique_id"]) for c in cameras] == [
            (0, "USB Camera", "0x1200000"),
            (1, "Laptop Camera", "6C707041"),
        ]
        assert all(not c["verified"] for c in cameras)

    def test_macos_discovery_uses_native_inventory_without_subprocess_or_capture(self):
        cameras = [{"index": 0, "name": "USB Camera"}]
        with (
            patch("platform.system", return_value="Darwin"),
            patch("camera.macos_cameras", return_value=cameras),
            patch("subprocess.run", side_effect=AssertionError("No foreign camera index source")),
            patch.dict(
                sys.modules, cv2=SimpleNamespace(VideoCapture=Mock(side_effect=AssertionError))
            ),
        ):
            assert CameraPool().discover() == cameras

    def test_macos_inventory_is_dispatched_on_the_camera_main_thread(self):
        result = [{"index": 0, "name": "USB Camera"}]
        dispatch = SimpleNamespace(call=Mock(return_value=result))
        with patch("platform.system", return_value="Darwin"):
            assert CameraPool(dispatcher=dispatch).discover() == result
        dispatch.call.assert_called_once_with(macos_cameras, timeout=5)

    def test_unavailable_native_inventory_does_not_guess_indices_from_other_tools(self):
        with (
            patch("platform.system", return_value="Darwin"),
            patch("camera.macos_cameras", side_effect=RuntimeError("Native inventory unavailable")),
            patch("subprocess.run", side_effect=AssertionError("Must not guess an index")),
        ):
            with self.assertRaisesRegex(RuntimeError, "Native inventory unavailable"):
                CameraPool().discover()

    def test_native_camera_choices_remain_within_capture_index_limit(self):
        with patch(
            "camera._avfoundation_devices", return_value=[("Camera", str(i)) for i in range(70)]
        ):
            assert [entry["index"] for entry in macos_cameras()] == list(range(64))
