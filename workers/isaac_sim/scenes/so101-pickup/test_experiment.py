"""Check composed physical constraints and camera convention without Isaac."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

_ROOT = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("_experiment", _ROOT / "build_experiment.py")
_EXPERIMENT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EXPERIMENT)


class _ExperimentTests(unittest.TestCase):
    def setUp(self):
        self._folder = tempfile.TemporaryDirectory()
        self.addCleanup(self._folder.cleanup)
        self._output = Path(self._folder.name)
        self._report = _EXPERIMENT._build(self._output)
        self._stage = Usd.Stage.Open(str(self._output / "scene.experimental.usda"))

    def test_stage_metadata_matches(self):
        source = Usd.Stage.Open(str(_ROOT / "scene.usda"))
        self.assertEqual(
            UsdGeom.GetStageMetersPerUnit(self._stage), UsdGeom.GetStageMetersPerUnit(source)
        )
        self.assertEqual(UsdGeom.GetStageUpAxis(self._stage), UsdGeom.GetStageUpAxis(source))
        self.assertEqual(self._stage.GetFramesPerSecond(), source.GetFramesPerSecond())
        self.assertEqual(self._stage.GetTimeCodesPerSecond(), source.GetTimeCodesPerSecond())
        self.assertEqual(self._stage.GetStartTimeCode(), source.GetStartTimeCode())
        self.assertEqual(self._stage.GetEndTimeCode(), source.GetEndTimeCode())
        self.assertEqual(self._stage.GetDefaultPrim().GetPath(), source.GetDefaultPrim().GetPath())

    def test_initial_pose_keeps_limits(self):
        source = Usd.Stage.Open(str(_ROOT / "scene.usda"))
        clipped = []
        for name, entry in self._report["initial_joints"].items():
            path = f"/World/Robot/joints/{name}"
            joint = UsdPhysics.RevoluteJoint(self._stage.GetPrimAtPath(path))
            original = UsdPhysics.RevoluteJoint(source.GetPrimAtPath(path))
            lower, upper = joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()
            self.assertEqual(lower, original.GetLowerLimitAttr().Get())
            self.assertEqual(upper, original.GetUpperLimitAttr().Get())
            degrees = entry["applied_degrees"]
            self.assertGreaterEqual(degrees, lower - 1e-5)
            self.assertLessEqual(degrees, upper + 1e-5)
            self.assertAlmostEqual(
                UsdPhysics.DriveAPI(joint.GetPrim(), "angular").GetTargetPositionAttr().Get(),
                degrees, places=4,
            )
            if entry["clipped"]:
                clipped.append(name)

        self.assertEqual(clipped, ["shoulder_lift"])
        roll = self._report["initial_joints"]["wrist_roll"]["applied_degrees"]
        self.assertGreater(roll, 107)
        self.assertLess(roll, 109)

    def test_link_frames_match_drives(self):
        cache = UsdGeom.XformCache()
        for name, entry in self._report["initial_joints"].items():
            joint = UsdPhysics.RevoluteJoint(
                self._stage.GetPrimAtPath(f"/World/Robot/joints/{name}")
            )
            parent = self._stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
            child = self._stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])
            parent_matrix = np.asarray(cache.GetLocalToWorldTransform(parent)).T
            child_matrix = np.asarray(cache.GetLocalToWorldTransform(child)).T
            actual = np.linalg.inv(parent_matrix) @ child_matrix
            origin = joint.GetLocalRot0Attr().Get()
            local_rotation = Rotation.from_quat([
                *origin.GetImaginary(), origin.GetReal()
            ]).as_matrix()
            expected_rotation = local_rotation @ Rotation.from_euler(
                "z", entry["applied_rad"]
            ).as_matrix()
            np.testing.assert_allclose(actual[:3, 3], joint.GetLocalPos0Attr().Get(), atol=1e-7)
            np.testing.assert_allclose(actual[:3, :3], expected_rotation, atol=1e-6)

    def test_camera_matches_fit(self):
        camera = UsdGeom.Camera(self._stage.GetPrimAtPath("/World/Cameras/Front"))
        cache = UsdGeom.XformCache()
        camera_to_world = np.asarray(cache.GetLocalToWorldTransform(camera.GetPrim())).T
        robot_to_world = np.asarray(cache.GetLocalToWorldTransform(
            self._stage.GetPrimAtPath("/World/Robot")
        )).T
        point = np.array([0.2, -0.03, 0.17, 1.0])
        usd_point = np.linalg.inv(camera_to_world) @ robot_to_world @ point
        camera_report = self._report["camera"]
        fitted_point = (
            np.asarray(camera_report["robot_to_camera_rotation"]) @ point[:3]
            + camera_report["robot_to_camera_translation"]
        )
        np.testing.assert_allclose(usd_point[:3] * [1, -1, -1], fitted_point, atol=1e-9)
        focal_pixels = (
            camera.GetFocalLengthAttr().Get() / camera.GetHorizontalApertureAttr().Get()
            * camera_report["source_image_width"]
        )
        self.assertAlmostEqual(focal_pixels, camera_report["focal_pixels"], places=3)

    def test_mapping_stays_unverified(self):
        candidate = yaml.safe_load((_ROOT / "calibration.candidate.yaml").read_text())
        generated = yaml.safe_load((self._output / "calibration.experimental.yaml").read_text())
        self.assertEqual(generated["status"], "unverified")
        self.assertEqual(generated["joints"], candidate["joints"])
        report = json.loads((self._output / "experiment.json").read_text())
        self.assertFalse(report["urdf_limits_changed"])
        self.assertFalse(report["mapping_support_extended"])


if __name__ == "__main__":
    unittest.main()
