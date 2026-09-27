import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from pxr import Gf, Usd, UsdGeom

from sim_worker.calibration_fit.replay import (
    KinematicReplay,
    LimitMode,
    RecordedStates,
    _evaluate,
)

_SCENE = Path(__file__).parents[1] / "scenes/so101-pickup"
_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.scene = self.root / "scene.usda"
        self.urdf = self.root / "robot.urdf"
        self.calibration = self.root / "calibration.yaml"
        self.landmarks = {"tip": {"link": "tip", "xyz": [1, 0, 0]}}
        self.urdf.write_text(
            "<robot><link name='base'/><link name='tip'/>"
            "<joint name='hinge' type='revolute'><parent link='base'/><child link='tip'/>"
            "<axis xyz='0 0 1'/><limit lower='-0.7853981633974483' "
            "upper='0.7853981633974483'/></joint></robot>"
        )
        self.mapping = {
            "api_version": "simulation.joints/v1alpha1",
            "status": "unverified",
            "source": "synthetic test",
            "joints": {
                "hinge": {"sim_rad": [-math.pi, math.pi], "policy": [-180, 180]},
            },
        }
        self.calibration.write_text(yaml.safe_dump(self.mapping))
        stage = Usd.Stage.CreateNew(str(self.scene))
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        UsdGeom.Xform.Define(stage, "/World")
        robot = UsdGeom.Xform.Define(stage, "/World/Robot")
        robot.AddTranslateOp().Set((0, 0, -4))
        camera = UsdGeom.Camera.Define(stage, "/World/Cameras/Front")
        camera.CreateFocalLengthAttr(36)
        camera.CreateHorizontalApertureAttr(36)
        camera.CreateVerticalApertureAttr(20.25)
        stage.GetRootLayer().Save()

    def _model(self):
        return KinematicReplay(
            scene=self.scene,
            urdf=self.urdf,
            calibration=self.calibration,
            landmarks=self.landmarks,
            image_size=(640, 360),
            joints=("hinge",),
        )

    def test_usd_axes_and_clipping(self):
        report = self._model().project((90,))
        np.testing.assert_allclose(report["requested_pixels"]["tip"], [320, 20], atol=1e-9)
        offset = 160 / math.sqrt(2)
        np.testing.assert_allclose(
            report["clipped_pixels"]["tip"], [320 + offset, 180 - offset], atol=1e-9
        )
        self.assertEqual(report["limit_clipped"], [True])
        self.assertAlmostEqual(report["requested_rad"][0], math.pi / 2)
        self.assertAlmostEqual(report["clipped_rad"][0], math.pi / 4)

    def test_projection_matches_gf(self):
        stage = Usd.Stage.Open(str(self.scene))
        camera = UsdGeom.Camera(stage.GetPrimAtPath("/World/Cameras/Front"))
        camera.GetHorizontalApertureOffsetAttr().Set(1.2)
        camera.GetVerticalApertureOffsetAttr().Set(-0.8)
        camera.AddTranslateOp().Set((0.2, -0.1, 0.5))
        camera.AddRotateXYZOp().Set((4, -7, 3))
        stage.GetRootLayer().Save()
        projection = self._model().project((0,))["requested_pixels"]["tip"]
        frustum = camera.GetCamera().frustum
        clip = Gf.Vec4d(1, 0, -4, 1) * (
            frustum.ComputeViewMatrix() * frustum.ComputeProjectionMatrix()
        )
        expected = [(clip[0] / clip[3] + 1) * 320, (1 - clip[1] / clip[3]) * 180]
        np.testing.assert_allclose(projection, expected, atol=1e-5)

    def test_parent_base_transform(self):
        stage = Usd.Stage.Open(str(self.scene))
        world = UsdGeom.Xformable(stage.GetPrimAtPath("/World"))
        world.AddTranslateOp().Set((10, 20, 30))
        world.AddRotateZOp().Set(45)
        stage.GetRootLayer().Save()
        np.testing.assert_allclose(
            self._model().project((0,))["requested_pixels"]["tip"], [480, 180], atol=1e-9
        )

    def test_frozen_inputs(self):
        model = self._model()
        before = model.project((0,))
        digest = model.provenance()["calibration_sha256"]
        self.mapping["joints"]["hinge"]["sim_rad"] = [-math.pi + 0.2, math.pi + 0.2]
        self.calibration.write_text(yaml.safe_dump(self.mapping))
        self.landmarks["tip"]["xyz"][0] = 2
        self.assertEqual(model.project((0,)), before)
        self.assertEqual(model.provenance()["calibration_sha256"], digest)
        self.assertNotEqual(self._model().project((0,)), before)
        metadata = model.provenance()
        metadata["principal_point"][0] += 50
        self.assertEqual(model.project((0,)), before)

    def test_frozen_camera(self):
        model = self._model()
        before = model.project((0,))
        stage = Usd.Stage.Open(str(self.scene))
        camera = UsdGeom.Camera(stage.GetPrimAtPath("/World/Cameras/Front"))
        camera.GetFocalLengthAttr().Set(45)
        stage.GetRootLayer().Save()
        self.assertEqual(model.project((0,)), before)
        self.assertNotEqual(self._model().project((0,)), before)

    def test_labeled_alignment(self):
        model = self._model()
        rows = [
            {
                "episode_index": 0,
                "frame_index": index,
                "index": index,
                "observation.state": [index * 90],
            }
            for index in range(2)
        ]
        records = RecordedStates(rows, ("hinge",))
        prediction = model.project((45,))["requested_pixels"]["tip"]
        observed = [prediction[0] + 3, prediction[1] + 4]
        labels = {
            "landmarks": self.landmarks,
            "samples": [
                {"episode": 0, "frame": frame, "split": "validation", "points": {"tip": observed}}
                for frame in range(2)
            ],
        }
        report = _evaluate(model, records, labels, [0], -0.5)
        self.assertEqual(report["errors"][0]["frame"], 0)
        self.assertEqual(len(report["rows"]), 1)
        self.assertEqual(report["rows"][0]["state"], [45])
        self.assertAlmostEqual(report["groups"]["validation"]["requested"]["rms_px"], 5)

    def test_bad_camera_and_mode(self):
        model = self._model()
        with self.assertRaisesRegex(ValueError, "LimitMode"):
            model.project_angles([0], True)
        with self.assertRaisesRegex(ValueError, "finite"):
            model.project_angles([math.nan], LimitMode.REQUESTED)
        stage = Usd.Stage.Open(str(self.scene))
        stage.GetPrimAtPath("/World/Robot").GetAttribute("xformOp:translate").Set((0, 0, 4))
        stage.GetRootLayer().Save()
        with self.assertRaisesRegex(ValueError, "front"):
            self._model().project((0,))

    def test_audit_not_camera_bound(self):
        report = self._model().audit_limits([[0], [90], [-90]])
        self.assertEqual(report["rows_with_limit_conflict"], 2)
        self.assertEqual(report["joints"]["hinge"]["limit_conflicts"], 2)
        with self.assertRaisesRegex(ValueError, "recorded"):
            self._model().audit_limits([])

    def test_matches_authored_fk(self):
        landmarks = json.loads((_SCENE / "evidence/calibration/landmarks.json").read_text())
        data = json.loads((_SCENE / "evidence/dataset.json").read_text())
        model = KinematicReplay(
            scene=_SCENE / "scene.experimental.usda",
            urdf=_SCENE / "robot/source/so101_new_calib.urdf",
            calibration=_SCENE / "calibration.experimental.yaml",
            landmarks=landmarks["landmarks"],
            image_size=(1920, 1080),
            joints=_JOINTS,
        )
        report = model.project(data["first_frame_raw"]["observation.state"])
        stage = Usd.Stage.Open(str(_SCENE / "scene.experimental.usda"))
        camera = UsdGeom.Camera(stage.GetPrimAtPath("/World/Cameras/Front")).GetCamera()
        transform = camera.frustum.ComputeViewMatrix() * camera.frustum.ComputeProjectionMatrix()
        cache = UsdGeom.XformCache()
        for name, landmark in landmarks["landmarks"].items():
            link = stage.GetPrimAtPath("/World/Robot/" + landmark["link"])
            world = cache.GetLocalToWorldTransform(link).Transform(Gf.Vec3d(*landmark["xyz"]))
            clip = Gf.Vec4d(*world, 1) * transform
            expected = [(clip[0] / clip[3] + 1) * 960, (1 - clip[1] / clip[3]) * 540]
            np.testing.assert_allclose(report["clipped_pixels"][name], expected, atol=1e-3)
        digest = hashlib.sha256((_SCENE / "calibration.experimental.yaml").read_bytes()).hexdigest()
        self.assertEqual(model.provenance()["calibration_sha256"], digest)


class RecordedStateTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {
                "episode_index": episode,
                "frame_index": frame,
                "index": episode * 3 + frame,
                "observation.state": [episode * 100 + frame * 10],
            }
            for episode in range(2)
            for frame in range(3)
        ]

    def test_interpolation_and_boundary(self):
        samples = RecordedStates(self.rows, ("hinge",))
        self.assertEqual(samples.sample(1, 1, -0.5), (105,))
        self.assertEqual(samples.sample(1, 0), (100,))
        with self.assertRaisesRegex(ValueError, "outside"):
            samples.sample(1, 0, -0.1)
        with self.assertRaisesRegex(ValueError, "outside"):
            samples.sample(0, 2, 0.1)
        self.assertEqual(samples.local_frame(1, 3), 0)
        self.assertEqual(samples.episodes, (0, 1))
        self.assertEqual(samples.frames(1), 3)
        with self.assertRaisesRegex(ValueError, "outside"):
            samples.local_frame(1, 2)

    def test_duplicate_missing_and_empty(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            RecordedStates([*self.rows, self.rows[0]], ("hinge",))
        with self.assertRaisesRegex(ValueError, "contiguous"):
            RecordedStates(self.rows[1:], ("hinge",))
        with self.assertRaisesRegex(ValueError, "empty"):
            RecordedStates([], ("hinge",))
        rows = copy.deepcopy(self.rows)
        rows[0]["observation.state"] = [math.nan]
        with self.assertRaisesRegex(ValueError, "finite"):
            RecordedStates(rows, ("hinge",))

    def test_input_copy(self):
        samples = RecordedStates(self.rows, ("hinge",))
        self.rows[0]["observation.state"][0] = 1000
        self.assertEqual(samples.sample(0, 0), (0,))


if __name__ == "__main__":
    unittest.main()
