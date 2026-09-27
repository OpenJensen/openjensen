"""Keep episode geometry, initial state, and rollout timing consistent."""

import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

_ROOT = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("_episode", _ROOT / "build_episode.py")
_EPISODE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EPISODE)
_CUP = "/World/Props/Cup"
_BOX = "/World/Props/Box"
_CAMERA = "/World/Cameras/Front"
_ROBOT = "/World/Robot"
_SOURCE_FILES = (
    "scene.usda", "scene.experimental.usda", "scene_config.json",
    "calibration.experimental.yaml",
)
_INITIAL_STATE = [
    4.351648330688477, -104.13186645507812, 90.5054931640625,
    75.82417297363281, 4.263736248016357, 1.3577733039855957,
]
_CONTROL = {"fps": 30, "physics_hz": 120, "steps": 150, "execute_steps": 50}
_FIXTURE_CUP = {
    "position": [-0.23, -0.06, 0.2005], "height": 0.088,
    "radius_bottom": 0.021, "radius_top": 0.03, "thickness": 0.0012,
    "mass": 0.012,
}
_TOLERANCE = 1e-7


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _EpisodeTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self._folder = Path(folder.name)
        self._profile = self._folder / "episode-001.json"
        self._profile.write_text(json.dumps({
            "episode": 1,
            "initial_state": _INITIAL_STATE,
            "cup": _FIXTURE_CUP,
            "control": _CONTROL,
            "source": {"dataset": "fixture", "episode": 1, "frame_index": 0},
        }))
        self._before = {name: _digest(_ROOT / name) for name in _SOURCE_FILES}
        self._output = self._folder / "generated"
        _EPISODE._build(self._profile, self._output)
        self._stage = Usd.Stage.Open(str(self._output / "scene.episode-001.usda"))
        self._report = json.loads((self._output / "episode-001-report.json").read_text())

    def test_cup_visual_and_colliders(self):
        # Rebuild every contact surface when changing the visible cup dimensions.
        paper = UsdGeom.Mesh(self._stage.GetPrimAtPath(f"{_CUP}/Paper"))
        points = np.asarray(paper.GetPointsAttr().Get())
        self.assertAlmostEqual(points[:, 2].min(), 0, places=7)
        self.assertAlmostEqual(points[:, 2].max(), _FIXTURE_CUP["height"], places=7)
        for height, radius in ((0, _FIXTURE_CUP["radius_bottom"]),
                               (_FIXTURE_CUP["height"], _FIXTURE_CUP["radius_top"])):
            ring = points[np.isclose(points[:, 2], height, atol=_TOLERANCE)]
            self.assertAlmostEqual(np.linalg.norm(ring[:, :2], axis=1).max(), radius, places=7)

        walls = list(self._stage.GetPrimAtPath(f"{_CUP}/Collision").GetChildren())
        self.assertEqual(len(walls), 33)
        for wall in walls:
            self.assertTrue(UsdPhysics.CollisionAPI(wall).GetCollisionEnabledAttr().Get())
            self.assertEqual(UsdGeom.Imageable(wall).GetPurposeAttr().Get(), "guide")
            if not wall.IsA(UsdGeom.Mesh):
                continue

            vertices = np.asarray(UsdGeom.Mesh(wall).GetPointsAttr().Get())
            self.assertAlmostEqual(vertices[:, 2].max(), _FIXTURE_CUP["height"], places=7)
            self.assertAlmostEqual(
                np.linalg.norm(vertices[:, :2], axis=1).max(),
                _FIXTURE_CUP["radius_top"], places=7,
            )
        bottom = UsdGeom.Cylinder(self._stage.GetPrimAtPath(f"{_CUP}/Collision/Bottom"))
        self.assertAlmostEqual(bottom.GetRadiusAttr().Get(), _FIXTURE_CUP["radius_bottom"])
        self.assertAlmostEqual(bottom.GetHeightAttr().Get(), _FIXTURE_CUP["thickness"])

    def test_cup_mass_and_support(self):
        cup = self._stage.GetPrimAtPath(_CUP)
        self.assertTrue(UsdPhysics.RigidBodyAPI(cup).GetRigidBodyEnabledAttr().Get())
        self.assertFalse(UsdPhysics.RigidBodyAPI(cup).GetKinematicEnabledAttr().Get())
        matrix = np.asarray(UsdGeom.Xformable(cup).GetLocalTransformation())
        np.testing.assert_allclose(matrix[:3, :3], np.eye(3), atol=_TOLERANCE)
        np.testing.assert_allclose(matrix[3, :3], _FIXTURE_CUP["position"], atol=_TOLERANCE)

        mass = UsdPhysics.MassAPI(cup)
        self.assertAlmostEqual(mass.GetMassAttr().Get(), _FIXTURE_CUP["mass"])
        np.testing.assert_allclose(
            mass.GetCenterOfMassAttr().Get(), [0, 0, _FIXTURE_CUP["height"] / 2],
            atol=_TOLERANCE,
        )
        radius = (_FIXTURE_CUP["radius_bottom"] + _FIXTURE_CUP["radius_top"]) / 2
        inertia_xy = _FIXTURE_CUP["mass"] * (radius**2 / 2 + _FIXTURE_CUP["height"]**2 / 12)
        np.testing.assert_allclose(
            mass.GetDiagonalInertiaAttr().Get(),
            [inertia_xy, inertia_xy, _FIXTURE_CUP["mass"] * radius**2], rtol=1e-6,
        )
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        board = self._stage.GetPrimAtPath("/World/Environment/Shelf/Board0")
        support = cache.ComputeWorldBound(board).ComputeAlignedRange()
        self.assertAlmostEqual(matrix[3, 2] - support.GetMax()[2], 0.0005, places=5)

    def test_label_matches_taper(self):
        label = self._stage.GetPrimAtPath(f"{_CUP}/Label")
        self.assertFalse(label.HasAPI(UsdPhysics.CollisionAPI))
        points = np.asarray(UsdGeom.Mesh(label).GetPointsAttr().Get())
        radii = np.linalg.norm(points[:, :2], axis=1)
        wall = _FIXTURE_CUP["radius_bottom"] + points[:, 2] / _FIXTURE_CUP["height"] * (
            _FIXTURE_CUP["radius_top"] - _FIXTURE_CUP["radius_bottom"]
        )
        self.assertTrue(np.all(radii > wall))
        self.assertTrue(np.all(radii - wall < 0.001))
        self.assertTrue(np.all(points[:, 2] > 0))
        self.assertTrue(np.all(points[:, 2] < _FIXTURE_CUP["height"]))

    def test_box_fit_and_clearance(self):
        # The fitted box must keep physical walls, an open top, and floor support.
        box_config = json.loads((_ROOT / "episode-001.json").read_text())["box"]
        profile = json.loads(self._profile.read_text())
        profile["box"] = box_config
        self._profile.write_text(json.dumps(profile))
        output = self._folder / "with-box"
        _EPISODE._build(self._profile, output)
        stage = Usd.Stage.Open(str(output / "scene.episode-001.usda"))
        box = stage.GetPrimAtPath(_BOX)
        self.assertFalse(box.HasAPI(UsdPhysics.RigidBodyAPI))
        np.testing.assert_allclose(
            box.GetAttribute("xformOp:translate").Get(), box_config["position"],
            atol=_TOLERANCE,
        )
        colliders = [prim for prim in Usd.PrimRange(box)
                     if prim.HasAPI(UsdPhysics.CollisionAPI)]
        self.assertEqual({prim.GetName() for prim in colliders},
                         {"Bottom", "Front", "Rear", "Left", "Right"})
        for collider in colliders:
            self.assertFalse(collider.HasAPI(UsdPhysics.RigidBodyAPI))
            self.assertTrue(UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get())

        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        bounds = {
            prim.GetName(): cache.ComputeWorldBound(prim).ComputeAlignedRange()
            for prim in colliders
        }
        width, depth, height = box_config["size"]
        thickness = box_config["thickness"]
        np.testing.assert_allclose(bounds["Bottom"].GetSize(),
                                   [width, depth, thickness], atol=_TOLERANCE)
        np.testing.assert_allclose(bounds["Front"].GetSize(),
                                   [width, thickness, height], atol=_TOLERANCE)
        np.testing.assert_allclose(bounds["Left"].GetSize(),
                                   [thickness, depth, height], atol=_TOLERANCE)
        floor = cache.ComputeWorldBound(
            stage.GetPrimAtPath("/World/Environment/Floor")
        ).ComputeAlignedRange()
        rail = cache.ComputeWorldBound(
            stage.GetPrimAtPath("/World/Environment/Shelf/Rail0Front")
        ).ComputeAlignedRange()
        self.assertAlmostEqual(bounds["Bottom"].GetMin()[2], floor.GetMax()[2], places=5)
        self.assertGreater(rail.GetMin()[1] - bounds["Rear"].GetMax()[1], 0)

    def test_initial_state_and_links(self):
        source = Usd.Stage.Open(str(_ROOT / "scene.usda"))
        cache = UsdGeom.XformCache()
        entries = self._report["initial_joints"]
        self.assertEqual([row["policy_state"] for row in entries.values()], _INITIAL_STATE)
        self.assertEqual([name for name, row in entries.items() if row["clipped"]],
                         ["shoulder_lift"])
        for name, row in entries.items():
            path = f"{_ROBOT}/joints/{name}"
            prim = self._stage.GetPrimAtPath(path)
            joint = UsdPhysics.RevoluteJoint(prim)
            original = UsdPhysics.RevoluteJoint(source.GetPrimAtPath(path))
            self.assertEqual(joint.GetLowerLimitAttr().Get(), original.GetLowerLimitAttr().Get())
            self.assertEqual(joint.GetUpperLimitAttr().Get(), original.GetUpperLimitAttr().Get())
            degrees = math.degrees(row["applied_rad"])
            self.assertAlmostEqual(prim.GetAttribute("state:angular:physics:position").Get(),
                                   degrees, places=4)
            self.assertAlmostEqual(
                UsdPhysics.DriveAPI(prim, "angular").GetTargetPositionAttr().Get(),
                degrees, places=4,
            )
            parent = self._stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
            child = self._stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])
            relative = np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(parent)).T) @ (
                np.asarray(cache.GetLocalToWorldTransform(child)).T
            )
            origin = joint.GetLocalRot0Attr().Get()
            rotation = Rotation.from_quat([*origin.GetImaginary(), origin.GetReal()]).as_matrix()
            expected = rotation @ Rotation.from_euler("z", row["applied_rad"]).as_matrix()
            np.testing.assert_allclose(relative[:3, :3], expected, atol=1e-6)
            np.testing.assert_allclose(relative[:3, 3], joint.GetLocalPos0Attr().Get(), atol=1e-7)

    def test_camera_and_base_unchanged(self):
        source = Usd.Stage.Open(str(_ROOT / "scene.usda"))
        original = source.GetPrimAtPath(_CAMERA)
        generated = self._stage.GetPrimAtPath(_CAMERA)
        for attribute in original.GetAttributes():
            self.assertEqual(attribute.Get(), generated.GetAttribute(attribute.GetName()).Get())
        self.assertEqual(self._before, {name: _digest(_ROOT / name) for name in _SOURCE_FILES})

    def test_rollout_timing_and_assets(self):
        self.assertEqual(UsdGeom.GetStageMetersPerUnit(self._stage), 1.0)
        self.assertEqual(UsdGeom.GetStageUpAxis(self._stage), "Z")
        self.assertEqual(self._stage.GetDefaultPrim().GetPath(), "/World")
        self.assertEqual(self._stage.GetEndTimeCode(), _CONTROL["steps"])
        self.assertEqual(self._stage.GetFramesPerSecond(), _CONTROL["fps"])
        self.assertEqual(self._stage.GetTimeCodesPerSecond(), _CONTROL["fps"])
        manifest = yaml.safe_load((self._output / "rollout.episode-001.yaml").read_text())
        self.assertEqual(manifest["control"], _CONTROL)
        self.assertEqual(manifest["scene"]["uri"], "scene.episode-001.usda")
        self.assertEqual(manifest["scene"]["camera"], _CAMERA)
        self.assertEqual(manifest["scene"]["articulation"], f"{_ROBOT}/joints/root_joint")
        calibration = self._output / manifest["calibration"]
        self.assertTrue(calibration.is_file())
        self.assertEqual(yaml.safe_load(calibration.read_text())["status"], "unverified")


if __name__ == "__main__":
    unittest.main()
