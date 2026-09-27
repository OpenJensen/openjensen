import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sim_worker.rollout.ground_truth import ObjectProbe, object_geometry


class ObjectProbeTests(unittest.TestCase):
    def test_reads_live_physics_not_authored_pose(self):
        body = SimpleNamespace(
            count=1,
            get_transforms=lambda: np.array([[1, 2, 3, 0, 0, 0, 1]]),
            get_velocities=lambda: np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]),
        )
        paths = []

        def create(path):
            paths.append(path)
            return body

        with patch(
            "sim_worker.rollout.ground_truth.object_geometry", return_value=((-1, -1, 0), (1, 1, 2))
        ):
            probe = ObjectProbe(
                None,
                SimpleNamespace(create_rigid_body_view=create, set_subspace_roots=paths.append),
                SimpleNamespace(object_prim="/World/Cup"),
            )
        observed = probe.read()
        self.assertEqual(paths, ["/", "/World/Cup"])
        self.assertEqual(observed.position_m, (1.0, 2.0, 3.0))
        self.assertEqual(observed.orientation_xyzw, (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(observed.angular_velocity_rad_s, (0.4, 0.5, 0.6))
        body.get_transforms = lambda: np.array([[4, 5, 6, 0, 0, 0, 1]])
        self.assertEqual(probe.read().position_m, (4.0, 5.0, 6.0))
        body.get_velocities = lambda: np.full((1, 6), np.nan)
        with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
            probe.read()


class USDGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from pxr import Usd, UsdGeom, UsdPhysics
        except ImportError as error:
            raise unittest.SkipTest(
                "Install pinned scene-build USD dependency for geometry tests"
            ) from error
        cls.Usd, cls.UsdGeom, cls.UsdPhysics = Usd, UsdGeom, UsdPhysics

    def test_actual_episode_cup_geometry_from_usd(self):
        root = Path(__file__).resolve().parents[1] / "scenes/so101-pickup"
        stage = self.Usd.Stage.Open(str(root / "scene.episode-001.usda"))
        minimum, maximum = object_geometry(stage, "/World/Props/Cup")
        self.assertAlmostEqual(minimum[2], 0, places=6)
        self.assertAlmostEqual(maximum[2], 0.1021984, places=6)
        # All colliders and the paper label are enclosed, not just the visual center.
        self.assertLessEqual(minimum[0], -0.0415697 + 1e-8)
        self.assertGreaterEqual(maximum[0], 0.0415697 - 1e-8)

    def _scene(self):
        stage = self.Usd.Stage.CreateInMemory()
        self.UsdGeom.SetStageUpAxis(stage, self.UsdGeom.Tokens.z)
        self.UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        prim = self.UsdGeom.Xform.Define(stage, "/World/Cup")
        self.UsdPhysics.RigidBodyAPI.Apply(prim.GetPrim())
        self.UsdGeom.Cube.Define(stage, "/World/Cup/Shape").CreateSizeAttr(2.0)
        return stage, prim

    def test_untransformed_bounds_exclude_object_pose(self):
        stage, prim = self._scene()
        prim.AddTranslateOp().Set((4, 5, 6))
        prim.AddRotateZOp().Set(45.0)
        for actual, expected in zip(
            object_geometry(stage, "/World/Cup"), ((-1, -1, -1), (1, 1, 1))
        ):
            for a, e in zip(actual, expected):
                self.assertAlmostEqual(a, e)

    def test_stale_authored_extents_cannot_shrink_geometry(self):
        stage, _ = self._scene()
        shape = self.UsdGeom.Cube(stage.GetPrimAtPath("/World/Cup/Shape"))
        shape.CreateExtentAttr([(-0.1, -0.1, -0.1), (0.1, 0.1, 0.1)])
        self.assertEqual(
            object_geometry(stage, "/World/Cup"), ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
        )

    def test_instanced_child_cannot_be_silently_omitted(self):
        stage, _ = self._scene()
        self.UsdGeom.Xform.Define(stage, "/Library/Large")
        self.UsdGeom.Cube.Define(stage, "/Library/Large/Shape").CreateSizeAttr(4.0)
        child = self.UsdGeom.Xform.Define(stage, "/World/Cup/Instance").GetPrim()
        child.GetReferences().AddInternalReference("/Library/Large")
        child.SetInstanceable(True)
        self.assertTrue(child.IsInstance())
        with self.assertRaisesRegex(ValueError, "instanced"):
            object_geometry(stage, "/World/Cup")

    def test_animated_ancestry_rejected(self):
        stage, _ = self._scene()
        world = self.UsdGeom.Xform.Define(stage, "/World")
        operation = world.AddScaleOp()
        operation.Set((1, 1, 1))
        operation.Set((2, 2, 2), 1.0)
        with self.assertRaisesRegex(ValueError, "ancestry"):
            object_geometry(stage, "/World/Cup")

    def test_scaled_or_animated_geometry_rejected(self):
        stage, prim = self._scene()
        prim.AddScaleOp().Set((1, 1, 2))
        with self.assertRaisesRegex(ValueError, "scaled"):
            object_geometry(stage, "/World/Cup")
        stage, _ = self._scene()
        shape = self.UsdGeom.Cube(stage.GetPrimAtPath("/World/Cup/Shape"))
        shape.GetSizeAttr().Set(3.0, 1.0)
        with self.assertRaisesRegex(ValueError, "static"):
            object_geometry(stage, "/World/Cup")

    def test_invalid_stage_and_missing_or_kinematic_object_rejected(self):
        stage, prim = self._scene()
        with self.assertRaisesRegex(ValueError, "existing"):
            object_geometry(stage, "/World/Missing")
        self.UsdGeom.SetStageMetersPerUnit(stage, 0.01)
        with self.assertRaisesRegex(ValueError, "meter"):
            object_geometry(stage, "/World/Cup")
        self.UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        self.UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
        with self.assertRaisesRegex(ValueError, "dynamic"):
            object_geometry(stage, "/World/Cup")
