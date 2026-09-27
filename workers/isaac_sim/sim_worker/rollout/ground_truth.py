"""Read evaluation geometry from USD and poses from the live physics view."""

import itertools
import math

from sim_worker.rollout.contracts import ObjectState
from sim_worker.rollout.evaluation import EvaluationSpec, world_bounds


def object_geometry(stage, path):
    # No USD/Isaac dependency is imported by the application or pure scorer.
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z or not math.isclose(
        UsdGeom.GetStageMetersPerUnit(stage), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("Evaluation currently requires a Z-up, meter-authored stage")
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError("Evaluation object must be an existing USD rigid body")
    body = UsdPhysics.RigidBodyAPI(prim)
    if not body.GetRigidBodyEnabledAttr().Get() or body.GetKinematicEnabledAttr().Get():
        raise ValueError("Evaluation object must be a dynamic rigid body")
    ancestor = prim.GetParent()
    while ancestor and not ancestor.IsPseudoRoot():
        if any(attr.GetNumTimeSamples() for attr in ancestor.GetAttributes()):
            raise ValueError("Evaluation requires static authored object ancestry")
        ancestor = ancestor.GetParent()
    # Physics transforms do not contain scale. Reject scaled/sheared ancestry
    # instead of comparing local geometry in a different coordinate system.
    cache = UsdGeom.XformCache()
    transform = cache.GetLocalToWorldTransform(prim)
    rows = [[float(transform[i][j]) for j in range(3)] for i in range(3)]
    if (
        any(
            not math.isclose(
                sum(rows[i][k] * rows[j][k] for k in range(3)),
                float(i == j),
                rel_tol=0.0,
                abs_tol=1e-7,
            )
            for i in range(3)
            for j in range(3)
        )
        or float(transform.GetDeterminant()) < 0
    ):
        raise ValueError("Evaluation does not support scaled, sheared or reflected rigid bodies")
    points = []
    inverse = transform.GetInverse()
    for child in Usd.PrimRange(prim):
        if child.IsInstance() or child.IsInstanceProxy() or child.IsInPrototype():
            raise ValueError("Evaluation does not support instanced object geometry")
        if any(attr.GetNumTimeSamples() for attr in child.GetAttributes()):
            raise ValueError("Evaluation requires static authored object geometry")
        if child != prim and child.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError("Evaluation does not support nested rigid bodies")
        if child.IsA(UsdGeom.Boundable):
            if child.GetTypeName() not in {"Mesh", "Cube", "Sphere", "Cylinder", "Cone", "Capsule"}:
                raise ValueError("Evaluation object contains unsupported geometry")
            # Recompute leaf geometry; stale authored extents must not shrink the
            # object. Transform into rigid-body space before taking the union.
            extent = UsdGeom.Boundable.ComputeExtentFromPlugins(
                UsdGeom.Boundable(child), Usd.TimeCode.Default()
            )
            if extent is None or len(extent) != 2:
                raise ValueError("Evaluation cannot bound object geometry")
            relative = cache.GetLocalToWorldTransform(child) * inverse
            for corner in itertools.product(*zip(*extent)):
                points.append(tuple(relative.Transform(Gf.Vec3d(*corner))))
    if not points or any(not math.isfinite(v) for point in points for v in point):
        raise ValueError("Evaluation object has no finite geometry")
    minimum = tuple(min(point[i] for point in points) for i in range(3))
    maximum = tuple(max(point[i] for point in points) for i in range(3))
    # Reuse the pure scorer's finite/positive validation before starting motion.
    world_bounds(ObjectState(path, (0, 0, 0), (0, 0, 0, 1), (0, 0, 0), (0, 0, 0), minimum, maximum))
    return minimum, maximum


class ObjectProbe:
    def __init__(self, stage, physics_view, spec: EvaluationSpec):
        self._path = spec.object_prim
        self._bounds = object_geometry(stage, self._path)
        # This driver owns one simulation view; bind readback to world space.
        physics_view.set_subspace_roots("/")
        self._body = physics_view.create_rigid_body_view(self._path)
        if self._body.count != 1:
            raise ValueError("Evaluation requires exactly one live rigid body")

    def read(self):
        from sim_worker.rollout.isaac import _array

        pose = _array(self._body.get_transforms())
        velocity = _array(self._body.get_velocities())
        if pose.shape != (1, 7) or velocity.shape != (1, 6):
            raise RuntimeError("Invalid live object pose or velocity shape")
        if not all(math.isfinite(float(v)) for v in [*pose[0], *velocity[0]]):
            raise RuntimeError("Nonfinite live object pose or velocity")
        return ObjectState(
            self._path,
            tuple(float(v) for v in pose[0, :3]),
            tuple(float(v) for v in pose[0, 3:]),
            tuple(float(v) for v in velocity[0, :3]),
            tuple(float(v) for v in velocity[0, 3:]),
            *self._bounds,
        )
