"""Check scene structure with OpenUSD; this does not run PhysX or Isaac Sim."""

import argparse
import math
from pathlib import Path
import sys

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

_SCENE = Path(__file__).resolve().with_name("scene.usda")
_ROBOT_PATH = Sdf.Path("/World/Robot")
_CUP_PATH = Sdf.Path("/World/Props/Cup")
_BOX_PATH = Sdf.Path("/World/Props/Box")
_LINK_COUNT = 7
_JOINT_NAMES = frozenset({
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
})
_CAMERA_PATHS = ("/World/Cameras/Front", "/World/Cameras/Overview")
_REQUIRED_PATHS = (
    "/World", str(_ROBOT_PATH), str(_CUP_PATH), str(_BOX_PATH),
    "/World/Environment/Shelf", "/World/Environment/Floor",
    "/World/Environment/Backdrop", "/World/Physics", *_CAMERA_PATHS,
)
_ASPECT_RATIO = 1920 / 1080
_EARTH_GRAVITY = 9.81
_GRAVITY_TOLERANCE = 0.02
_COMPARISON_TOLERANCE = 1e-5


def _check_stage(stage, scene_path):
    errors = [str(error) for error in stage.GetCompositionErrors()]
    _, _, unresolved = UsdUtils.ComputeAllDependencies(str(scene_path))
    errors.extend(f"Unresolved dependency: {path}" for path in unresolved)

    for path in _REQUIRED_PATHS:
        if not stage.GetPrimAtPath(path):
            errors.append(f"Missing prim: {path}")

    if stage.GetDefaultPrim() != stage.GetPrimAtPath("/World"):
        errors.append("The default prim must be /World.")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        errors.append("The stage must use Z up.")
    if not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0):
        errors.append("The stage must use meters.")
    return errors


def _check_physics(stage):
    errors = []
    physics = UsdPhysics.Scene(stage.GetPrimAtPath("/World/Physics"))
    if not physics:
        return ["/World/Physics must be a PhysicsScene."]

    direction = physics.GetGravityDirectionAttr().Get()
    magnitude = physics.GetGravityMagnitudeAttr().Get()
    if direction is None or not all(math.isfinite(v) for v in direction):
        errors.append("Gravity direction is missing or nonfinite.")
    elif any(abs(v - expected) > _COMPARISON_TOLERANCE
             for v, expected in zip(direction, (0.0, 0.0, -1.0))):
        errors.append("Gravity must point down the Z axis.")
    if magnitude is None or not math.isclose(
            magnitude, _EARTH_GRAVITY, abs_tol=_GRAVITY_TOLERANCE):
        errors.append("Gravity magnitude must be approximately 9.81 m/s².")
    return errors


def _check_mass(prim):
    mass_api = UsdPhysics.MassAPI(prim)
    if not mass_api:
        return [f"Missing mass API: {prim.GetPath()}"]

    errors = []
    mass = mass_api.GetMassAttr().Get()
    inertia = mass_api.GetDiagonalInertiaAttr().Get()
    if mass is None or not math.isfinite(mass) or mass <= 0:
        errors.append(f"Mass must be finite and positive: {prim.GetPath()}")
    if inertia is None or any(not math.isfinite(v) or v <= 0 for v in inertia):
        errors.append(f"Inertia must be finite and positive: {prim.GetPath()}")
    return errors


def _check_bodies(stage):
    errors = []
    bodies = [prim for prim in stage.Traverse()
              if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    robot_bodies = [prim for prim in bodies
                    if prim.GetPath().HasPrefix(_ROBOT_PATH)]
    if len(robot_bodies) != _LINK_COUNT:
        errors.append(f"Expected {_LINK_COUNT} robot links, got {len(robot_bodies)}.")

    prop_bodies = {prim.GetPath() for prim in bodies
                   if not prim.GetPath().HasPrefix(_ROBOT_PATH)}
    if prop_bodies != {_CUP_PATH}:
        errors.append(f"Only the cup should be dynamic outside the robot: {prop_bodies}")

    for prim in bodies:
        errors.extend(_check_mass(prim))
        body = UsdPhysics.RigidBodyAPI(prim)
        if not body.GetRigidBodyEnabledAttr().Get():
            errors.append(f"Rigid body is disabled: {prim.GetPath()}")
        if body.GetKinematicEnabledAttr().Get():
            errors.append(f"Rigid body must be dynamic: {prim.GetPath()}")

        # Every simulated body needs a shape for contact, including the cup walls.
        colliders = [child for child in Usd.PrimRange(prim)
                     if child.HasAPI(UsdPhysics.CollisionAPI)
                     and UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr().Get()]
        if not colliders:
            errors.append(f"Rigid body has no enabled collider: {prim.GetPath()}")

    box = stage.GetPrimAtPath(_BOX_PATH)
    if box and not any(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in Usd.PrimRange(box)):
        errors.append("The box needs static colliders.")
    return errors


def _check_drive(prim):
    drive = UsdPhysics.DriveAPI(prim, "angular")
    if not drive:
        return [f"Missing angular drive: {prim.GetPath()}"]

    errors = []
    values = {
        "stiffness": drive.GetStiffnessAttr().Get(),
        "damping": drive.GetDampingAttr().Get(),
        "maxForce": drive.GetMaxForceAttr().Get(),
    }
    for name, value in values.items():
        if value is None or not math.isfinite(value) or value <= 0:
            errors.append(f"Drive {name} must be finite and positive: {prim.GetPath()}")

    target = drive.GetTargetPositionAttr().Get()
    joint = UsdPhysics.RevoluteJoint(prim)
    lower = joint.GetLowerLimitAttr().Get()
    upper = joint.GetUpperLimitAttr().Get()
    if lower is None or upper is None or not all(map(math.isfinite, (lower, upper))):
        errors.append(f"Joint limits must be finite: {prim.GetPath()}")
    elif lower >= upper:
        errors.append(f"Joint limits are reversed or empty: {prim.GetPath()}")
    elif target is None or not math.isfinite(target) or not lower <= target <= upper:
        errors.append(f"Drive target is outside joint limits: {prim.GetPath()}")

    # Articulations initialize from joint state, independently of drive targets.
    position = prim.GetAttribute("state:angular:physics:position").Get()
    velocity = prim.GetAttribute("state:angular:physics:velocity").Get()
    if position is None or not math.isfinite(position) or position != target:
        errors.append(f"Initial joint state must match its drive target: {prim.GetPath()}")
    if velocity is None or not math.isfinite(velocity) or velocity != 0:
        errors.append(f"Initial joint velocity must be zero: {prim.GetPath()}")
    return errors


def _check_joints(stage):
    robot = stage.GetPrimAtPath(_ROBOT_PATH)
    if not robot:
        return []

    errors = []
    joints = [prim for prim in Usd.PrimRange(robot) if prim.IsA(UsdPhysics.RevoluteJoint)]
    names = [prim.GetName() for prim in joints]
    if len(names) != len(_JOINT_NAMES) or set(names) != _JOINT_NAMES:
        errors.append(f"Unexpected revolute joints: {names}")

    articulation_roots = [prim for prim in Usd.PrimRange(robot)
                          if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(articulation_roots) != 1:
        errors.append(f"Expected one articulation root, got {len(articulation_roots)}.")

    fixed_roots = []
    for prim in Usd.PrimRange(robot):
        if not prim.IsA(UsdPhysics.Joint):
            continue

        joint = UsdPhysics.Joint(prim)
        parents = joint.GetBody0Rel().GetTargets()
        children = joint.GetBody1Rel().GetTargets()
        if prim.IsA(UsdPhysics.FixedJoint) and sorted((len(parents), len(children))) == [0, 1]:
            fixed_roots.append(prim)
        elif len(parents) != 1 or len(children) != 1:
            errors.append(f"Joint needs one parent and child body: {prim.GetPath()}")

        for path in parents + children:
            body = stage.GetPrimAtPath(path)
            if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI):
                errors.append(f"Joint references an invalid rigid body: {path}")

        if prim.IsA(UsdPhysics.RevoluteJoint):
            errors.extend(_check_drive(prim))

    if len(fixed_roots) != 1:
        errors.append(f"Expected one joint fixing the robot to world, got {len(fixed_roots)}.")
    return errors


def _check_geometry(stage):
    errors = []
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [
        UsdGeom.Tokens.default_, UsdGeom.Tokens.render,
        UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide,
    ])
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Boundable):
            continue

        bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if bounds.IsEmpty() or not all(math.isfinite(v) for v in (*bounds.GetMin(), *bounds.GetMax())):
            errors.append(f"Geometry has empty or nonfinite bounds: {prim.GetPath()}")

        if not prim.IsA(UsdGeom.Mesh):
            continue

        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if points is None or not len(points) or any(not math.isfinite(v) for p in points for v in p):
            errors.append(f"Mesh has missing or nonfinite points: {prim.GetPath()}")
            continue
        if counts is None or indices is None or sum(counts) != len(indices):
            errors.append(f"Mesh face counts do not match indices: {prim.GetPath()}")
            continue
        if any(count < 3 for count in counts) or any(i < 0 or i >= len(points) for i in indices):
            errors.append(f"Mesh has invalid face topology: {prim.GetPath()}")
    return errors


def _check_cameras(stage):
    errors = []
    for path in _CAMERA_PATHS:
        camera = UsdGeom.Camera(stage.GetPrimAtPath(path))
        if not camera:
            errors.append(f"Missing camera schema: {path}")
            continue

        horizontal = camera.GetHorizontalApertureAttr().Get()
        vertical = camera.GetVerticalApertureAttr().Get()
        if vertical <= 0 or not math.isclose(horizontal / vertical, _ASPECT_RATIO,
                                            rel_tol=_COMPARISON_TOLERANCE):
            errors.append(f"Camera aperture must have 1920:1080 aspect: {path}")
        near, far = camera.GetClippingRangeAttr().Get()
        if not all(map(math.isfinite, (near, far))) or not 0 < near < far:
            errors.append(f"Invalid camera clipping range: {path}")
    return errors


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", nargs="?", type=Path, default=_SCENE)
    scene_path = parser.parse_args().scene.resolve()
    if not scene_path.is_file():
        parser.error(f"Scene not found: {scene_path}")

    try:
        stage = Usd.Stage.Open(str(scene_path))
        errors = _check_stage(stage, scene_path)
        for check in (_check_physics, _check_bodies, _check_joints,
                      _check_geometry, _check_cameras):
            errors.extend(check(stage))
    except Exception as error:
        print(f"Structural validation failed: {error}", file=sys.stderr)
        return 1

    if errors:
        print("Structural validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Structural validation passed: resolved assets, 7 robot links, 6 driven joints,")
    print("fixed base, dynamic cup, static box, finite geometry, physics, and 16:9 cameras.")
    print("Isaac Sim / PhysX runtime behavior has not been tested by this command.")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
