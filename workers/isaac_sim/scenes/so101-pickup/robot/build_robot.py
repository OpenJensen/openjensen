"""Convert the pinned SO101 URDF and STL files into an offline USD articulation."""

import argparse
from enum import IntEnum
import json
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

_FOLDER = Path(__file__).resolve().parent
_URDF = _FOLDER / "source" / "so101_new_calib.urdf"
_ROOT = "/SO101"
_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
_INITIAL_DEGREES = (9.36, -100.0, 96.57, 50.0, 7.96, 1.43)
_SOURCE_REVISION = "5f6d2b876a53a4872e405b991dd925556c9e38a4"
_POSITION_ITERATIONS = 16
_VELOCITY_ITERATIONS = 4
_DRIVE_STIFFNESS = 1.0
_DRIVE_DAMPING = 0.08
_CONTACT_OFFSET = 0.001
_HULL_VERTEX_LIMIT = 64
_MAX_CONVEX_HULLS = 16
_STL_HEADER_BYTES = 84
_STL_TRIANGLE_BYTES = 50
_STL_DTYPE = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
_COLORS = {"plastic": (0.86, 0.85, 0.80), "motor": (0.025, 0.029, 0.033), "gripper": (0.045, 0.05, 0.055), "orange": (1.0, 0.18, 0.003)}


class _CoverAxis(IntEnum):
    Y = 1
    Z = 2


def _values(text):
    return np.array([float(value) for value in text.split()])


def _origin(element):
    origin = element.find("origin")
    matrix = np.eye(4)
    if origin is None:
        return matrix

    roll, pitch, yaw = _values(origin.get("rpy", "0 0 0"))
    cr, cp, cy = np.cos([roll, pitch, yaw])
    sr, sp, sy = np.sin([roll, pitch, yaw])
    matrix[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                      [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                      [-sp, cp*sr, cp*cr]]
    matrix[:3, 3] = _values(origin.get("xyz", "0 0 0"))
    return matrix


def _quat(rotation):
    # USD matrices use row vectors; URDF transforms use column vectors.
    quat = Gf.Matrix3d(rotation.T.tolist()).ExtractRotation().GetQuat()
    return Gf.Quatf(quat.GetReal(), Gf.Vec3f(quat.GetImaginary()))


def _transform(xform, matrix):
    xform.AddTranslateOp().Set(Gf.Vec3d(*matrix[:3, 3]))
    xform.AddOrientOp().Set(_quat(matrix[:3, :3]))


def _schema(prim, name):
    schemas = prim.GetMetadata("apiSchemas") or Sdf.TokenListOp()
    prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(list(schemas.GetAppliedItems()) + [name]))


def _attr(prim, name, kind, value):
    prim.CreateAttribute(name, kind, custom=False).Set(value)


def _materials(stage):
    result = {}
    for name, color in _COLORS.items():
        material = UsdShade.Material.Define(stage, f"{_ROOT}/materials/{name}")
        shader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        result[name] = material
    return result


def _mesh_asset(source, output):
    data = source.read_bytes()
    count = struct.unpack_from("<I", data, _STL_HEADER_BYTES - 4)[0]
    if len(data) != _STL_HEADER_BYTES + _STL_TRIANGLE_BYTES * count:
        raise ValueError(f"Expected binary STL: {source}")

    triangles = np.frombuffer(data, _STL_DTYPE, count, _STL_HEADER_BYTES)
    vertices, indices = np.unique(triangles["vertices"].reshape(-1, 3), axis=0, return_inverse=True)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    mesh = UsdGeom.Mesh.Define(stage, "/Mesh")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(count, 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices.astype(np.int32)))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(triangles["normal"].copy()))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.uniform)
    mesh.CreateExtentAttr([Gf.Vec3f(*vertices.min(axis=0).tolist()), Gf.Vec3f(*vertices.max(axis=0).tolist())])
    stage.SetDefaultPrim(mesh.GetPrim())
    stage.GetRootLayer().Save()
    return count


def _mass(prim, element):
    inertial = element.find("inertial")
    tensor = inertial.find("inertia")
    ixx, ixy, ixz, iyy, iyz, izz = [float(tensor.get(key)) for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")]
    origin = _origin(inertial)
    inertia = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    inertia = origin[:3, :3] @ inertia @ origin[:3, :3].T
    diagonal, axes = np.linalg.eigh(inertia)
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1

    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(float(inertial.find("mass").get("value")))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*origin[:3, 3]))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*diagonal))
    mass.CreatePrincipalAxesAttr(_quat(axes))


def _visuals(stage, link, element, materials):
    for index, visual in enumerate(element.findall("visual")):
        source_name = Path(visual.find("geometry/mesh").get("filename")).stem
        mesh = UsdGeom.Mesh.Define(stage, f"{link.GetPath()}/visual_{index}_{source_name}")
        mesh.GetPrim().GetReferences().AddReference(f"./meshes/{source_name}.usdc")
        _transform(mesh, _origin(visual))
        material_name = "plastic"
        if visual.find("material").get("name") == "sts3215":
            material_name = "motor"
        elif element.get("name") in ("gripper_link", "moving_jaw_so101_v1_link"):
            material_name = "gripper"

        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(materials[material_name])
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*_COLORS[material_name])])

        # Convex decomposition preserves the open gripper and avoids dynamic triangle meshes.
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexDecomposition")
        _schema(mesh.GetPrim(), "PhysxCollisionAPI")
        _schema(mesh.GetPrim(), "PhysxConvexDecompositionCollisionAPI")
        _attr(mesh.GetPrim(), "physxCollision:contactOffset", Sdf.ValueTypeNames.Float, _CONTACT_OFFSET)
        _attr(mesh.GetPrim(), "physxCollision:restOffset", Sdf.ValueTypeNames.Float, 0.0)
        _attr(mesh.GetPrim(), "physxConvexDecompositionCollision:hullVertexLimit", Sdf.ValueTypeNames.Int, _HULL_VERTEX_LIMIT)
        _attr(mesh.GetPrim(), "physxConvexDecompositionCollision:maxConvexHulls", Sdf.ValueTypeNames.Int, _MAX_CONVEX_HULLS)


def _link_poses(robot, degrees):
    poses = {"base_link": np.eye(4)}
    remaining = list(robot.findall("joint"))
    while remaining:
        before = len(remaining)
        for joint in remaining[:]:
            parent = joint.find("parent").get("link")
            if parent not in poses:
                continue

            origin = _origin(joint)
            rotation = np.eye(4)
            if joint.get("type") == "revolute":
                angle = math.radians(degrees[joint.get("name")])
                cosine, sine = math.cos(angle), math.sin(angle)
                rotation[:3, :3] = [[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]]

            child = joint.find("child").get("link")
            poses[child] = poses[parent] @ origin @ rotation
            remaining.remove(joint)
        if len(remaining) == before:
            raise ValueError("URDF contains a disconnected or cyclic joint tree")
    return poses


def _joint(stage, element, degrees):
    name = element.get("name")
    joint = UsdPhysics.RevoluteJoint.Define(stage, f"{_ROOT}/joints/{name}")
    joint.CreateBody0Rel().SetTargets([f"{_ROOT}/{element.find('parent').get('link')}"])
    joint.CreateBody1Rel().SetTargets([f"{_ROOT}/{element.find('child').get('link')}"])
    origin = _origin(element)
    joint.CreateLocalPos0Attr(Gf.Vec3f(*origin[:3, 3]))
    joint.CreateLocalRot0Attr(_quat(origin[:3, :3]))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
    joint.CreateAxisAttr(UsdPhysics.Tokens.z)
    joint.CreateCollisionEnabledAttr(False)
    limit = element.find("limit")
    lower, upper = [math.degrees(float(limit.get(key))) for key in ("lower", "upper")]
    if not lower <= degrees[name] <= upper:
        raise ValueError(f"{name} pose {degrees[name]} outside [{lower}, {upper}] degrees")

    joint.CreateLowerLimitAttr(lower)
    joint.CreateUpperLimitAttr(upper)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr(UsdPhysics.Tokens.force)
    drive.CreateTargetPositionAttr(degrees[name])
    drive.CreateTargetVelocityAttr(0.0)
    drive.CreateStiffnessAttr(_DRIVE_STIFFNESS)
    drive.CreateDampingAttr(_DRIVE_DAMPING)
    drive.CreateMaxForceAttr(float(limit.get("effort")))
    _schema(joint.GetPrim(), "PhysicsJointStateAPI:angular")
    _attr(joint.GetPrim(), "state:angular:physics:position", Sdf.ValueTypeNames.Float, degrees[name])
    _attr(joint.GetPrim(), "state:angular:physics:velocity", Sdf.ValueTypeNames.Float, 0.0)
    _schema(joint.GetPrim(), "PhysxJointAPI")
    _attr(joint.GetPrim(), "physxJoint:maxJointVelocity", Sdf.ValueTypeNames.Float, math.degrees(float(limit.get("velocity"))))


def _finger_cover(stage, name, axis, center, sections, material):
    # Tapered exterior covers approximate the orange hardware visible in the video.
    points = []
    cross_axis = 3 - axis
    for along, low, high, width in sorted(sections):
        for x, across in ((low, -width/2), (high, -width/2), (high, width/2), (low, width/2)):
            point = [x, 0.0, 0.0]
            point[axis] = along
            point[cross_axis] = center + across
            points.append(point)

    last = len(points) - 4
    faces = [[3, 2, 1, 0], [last + index for index in range(4)]]
    for section in range(len(sections) - 1):
        start = section * 4
        for corner in range(4):
            following = (corner + 1) % 4
            faces.append([start + corner, start + following, start + 4 + following, start + 4 + corner])
    if axis == _CoverAxis.Y:
        faces = [list(reversed(face)) for face in faces]

    mesh = UsdGeom.Mesh.Define(stage, f"{_ROOT}/{name}/orange_cover")
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
    mesh.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    array = np.asarray(points)
    mesh.CreateExtentAttr([Gf.Vec3f(*array.min(0)), Gf.Vec3f(*array.max(0))])
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*_COLORS["orange"])])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    _schema(mesh.GetPrim(), "PhysxCollisionAPI")
    _attr(mesh.GetPrim(), "physxCollision:contactOffset", Sdf.ValueTypeNames.Float, _CONTACT_OFFSET)
    _attr(mesh.GetPrim(), "physxCollision:restOffset", Sdf.ValueTypeNames.Float, 0.0)


def _build(degrees):
    robot = ET.parse(_URDF).getroot()
    mesh_dir = _FOLDER / "meshes"
    mesh_dir.mkdir(exist_ok=True)
    triangles = 0
    sources = sorted({mesh.get("filename") for mesh in robot.iter("mesh")})
    for filename in sources:
        source = _URDF.parent / filename
        triangles += _mesh_asset(source, mesh_dir / f"{source.stem}.usdc")

    stage = Usd.Stage.CreateNew(str(_FOLDER / "so101.usda"))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, _ROOT)
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetAssetInfoByKey("name", "SO101 follower (TheRobotStudio, new calibration)")
    root.GetPrim().SetCustomDataByKey("sourceRevision", _SOURCE_REVISION)
    materials = _materials(stage)
    poses = _link_poses(robot, degrees)

    for element in robot.findall("link"):
        name = element.get("name")
        if not element.findall("visual"):
            continue

        link = UsdGeom.Xform.Define(stage, f"{_ROOT}/{name}")
        _transform(link, poses[name])
        UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
        _mass(link.GetPrim(), element)
        _visuals(stage, link, element, materials)

    _finger_cover(stage, "gripper_link", _CoverAxis.Z, 0.0,
                  [(-0.098, -0.019, -0.013, 0.008), (-0.09, -0.022, -0.016, 0.015),
                   (-0.04, -0.040, -0.034, 0.032)], materials["orange"])
    _finger_cover(stage, "moving_jaw_so101_v1_link", _CoverAxis.Y, 0.0189,
                  [(-0.078, 0.004, 0.009, 0.012), (-0.036, 0.010, 0.016, 0.030)], materials["orange"])

    # The massless tool frame belongs to the gripper body, not a separate rigid body.
    frame = UsdGeom.Xform.Define(stage, f"{_ROOT}/gripper_link/gripper_frame_link")
    frame_joint = next(joint for joint in robot.findall("joint") if joint.get("name") == "gripper_frame_joint")
    _transform(frame, _origin(frame_joint))
    for element in robot.findall("joint"):
        if element.get("type") == "revolute":
            _joint(stage, element, degrees)

    fixed = UsdPhysics.FixedJoint.Define(stage, f"{_ROOT}/joints/root_joint")
    fixed.CreateBody0Rel().SetTargets([f"{_ROOT}/base_link"])
    UsdPhysics.ArticulationRootAPI.Apply(fixed.GetPrim())
    _schema(fixed.GetPrim(), "PhysxArticulationAPI")
    _attr(fixed.GetPrim(), "physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool, False)
    _attr(fixed.GetPrim(), "physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int, _POSITION_ITERATIONS)
    _attr(fixed.GetPrim(), "physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int, _VELOCITY_ITERATIONS)
    stage.GetRootLayer().Save()
    report = {"joint_degrees": degrees, "source_triangles": triangles,
              "link_positions_m": {name: pose[:3, 3].tolist() for name, pose in poses.items()}}
    (_FOLDER / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-degrees", type=float, nargs=len(_JOINTS), default=_INITIAL_DEGREES,
                        metavar="DEG", help="shoulder_pan shoulder_lift elbow_flex wrist_flex wrist_roll gripper")
    args = parser.parse_args()
    _build(dict(zip(_JOINTS, args.joint_degrees)))


if __name__ == "__main__":
    _main()
