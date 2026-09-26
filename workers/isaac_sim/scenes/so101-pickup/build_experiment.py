"""Build an explicitly unverified scene overlay for bounded ACT motion tests."""

import argparse
import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

_ROOT = Path(__file__).resolve().parent
_ROBOT = "/World/Robot"
_CAMERA = "/World/Cameras/Front"
_SCENE = "scene.experimental.usda"
_MAP = "calibration.experimental.yaml"
_REPORT = "experiment.json"
_FIT = "evidence/calibration/report.json"
_JOINTS = (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"
)
_CAMERA_AXES = np.diag([1.0, -1.0, -1.0])
_STAGE_METADATA = (
    "defaultPrim", "metersPerUnit", "upAxis", "framesPerSecond", "timeCodesPerSecond",
    "startTimeCode", "endTimeCode",
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _origin(joint):
    matrix = np.eye(4)
    origin = joint.find("origin")
    if origin is None:
        return matrix

    rpy = [float(value) for value in origin.get("rpy", "0 0 0").split()]
    matrix[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    matrix[:3, 3] = [float(value) for value in origin.get("xyz", "0 0 0").split()]
    return matrix


def _poses(robot, angles):
    poses = {"base_link": np.eye(4)}
    remaining = list(robot.findall("joint"))
    while remaining:
        ready = [joint for joint in remaining if joint.find("parent").get("link") in poses]
        if not ready:
            raise ValueError("URDF has a disconnected or cyclic joint tree")

        for joint in ready:
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")
            rotation = np.eye(4)
            if joint.get("type") != "fixed":
                axis = np.array([float(value) for value in joint.find("axis").get("xyz").split()])
                rotation[:3, :3] = Rotation.from_rotvec(
                    axis / np.linalg.norm(axis) * angles[joint.get("name")]
                ).as_matrix()

            poses[child] = poses[parent] @ _origin(joint) @ rotation
            remaining.remove(joint)
    return poses


def _initial(robot, mapping, state):
    angles, entries = {}, {}
    for name, value in zip(_JOINTS, state, strict=True):
        curve = mapping["joints"][name]
        policy_low, policy_high = curve["policy"]
        sim_low, sim_high = curve["sim_rad"]
        requested = sim_low + (value - policy_low) * (sim_high - sim_low) / (
            policy_high - policy_low
        )
        limit = robot.find(f"joint[@name='{name}']/limit")
        lower, upper = (float(limit.get(key)) for key in ("lower", "upper"))
        applied = float(np.clip(requested, lower, upper))
        angles[name] = applied
        entries[name] = {
            "policy_state": value,
            "requested_rad": requested,
            "applied_rad": applied,
            "applied_degrees": math.degrees(applied),
            "limit_rad": [lower, upper],
            "clipped": applied != requested,
        }
    return angles, entries


def _robot_pose(stage, robot, angles):
    # Match rigid-body poses to drive state so physics starts without a pose jump.
    poses = _poses(robot, angles)
    for link in robot.findall("link"):
        if not link.findall("visual"):
            continue

        name = link.get("name")
        matrix = poses[name]
        prim = stage.GetPrimAtPath(f"{_ROBOT}/{name}")
        prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*matrix[:3, 3]))
        quat = Gf.Matrix3d(matrix[:3, :3].T.tolist()).ExtractRotation().GetQuat()
        prim.GetAttribute("xformOp:orient").Set(
            Gf.Quatf(quat.GetReal(), Gf.Vec3f(quat.GetImaginary()))
        )

    for name, value in angles.items():
        prim = stage.GetPrimAtPath(f"{_ROBOT}/joints/{name}")
        degrees = math.degrees(value)
        UsdPhysics.DriveAPI(prim, "angular").GetTargetPositionAttr().Set(degrees)
        prim.GetAttribute("state:angular:physics:position").Set(degrees)


def _camera_pose(stage, fit, config):
    parameters = dict(zip(fit["parameter_names"], fit["parameters"], strict=True))
    rotation = Rotation.from_rotvec([
        parameters[f"camera_r{axis}"] for axis in "xyz"
    ]).as_matrix()
    translation = np.array([parameters[f"camera_t{axis}"] for axis in "xyz"])

    # Invert robot-to-camera, then convert computer-vision camera axes to USD.
    robot_to_world = np.asarray(
        UsdGeom.Xformable(stage.GetPrimAtPath(_ROBOT)).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
    ).T
    camera_to_robot = np.eye(4)
    camera_to_robot[:3, :3] = rotation.T @ _CAMERA_AXES
    camera_to_robot[:3, 3] = -rotation.T @ translation
    camera_to_world = robot_to_world @ camera_to_robot
    camera = UsdGeom.Camera(stage.GetPrimAtPath(_CAMERA))
    camera.GetPrim().GetAttribute("xformOp:transform").Set(
        Gf.Matrix4d(camera_to_world.T.tolist())
    )
    width = config["confirmed_metadata"]["camera"]["width"]
    focal = parameters["focal_px"] * camera.GetHorizontalApertureAttr().Get() / width
    camera.GetFocalLengthAttr().Set(focal)
    return {
        "robot_to_camera_rotation": rotation.tolist(),
        "robot_to_camera_translation": translation.tolist(),
        "world_position": camera_to_world[:3, 3].tolist(),
        "focal_pixels": parameters["focal_px"],
        "focal_mm": focal,
        "source_image_width": width,
    }


def _build(output):
    output.mkdir(parents=True, exist_ok=True)
    fit = json.loads((_ROOT / _FIT).read_text())
    dataset = json.loads((_ROOT / "evidence/dataset.json").read_text())
    mapping = yaml.safe_load((_ROOT / "calibration.candidate.yaml").read_text())
    robot_path = _ROOT / "robot/source/so101_new_calib.urdf"
    robot = ET.parse(robot_path).getroot()
    state = dataset["first_frame_raw"]["observation.state"]
    angles, initial = _initial(robot, mapping, state)

    stage = Usd.Stage.CreateNew(str(output / _SCENE))
    stage.GetRootLayer().subLayerPaths = [os.path.relpath(_ROOT / "scene.usda", output)]
    # Stage metadata does not compose from sublayers; preserve source units and timing.
    source = Usd.Stage.Open(str(_ROOT / "scene.usda"))
    metadata = {key: source.GetMetadata(key) for key in _STAGE_METADATA}
    for key, value in metadata.items():
        stage.SetMetadata(key, value)

    stage.GetRootLayer().customLayerData = {
        "calibration": "unverified",
        "purpose": "bounded ACT motion experiment; no task-success claim",
        "source": _FIT,
    }
    _robot_pose(stage, robot, angles)
    camera = _camera_pose(stage, fit, dataset)
    stage.GetRootLayer().Save()

    # Preserve observed mapping support; experimental runtime controls extrapolation.
    mapping["status"] = "unverified"
    mapping["source"] = f"{_FIT}; experimental only; see {_REPORT}"
    (output / _MAP).write_text(
        "# Unverified fit. Only use with explicit experimental rollout mode.\n"
        + yaml.safe_dump(mapping, sort_keys=False)
    )
    report = {
        "status": "experimental_unverified",
        "fit_source": _FIT,
        "fit_sha256": _sha256(_ROOT / _FIT),
        "urdf_sha256": _sha256(robot_path),
        "source_scene_sha256": _sha256(_ROOT / "scene.usda"),
        "stage_metadata": metadata,
        "initial_state_source": "evidence/dataset.json:first_frame_raw.observation.state",
        "initial_joints": initial,
        "camera": camera,
        "urdf_limits_changed": False,
        "mapping_support_extended": False,
        "limitations": fit["acceptance"]["failures"],
    }
    (output / _REPORT).write_text(json.dumps(report, indent=2) + "\n")
    return report


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_ROOT)
    args = parser.parse_args()
    report = _build(args.output.resolve())
    print(json.dumps({"status": report["status"], "scene": str(args.output / _SCENE)}))


if __name__ == "__main__":
    _main()
