"""Fit dataset joint coordinates to rigid landmarks in an SO101 URDF."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

_JOINTS = (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"
)
_PARAMETERS = (
    "camera_rx", "camera_ry", "camera_rz", "camera_tx", "camera_ty", "camera_tz",
    "focal_px", "shoulder_lift_offset", "elbow_flex_offset", "wrist_flex_offset",
    "wrist_roll_offset", "gripper_offset", "gripper_slope",
)
_CAMERA_ROTATION = slice(0, 3)
_CAMERA_TRANSLATION = slice(3, 6)
_FOCAL = 6
_OFFSETS = slice(7, 12)
_GRIPPER_SLOPE = 12
_ARM_JOINTS = 5
_MIN_DEPTH = 1e-6
_MAX_EVALUATIONS = 2000


@dataclass(frozen=True)
class _Joint:
    name: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    kind: str


def _vector(values, size, name):
    result = np.asarray(values, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} requires {size} finite values")
    return result


def _origin(element):
    result = np.eye(4)
    origin = element.find("origin")
    if origin is None:
        return result

    # URDF fixed-axis roll/pitch/yaw uses Rz(yaw) Ry(pitch) Rx(roll).
    xyz = _vector(origin.get("xyz", "0 0 0").split(), 3, "origin.xyz")
    rpy = _vector(origin.get("rpy", "0 0 0").split(), 3, "origin.rpy")
    result[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    result[:3, 3] = xyz
    return result


def _tree(path):
    robot = ET.parse(path).getroot()
    links = {element.get("name") for element in robot.findall("link")}
    joints = []
    children = set()
    names = set()
    for element in robot.findall("joint"):
        name, kind = element.get("name"), element.get("type")
        parent = element.find("parent").get("link")
        child = element.find("child").get("link")
        if kind not in {"fixed", "revolute", "continuous"}:
            raise ValueError(f"Unsupported joint type: {kind}")
        if parent not in links or child not in links or child in children or name in names:
            raise ValueError("URDF has invalid or duplicate joint topology")
        if kind != "fixed" and name not in _JOINTS:
            raise ValueError(f"Unsupported moving joint: {name}")

        axis_element = None if kind == "fixed" else element.find("axis")
        axis = "1 0 0" if axis_element is None else axis_element.get("xyz", "1 0 0")
        axis = _vector(axis.split(), 3, "joint.axis")
        length = np.linalg.norm(axis)
        if length == 0:
            raise ValueError("Joint axis cannot be zero")
        joints.append(_Joint(name, parent, child, _origin(element), axis / length, kind))
        children.add(child)
        names.add(name)

    if names.intersection(_JOINTS) != set(_JOINTS):
        raise ValueError("URDF must contain the six SO101 joints")
    roots = links - children
    if len(roots) != 1:
        raise ValueError("URDF must have one root link")

    root = roots.pop()
    ordered, reached = [], {root}
    while joints:
        ready = [joint for joint in joints if joint.parent in reached]
        if not ready:
            raise ValueError("URDF has a disconnected or cyclic joint tree")
        for joint in ready:
            ordered.append(joint)
            reached.add(joint.child)
            joints.remove(joint)
    return root, tuple(ordered), links


class CalibrationFit:
    """Camera coordinates are +X right, +Y down, +Z forward; angles are radians.

    Parameters: camera rotvec[3], translation[3], focal_px, offsets[5], gripper_slope.
    Offsets are ordered shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.
    The pan offset is fixed at zero to remove its gauge with camera pose.
    """

    def __init__(self, urdf: Path, landmarks: dict, image_size: tuple, signs: tuple):
        self._root, self._joints, links = _tree(urdf)
        self._size = _vector(image_size, 2, "image_size")
        if np.any(self._size <= 0):
            raise ValueError("Image dimensions must be positive")
        self._signs = _vector(signs, _ARM_JOINTS, "signs")
        if not np.isin(self._signs, [-1, 1]).all():
            raise ValueError("Arm signs must be -1 or +1")
        self._landmarks = {}
        if not landmarks:
            raise ValueError("At least one landmark is required")
        for name, landmark in landmarks.items():
            if landmark["link"] not in links:
                raise ValueError(f"Unknown landmark link: {landmark['link']}")
            self._landmarks[name] = (
                landmark["link"], _vector(landmark["xyz"], 3, f"{name}.xyz")
            )

    def project(self, state, parameters):
        """Project every named rigid landmark; never clip behind-camera points."""
        parameters = self._parameters(parameters)
        points, depths = self._project(state, parameters)
        if np.any(depths <= _MIN_DEPTH):
            raise ValueError("Landmarks must lie in front of the camera")
        return {name: point.tolist() for name, point in zip(self._landmarks, points)}

    def fit(self, fit_rows, validation_rows, initial, lower, upper):
        """Fit only fit_rows; report held-out errors and conditional uncertainty."""
        initial = self._parameters(initial)
        lower = _vector(lower, len(_PARAMETERS), "lower bounds")
        upper = _vector(upper, len(_PARAMETERS), "upper bounds")
        if np.any(lower >= upper) or lower[_FOCAL] <= 0:
            raise ValueError("Bounds must increase and focal length must stay positive")
        if np.any(initial < lower) or np.any(initial > upper):
            raise ValueError("Initial parameters must lie inside the bounds")
        fit_data, validation_data = self._rows(fit_rows), self._rows(validation_rows)
        if sum(len(row[1]) for row in fit_data) * 2 <= len(_PARAMETERS):
            raise ValueError("Fit needs more measured pixel coordinates than parameters")

        result = least_squares(
            lambda values: self._residuals(fit_data, values), initial,
            bounds=(lower, upper), x_scale="jac", max_nfev=_MAX_EVALUATIONS,
            ftol=1e-11, xtol=1e-11, gtol=1e-11,
        )
        errors = self._residuals(fit_data, result.x)
        uncertainty = _uncertainty(result.jac, errors)
        return {
            "parameters": result.x.tolist(),
            "parameter_names": list(_PARAMETERS),
            "signs": self._signs.astype(int).tolist(),
            "pan_offset_rad": 0.0,
            "camera_coordinates": "+Xright,+Ydown,+Zforward",
            "principal_point": (self._size / 2).tolist(),
            "solver": {
                "success": bool(result.success), "message": result.message,
                "evaluations": result.nfev, "optimality": result.optimality,
                "active_bounds": result.active_mask.tolist(),
            },
            "fit": self._metrics(fit_data, result.x),
            "validation": self._metrics(validation_data, result.x),
            "uncertainty": uncertainty,
            "verification": "unverified",
        }

    def _parameters(self, values):
        values = _vector(values, len(_PARAMETERS), "parameters")
        if values[_FOCAL] <= 0:
            raise ValueError("Focal length must be positive")
        return values

    def _poses(self, state, parameters):
        state = _vector(state, len(_JOINTS), "state")
        angles = np.zeros(len(_JOINTS))
        angles[:_ARM_JOINTS] = self._signs * np.deg2rad(state[:_ARM_JOINTS])
        angles[-1] = parameters[_GRIPPER_SLOPE] * state[-1]
        angles[1:] += parameters[_OFFSETS]
        values = dict(zip(_JOINTS, angles))
        poses = {self._root: np.eye(4)}
        for joint in self._joints:
            rotation = np.eye(4)
            if joint.kind != "fixed":
                rotation[:3, :3] = Rotation.from_rotvec(
                    joint.axis * values[joint.name]
                ).as_matrix()
            poses[joint.child] = poses[joint.parent] @ joint.origin @ rotation
        return poses

    def _project(self, state, parameters):
        poses = self._poses(state, parameters)
        points = np.array([
            poses[link][:3, :3] @ point + poses[link][:3, 3]
            for link, point in self._landmarks.values()
        ])
        rotation = Rotation.from_rotvec(parameters[_CAMERA_ROTATION]).as_matrix()
        points = points @ rotation.T + parameters[_CAMERA_TRANSLATION]
        depths = points[:, 2]
        # Keep solver residuals finite; metrics reject invalid fitted depths below.
        projected = points[:, :2] / np.maximum(depths[:, None], _MIN_DEPTH)
        return projected * parameters[_FOCAL] + self._size / 2, depths

    def _rows(self, rows):
        names = list(self._landmarks)
        result = []
        for row in rows:
            state = _vector(row["state"], len(_JOINTS), "row.state")
            if not row["points"]:
                raise ValueError("Observation must have at least one visible landmark")
            indices, pixels = [], []
            for name, point in row["points"].items():
                if name not in self._landmarks:
                    raise ValueError(f"Unknown observed landmark: {name}")
                indices.append(names.index(name))
                pixels.append(_vector(point, 2, f"{name}.pixels"))
            result.append((state, np.array(indices), np.array(pixels)))
        return result

    def _residuals(self, rows, parameters):
        residuals = [
            (self._project(state, parameters)[0][indices] - pixels).ravel()
            for state, indices, pixels in rows
        ]
        return np.concatenate(residuals) if residuals else np.array([])

    def _metrics(self, rows, parameters):
        if not rows:
            return None
        errors = self._residuals(rows, parameters).reshape(-1, 2)
        norms = np.linalg.norm(errors, axis=1)
        depths = np.concatenate([
            self._project(state, parameters)[1][indices] for state, indices, _ in rows
        ])
        per_landmark = {}
        for index, name in enumerate(self._landmarks):
            selected = [
                self._project(state, parameters)[0][indices[indices == index]]
                - pixels[indices == index]
                for state, indices, pixels in rows if index in indices
            ]
            if selected:
                residuals = np.concatenate(selected)
                per_landmark[name] = {
                    "count": len(residuals),
                    "rms_px": float(np.sqrt(np.mean(np.sum(residuals**2, axis=1)))),
                }
        return {
            "frames": len(rows), "points": len(norms),
            "rms_px": float(np.sqrt(np.mean(norms**2))),
            "median_px": float(np.median(norms)),
            "p95_px": float(np.percentile(norms, 95)),
            "max_px": float(norms.max()),
            "behind_camera_points": int(np.sum(depths <= _MIN_DEPTH)),
            "per_landmark": per_landmark,
        }


def _uncertainty(jacobian, residuals):
    _, singular, vh = np.linalg.svd(jacobian, full_matrices=False)
    tolerance = np.finfo(float).eps * max(jacobian.shape) * singular[0]
    rank = int(np.sum(singular > tolerance))
    columns = jacobian.shape[1]
    condition = float(singular[0] / singular[-1]) if rank == columns else None
    standard_error = None
    if rank == columns:
        variance = float(residuals @ residuals) / (len(residuals) - columns)
        covariance = (vh.T / singular**2) @ vh * variance
        standard_error = np.sqrt(np.maximum(np.diag(covariance), 0)).tolist()
    return {
        "jacobian_singular_values": singular.tolist(), "jacobian_rank": rank,
        "condition_number": condition, "standard_errors": standard_error,
        "scope": "Local linear fit; fixed signs, pinhole intrinsics and independent pixel noise",
    }
