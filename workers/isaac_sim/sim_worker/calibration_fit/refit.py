"""Offline constrained camera/joint fits; held-out image labels never select fits."""

import argparse
import copy
import hashlib
import inspect
import json
import math
import xml.etree.ElementTree as ET
from enum import Enum
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml
from pxr import Usd, UsdGeom
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from sim_worker.calibration_fit import CalibrationFit
from sim_worker.calibration_fit.replay import RecordedStates

_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
_TIPS = ("fixed_tip", "moving_tip")
_TRAIN_EPISODES = {0, 2}
_HOLDOUT_EPISODES = {1, 3}
_LAG_BOUNDS = (-6.0, 0.0)
_LAG_SEEDS = (-5.0, -2.6733349882040764, -0.5)
_PIXEL_SIGMA = 3.0
_JAW_SIGMA_M = 0.002
_JAW_BOUND_M = 0.005
_SUPPORT_TOL = 1e-8
_MAX_EVALUATIONS = 600
_RMS_LIMIT_PX = 9.0
_MAX_LIMIT_PX = 24.0
_ANGLE_STD_LIMIT_DEG = 2.0
_TIME_TOLERANCE_S = 1e-6
_PARAM_NAMES = (
    "camera_rx",
    "camera_ry",
    "camera_rz",
    "camera_tx",
    "camera_ty",
    "camera_tz",
    "focal_px",
    "shoulder_lift_offset",
    "elbow_flex_offset",
    "wrist_flex_offset",
    "wrist_roll_offset",
    "gripper_start_angle",
    "gripper_remaining_fraction",
    "lag_frames",
)
_SIGNS = ((1, 1, 1, 1, 1), (1, 1, 1, 1, -1))


class _Mode(Enum):
    FIXED = "fixed_current_camera"
    FREE = "free_camera"
    GEOMETRY = "free_camera_jaw_prior"


class _LimitMode(Enum):
    REQUESTED = "requested"
    CLIPPED = "clipped"


def _read(path):
    return json.loads(path.read_text())


def _write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recorded_source(dataset, provenance):
    metadata = _read(provenance)
    files = [item for item in metadata["source_files"] if item["path"].startswith("data/")]
    if len(files) != 1 or _hash(dataset) != files[0]["sha256"]:
        raise ValueError("Dataset must match the pinned Parquet SHA256")
    info = metadata["confirmed_metadata"]
    if tuple(name.removesuffix(".pos") for name in info["joint_order"]) != _JOINTS:
        raise ValueError("Pinned joint order differs from the SO101 fit")
    fps = info["fps"]
    if type(fps) not in (float, int) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Pinned frame rate must be positive")
    rows = pq.read_table(dataset).to_pylist()
    records = RecordedStates(rows, _JOINTS)
    if len(rows) != info["frames"] or len(records.episodes) != info["episodes"]:
        raise ValueError("Dataset counts differ from pinned metadata")
    declared = {item["episode"]: item for item in metadata["episodes"]}
    if set(declared) != set(records.episodes):
        raise ValueError("Dataset episodes differ from pinned metadata")

    grouped = {episode: [] for episode in records.episodes}
    for row in rows:
        values = np.asarray(row["action"], dtype=float)
        if values.shape != (len(_JOINTS),) or not np.isfinite(values).all():
            raise ValueError("Recorded action must have six finite values")
        if not math.isclose(row["timestamp"], row["frame_index"] / fps, abs_tol=_TIME_TOLERANCE_S):
            raise ValueError("Recorded timestamp differs from pinned control rate")
        grouped[row["episode_index"]].append(row["index"])
    for episode, indices in grouped.items():
        expected = declared[episode]
        if (
            len(indices) != expected["frames"]
            or [min(indices), max(indices) + 1] != expected["data_rows_half_open"]
        ):
            raise ValueError("Episode boundaries differ from pinned metadata")
    return (
        {(row["episode_index"], row["index"]): row for row in rows},
        {
            "dataset": metadata["dataset"],
            "recorded_file": files[0],
            "provenance_sha256": _hash(provenance),
            "episode_boundaries_verified": True,
            "joint_order_verified": True,
            "timestamp_rate_verified": True,
        },
    )


def _limits(path):
    robot = ET.parse(path).getroot()
    return np.array(
        [
            [
                float(robot.find(f"joint[@name='{name}']/limit").get(key))
                for key in ("lower", "upper")
            ]
            for name in _JOINTS
        ]
    )


def _camera(path, width):
    stage = Usd.Stage.Open(str(path))
    camera = UsdGeom.Camera(stage.GetPrimAtPath("/World/Cameras/Front"))
    camera_world = np.array(
        UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    ).T
    base = np.array(
        UsdGeom.Xformable(stage.GetPrimAtPath("/World/Robot")).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
    ).T
    axes = np.diag([1.0, -1.0, -1.0, 1.0])
    camera_from_robot = axes @ np.linalg.inv(camera_world) @ base
    focal = camera.GetFocalLengthAttr().Get() / camera.GetHorizontalApertureAttr().Get() * width
    return np.r_[
        Rotation.from_matrix(camera_from_robot[:3, :3]).as_rotvec(), camera_from_robot[:3, 3], focal
    ], base


def _state(row, source, lag):
    index = row["frame"] + lag
    left, right = int(np.floor(index)), int(np.ceil(index))
    keys = ((row["episode"], left), (row["episode"], right))
    if any(key not in source for key in keys):
        raise ValueError("Timing interpolation would cross episode boundary")
    a, b = (np.asarray(source[key]["observation.state"]) for key in keys)
    return a + (b - a) * (index - left)


def _support(source, episodes):
    selected = [row for row in source.values() if row["episode_index"] in episodes]
    vectors = np.array([row[key] for row in selected for key in ("observation.state", "action")])
    return np.array([vectors.min(axis=0), vectors.max(axis=0)]).T


def _bounds(support, limits, signs):
    angular = np.deg2rad(support[:5]) * np.array(signs)[:, None]
    low_offsets = limits[1:5, 0] - angular[1:].min(axis=1)
    high_offsets = limits[1:5, 1] - angular[1:].max(axis=1)
    if np.any(low_offsets >= high_offsets):
        raise ValueError("The observed support cannot fit the URDF with an offset")
    low = np.r_[
        [-6.3] * 3, [-4.0, -4.0, 0.2], 500.0, low_offsets, limits[-1, 0], 0.001, _LAG_BOUNDS[0]
    ]
    high = np.r_[
        [6.3] * 3, [4.0, 4.0, 5.0], 6000.0, high_offsets, limits[-1, 1] - 1e-7, 1.0, _LAG_BOUNDS[1]
    ]
    return low, high


def _physical(values, support, limits):
    # The observed gripper interval maps monotonically inside URDF limits.
    start = values[11]
    end = start + (limits[-1, 1] - start) * values[12]
    slope = (end - start) / (support[-1, 1] - support[-1, 0])
    offset = start - slope * support[-1, 0]
    return np.r_[values[:11], offset, slope]


def _encode(parameters, support, limits, lag):
    start = parameters[11] + parameters[12] * support[-1, 0]
    end = parameters[11] + parameters[12] * support[-1, 1]
    fraction = (end - start) / (limits[-1, 1] - start)
    return np.r_[parameters[:11], start, fraction, lag]


def _angles(states, parameters, signs):
    values = np.deg2rad(states) * np.r_[signs, 1]
    values[..., -1] = states[..., -1] * parameters[12]
    values[..., 1:] += parameters[7:12]
    return values


def _model(context, signs, shifts):
    landmarks = copy.deepcopy(context["landmarks"])
    for index, name in enumerate(_TIPS):
        landmarks[name]["xyz"] = (
            np.array(landmarks[name]["xyz"]) + shifts[index * 3 : index * 3 + 3]
        ).tolist()
    return context["factory"](context["urdf"], landmarks, tuple(context["image_size"]), signs)


def _predictions(context, rows, values, signs, shifts, limit_mode):
    model = _model(context, signs, shifts)
    physical = _physical(values, context["support"], context["limits"])
    result = []
    for row in rows:
        state = _state(row, context["source"], values[13])
        if limit_mode == _LimitMode.CLIPPED:
            angles = np.clip(_angles(state, physical, signs), *context["limits"].T)
            state = np.rad2deg(angles)
            state[:5] = np.rad2deg(angles[:5] - np.r_[0.0, physical[7:11]]) / np.array(signs)
            state[-1] = (angles[-1] - physical[11]) / physical[12]
        points = model.project(state, physical)
        for name, target in row["points"].items():
            result.append((row["frame"], name, np.asarray(points[name]) - target))
    return result


def _residual(context, rows, values, signs, mode):
    shifts = values[14:] if mode == _Mode.GEOMETRY else np.zeros(6)
    predicted = _predictions(context, rows, values, signs, shifts, _LimitMode.REQUESTED)
    residual = np.concatenate([entry[2] for entry in predicted]) / _PIXEL_SIGMA
    if mode == _Mode.GEOMETRY:
        residual = np.r_[residual, shifts / _JAW_SIGMA_M]
    return residual


def _metrics(context, rows, values, signs, shifts, limit_mode=_LimitMode.REQUESTED):
    points = _predictions(context, rows, values, signs, shifts, limit_mode)
    errors = np.array([np.linalg.norm(entry[2]) for entry in points])
    grouped = {}
    for _, name, residual in points:
        grouped.setdefault(name, []).append(float(np.linalg.norm(residual)))
    return {
        "frames": len(rows),
        "points": len(points),
        "rms_px": float(np.sqrt(np.mean(errors**2))),
        "max_px": float(errors.max()),
        "p95_px": float(np.percentile(errors, 95)),
        "per_landmark": {
            name: {"count": len(values), "rms_px": float(np.sqrt(np.mean(np.square(values))))}
            for name, values in grouped.items()
        },
    }


def _uncertainty(result, active, labels):
    _, singular, vh = np.linalg.svd(result.jac, full_matrices=False)
    rank = int(np.sum(singular > np.finfo(float).eps * max(result.jac.shape) * singular[0]))
    report = {
        "rank": rank,
        "parameters": len(active),
        "parameter_names": [labels[i] for i in active],
        "condition_number": float(singular[0] / singular[-1]),
        "active_bounds": {
            labels[i]: int(flag) for i, flag in zip(active, result.active_mask, strict=True) if flag
        },
        "scope": (
            "Local linear diagnostic only. Active constraints and approximate jaw geometry "
            "invalidate an unconstrained physical confidence interpretation."
        ),
    }
    if rank != len(active):
        report["standard_errors"] = None
        return report
    variance = float(result.fun @ result.fun) / (len(result.fun) - len(active))
    covariance = (vh.T / singular**2) @ vh * variance
    report["standard_errors"] = dict(
        zip(
            report["parameter_names"],
            np.sqrt(np.maximum(np.diag(covariance), 0)).tolist(),
            strict=True,
        )
    )
    report["covariance"] = covariance.tolist()
    return report


def _fit(context, mode, signs, seeds):
    low, high = _bounds(context["support"], context["limits"], signs)
    labels = list(_PARAM_NAMES)
    active = list(range(7, 14)) if mode == _Mode.FIXED else list(range(14))
    if mode == _Mode.GEOMETRY:
        low, high = np.r_[low, [-_JAW_BOUND_M] * 6], np.r_[high, [_JAW_BOUND_M] * 6]
        active += list(range(14, 20))
        labels += [name + "_" + axis + "_m" for name in _TIPS for axis in "xyz"]
    candidates = []
    for index, seed in enumerate(seeds):
        initial = np.r_[seed[:14], np.zeros(6)] if mode == _Mode.GEOMETRY else seed[:14].copy()
        if mode == _Mode.FIXED:
            initial[:7] = context["camera"]
        initial = np.clip(initial, low + 1e-7, high - 1e-7)

        def expand(values):
            full = initial.copy()
            full[active] = values
            return full

        fit = least_squares(
            lambda values: _residual(context, context["train"], expand(values), signs, mode),
            initial[active],
            bounds=(low[active], high[active]),
            x_scale="jac",
            max_nfev=_MAX_EVALUATIONS,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        candidates.append((fit, expand(fit.x)))
        print(
            json.dumps(
                {
                    "mode": mode.value,
                    "signs": signs,
                    "seed": index,
                    "train_cost": float(fit.cost),
                    "success": bool(fit.success),
                }
            ),
            flush=True,
        )
    result, values = min(candidates, key=lambda pair: pair[0].cost)
    shifts = values[14:] if mode == _Mode.GEOMETRY else np.zeros(6)
    return {
        "status": "unverified",
        "mode": mode.value,
        "signs": list(signs),
        "optimizer_parameters": values.tolist(),
        "parameters": _physical(values, context["support"], context["limits"]).tolist(),
        "parameter_names": list(_PARAM_NAMES[:11]) + ["gripper_offset", "gripper_slope"],
        "lag_frames": float(values[13]),
        "jaw_shifts_m": dict(zip(_TIPS, shifts.reshape(2, 3).tolist(), strict=True)),
        "train": _metrics(context, context["train"], values, signs, shifts),
        "train_objective": float(result.cost),
        "solver": {
            "success": bool(result.success),
            "message": result.message,
            "evaluations": result.nfev,
        },
        "uncertainty": _uncertainty(result, active, labels),
        "selection": "Minimum training objective; held-out labels and states excluded.",
        "geometry_prior": {
            "bound_m_per_coordinate": _JAW_BOUND_M,
            "gaussian_sigma_m": _JAW_SIGMA_M,
            "pixel_sigma_px": _PIXEL_SIGMA,
        }
        if mode == _Mode.GEOMETRY
        else None,
    }


def _support_audit(context, parameters, signs):
    report = {}
    for feature in ("observation.state", "action"):
        states = np.array([row[feature] for row in context["source"].values()])
        mapped = _angles(states, np.array(parameters), signs)
        violations = (mapped < context["limits"][:, 0] - _SUPPORT_TOL) | (
            mapped > context["limits"][:, 1] + _SUPPORT_TOL
        )
        report[feature] = {
            "rows": len(states),
            "rows_requiring_clipping": int(violations.any(axis=1).sum()),
            "joints": {
                name: {
                    "range_rad": [float(mapped[:, index].min()), float(mapped[:, index].max())],
                    "urdf_rad": context["limits"][index].tolist(),
                    "clipping_count": int(violations[:, index].sum()),
                    "max_excess_rad": float(
                        np.maximum(
                            np.maximum(
                                context["limits"][index, 0] - mapped[:, index],
                                mapped[:, index] - context["limits"][index, 1],
                            ),
                            0,
                        ).max()
                    ),
                }
                for index, name in enumerate(_JOINTS)
            },
        }
    return report


def _shelf_metrics(context, parameters):
    points = np.array(
        [
            [-0.5, -0.16, 0.2],
            [0.5, -0.16, 0.2],
            [-0.5, 0.16, 0.2],
            [0.5, 0.16, 0.2],
            [-0.5, -0.16, 0.67],
            [0.5, -0.16, 0.67],
            [-0.5, -0.16, 0.011],
            [0.5, -0.16, 0.011],
        ]
    )
    robot_points = (np.c_[points, np.ones(len(points))] @ np.linalg.inv(context["base"]).T)[:, :3]
    transformed = (
        robot_points @ Rotation.from_rotvec(parameters[:3]).as_matrix().T + parameters[3:6]
    )
    pixels = (
        transformed[:, :2] / transformed[:, 2, None] * parameters[6]
        + np.array(context["image_size"]) / 2
    )
    reference = np.array(context["shelf"]["correspondences_image_pixels"])
    error = np.linalg.norm(pixels - reference, axis=1)
    return {
        "rms_px": float(np.sqrt(np.mean(error**2))),
        "max_px": float(error.max()),
        "scope": "Shelf labels are diagnostic only; never optimized or used for selection.",
        "projected_pixels": pixels.tolist(),
    }


def _evaluate(context, result):
    values, signs = np.array(result["optimizer_parameters"]), result["signs"]
    shifts = np.array([result["jaw_shifts_m"][name] for name in _TIPS]).ravel()
    result["heldout"] = _metrics(context, context["heldout"], values, signs, shifts)
    result["full_support"] = _support_audit(context, result["parameters"], signs)
    result["shelf_camera_check"] = _shelf_metrics(context, np.array(result["parameters"]))
    result["acceptance"] = _acceptance(context, result)
    result["recommendation"] = "do_not_apply"
    return result


def _acceptance(context, result):
    uncertainty = result["uncertainty"]
    errors = uncertainty["standard_errors"] or {}
    offsets = {
        name: float(np.rad2deg(errors[name])) if name in errors else None
        for name in _PARAM_NAMES[7:11]
    }
    gripper_errors = []
    if uncertainty.get("covariance"):
        names = uncertainty["parameter_names"]
        indices = [names.index(name) for name in _PARAM_NAMES[11:13]]
        covariance = np.array(uncertainty["covariance"])[np.ix_(indices, indices)]
        start, fraction = result["optimizer_parameters"][11:13]
        for position, value in zip((0.0, 1.0), context["support"][-1], strict=True):
            gradient = np.array(
                [1 - fraction * position, (context["limits"][-1, 1] - start) * position]
            )
            error = float(np.rad2deg(np.sqrt(max(0.0, gradient @ covariance @ gradient))))
            gripper_errors.append({"policy": float(value), "std_deg": error})
    checks = {
        "heldout_rms": result["heldout"]["rms_px"] <= _RMS_LIMIT_PX,
        "heldout_max": result["heldout"]["max_px"] <= _MAX_LIMIT_PX,
        "solver_converged": result["solver"]["success"],
        "no_active_bounds": not uncertainty["active_bounds"],
        "offset_uncertainty": all(
            value is not None and value <= _ANGLE_STD_LIMIT_DEG for value in offsets.values()
        ),
        "gripper_uncertainty": len(gripper_errors) == 2
        and all(item["std_deg"] <= _ANGLE_STD_LIMIT_DEG for item in gripper_errors),
        "full_support_inside_urdf": all(
            item["rows_requiring_clipping"] == 0 for item in result["full_support"].values()
        ),
        "independent_geometry": False,
        "independent_sign_evidence": False,
    }
    return {
        "status": "unverified",
        "accepted": False,
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
        "thresholds": {
            "heldout_rms_px": _RMS_LIMIT_PX,
            "heldout_max_px": _MAX_LIMIT_PX,
            "angle_std_deg": _ANGLE_STD_LIMIT_DEG,
        },
        "conditional_offset_std_deg": offsets,
        "conditional_gripper_endpoint_std_deg": gripper_errors,
        "uncertainty_warning": (
            "Local covariance conditions on chosen signs and assumed geometry. "
            "Active bounds and imposed priors prevent independent physical confidence."
        ),
    }


def _candidate_map(context, result):
    support = context["support"]
    mapped = _angles(support.T, np.array(result["parameters"]), result["signs"]).T
    curves = {}
    for index, name in enumerate(_JOINTS):
        order = np.argsort(mapped[index])
        curves[name] = {
            "sim_rad": mapped[index, order].tolist(),
            "policy": support[index, order].tolist(),
        }
    return {
        "api_version": "simulation.joints/v1alpha1",
        "status": "unverified",
        "source": (
            "Offline fit; paired camera and landmark hypotheses in selected-candidate.json. "
            "Do not apply independently."
        ),
        "joints": curves,
    }


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--landmarks", type=Path)
    parser.add_argument("--prior-report", type=Path)
    parser.add_argument("--provenance", type=Path)
    args = parser.parse_args()
    scene = args.scene_dir.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    landmark_path = args.landmarks or scene / "evidence/calibration/landmarks.json"
    prior_path = args.prior_report or scene / "evidence/calibration/report.json"
    data, prior = _read(landmark_path), _read(prior_path)
    provenance = args.provenance or scene / "evidence/dataset.json"
    source, source_audit = _recorded_source(args.dataset, provenance)
    constraint_episodes = {row["episode_index"] for row in source.values()} - _HOLDOUT_EPISODES
    support = _support(source, constraint_episodes)
    camera, base = _camera(scene / "scene.usda", data["image_size"][0])
    urdf = scene / "robot/source/so101_new_calib.urdf"
    context = {
        "factory": CalibrationFit,
        "source": source,
        "support": support,
        "limits": _limits(urdf),
        "urdf": urdf,
        "landmarks": data["landmarks"],
        "image_size": data["image_size"],
        "camera": camera,
        "base": base,
        "shelf": _read(scene / "evidence/camera_fit.json"),
        "train": [row for row in data["samples"] if row["episode"] in _TRAIN_EPISODES],
        "heldout": [row for row in data["samples"] if row["episode"] in _HOLDOUT_EPISODES],
    }
    protocol = {
        "status": "unverified",
        "train_image_episodes": sorted(_TRAIN_EPISODES),
        "heldout_episodes": sorted(_HOLDOUT_EPISODES),
        "support_constraint_episodes": sorted(constraint_episodes),
        "support": support.tolist(),
        "lag_definition": (
            "Video i uses episode-local observation.state[i + lag]; "
            "frame indices are global dataset indices."
        ),
        "lag_bounds_frames": list(_LAG_BOUNDS),
        "sign_hypotheses": [list(signs) for signs in _SIGNS],
        "geometry_prior": {"bound_m": _JAW_BOUND_M, "sigma_m": _JAW_SIGMA_M},
        "inputs_sha256": {
            str(path): _hash(path)
            for path in (args.dataset, landmark_path, prior_path, urdf, scene / "scene.usda")
        },
        "runtime_modified": False,
        "simulation_started": False,
    }
    protocol["source_audit"] = source_audit
    protocol["implementation_sha256"] = {
        "refit": _hash(Path(__file__)),
        "core": _hash(Path(inspect.getfile(CalibrationFit))),
    }
    _write(args.output / "protocol.json", protocol)
    old = np.array(prior["parameters"][:13])
    initial = [_encode(old, support, context["limits"], lag) for lag in _LAG_SEEDS]
    fits = []
    # Freeze every training solution before computing held-out metrics.
    for signs in _SIGNS:
        for mode in (_Mode.FIXED, _Mode.FREE):
            result = _fit(context, mode, signs, initial)
            name = mode.value + ("_roll_positive" if signs[-1] == 1 else "_roll_negative")
            result["name"] = name
            fits.append(result)
            _write(args.output / (name + ".training.json"), result)
    selected = min(
        (fit for fit in fits if fit["mode"] == _Mode.FREE.value),
        key=lambda fit: fit["train_objective"],
    )
    geometry_seeds = [np.array(selected["optimizer_parameters"]) for _ in _LAG_SEEDS]
    for seed, lag in zip(geometry_seeds, _LAG_SEEDS, strict=True):
        seed[13] = lag
    geometry = _fit(context, _Mode.GEOMETRY, tuple(selected["signs"]), geometry_seeds)
    geometry["name"] = "jaw_geometry_sensitivity"
    fits.append(geometry)
    _write(
        args.output / "selection-frozen.json",
        {
            "selected_nominal": selected,
            "geometry_sensitivity": geometry,
            "selection_uses_heldout": False,
            "note": (
                "Export uses nominal geometry. Adjusted geometry is a sensitivity diagnostic, "
                "not independently measured."
            ),
        },
    )
    baselines = []
    for name, parameters in [
        ("old_camera_old_map", old),
        ("current_camera_old_map", np.r_[camera, old[7:]]),
    ]:
        values = _encode(parameters, support, context["limits"], prior["lag_frames"])
        result = {
            "name": name,
            "parameters": parameters.tolist(),
            "lag_frames": float(values[13]),
            "train": _metrics(context, context["train"], values, (1, 1, 1, 1, 1), np.zeros(6)),
            "heldout": _metrics(context, context["heldout"], values, (1, 1, 1, 1, 1), np.zeros(6)),
            "train_with_urdf_clipping": _metrics(
                context, context["train"], values, (1, 1, 1, 1, 1), np.zeros(6), _LimitMode.CLIPPED
            ),
            "heldout_with_urdf_clipping": _metrics(
                context,
                context["heldout"],
                values,
                (1, 1, 1, 1, 1),
                np.zeros(6),
                _LimitMode.CLIPPED,
            ),
            "full_support": _support_audit(context, parameters, (1, 1, 1, 1, 1)),
            "shelf_camera_check": _shelf_metrics(context, parameters),
        }
        baselines.append(result)
    results = [_evaluate(context, result) for result in fits]
    selected = next(result for result in results if result["name"] == selected["name"])
    _write(args.output / "selected-candidate.json", selected)
    (args.output / "calibration.candidate.yaml").write_text(
        yaml.safe_dump(_candidate_map(context, selected), sort_keys=False)
    )
    report = {
        "protocol": protocol,
        "baselines": baselines,
        "candidates": results,
        "selected_nominal_candidate": selected["name"],
        "candidate_status": "unverified",
        "limitations": [
            "Two training episodes provide robot labels; holdout cannot establish metric geometry.",
            (
                "Current camera fits shelf context; robot-only free-camera fits can harm the "
                "background and do not calibrate the whole scene."
            ),
            "Support uses non-heldout data, not hardware calibration. URDF limits are unchanged.",
            (
                "Pan zero, principal point, lens model, and joint signs are assumptions. "
                "Arm scale is fixed at signed pi/180 radians per policy unit."
            ),
            "Approximate jaw labels create ambiguity. The ±5mm geometry prior is assumed.",
            (
                "Less clipping does not establish physical accuracy. "
                "Local covariance under active constraints cannot guarantee calibration."
            ),
            "Training-only fitted timing must not become negative runtime policy latency.",
            "Review camera and map as a pair. Runtime calibration and scene were not modified.",
        ],
    }
    _write(args.output / "report.json", report)
    summary = {
        "selected": selected["name"],
        "baseline": [
            {
                "name": item["name"],
                "train_rms": item["train"]["rms_px"],
                "heldout_rms": item["heldout"]["rms_px"],
                "shelf_rms": item["shelf_camera_check"]["rms_px"],
            }
            for item in baselines
        ],
        "candidates": [
            {
                "name": item["name"],
                "train_rms": item["train"]["rms_px"],
                "heldout_rms": item["heldout"]["rms_px"],
                "shelf_rms": item["shelf_camera_check"]["rms_px"],
                "lag_frames": item["lag_frames"],
                "active_bounds": item["uncertainty"]["active_bounds"],
            }
            for item in results
        ],
    }
    _write(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    _main()
