"""Fail-closed acceptance checks for a visual calibration fit."""

import math

import numpy as np

_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
_ARM_OFFSETS = tuple(f"{name}_offset" for name in _JOINTS[1:-1])
_PARAMETERS = {
    "camera_rx",
    "camera_ry",
    "camera_rz",
    "camera_tx",
    "camera_ty",
    "camera_tz",
    "focal_px",
    *_ARM_OFFSETS,
    "gripper_offset",
    "gripper_slope",
}
_THRESHOLDS = {
    "validation_rms_pixels": 9,
    "validation_max_pixels": 24,
    "joint_offset_std_degrees": 2,
    "gripper_angle_std_degrees": 2,
}
_SOURCES = ("state", "action")
_COVARIANCE_TOLERANCE = 1e-10


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _vector(values, size):
    return (
        isinstance(values, (list, tuple))
        and len(values) == size
        and all(_finite(value) for value in values)
    )


def _record(checks, failures, name, condition, reason):
    checks[name] = bool(condition)
    if not condition:
        failures.append(reason)


def _episodes(metadata):
    groups = [metadata.get(name) for name in ("fit_episodes", "validation_episodes")]
    for group in groups:
        if not isinstance(group, list) or not group:
            return False
        if any(isinstance(value, bool) or not isinstance(value, (str, int)) for value in group):
            return False
        if any(isinstance(value, str) and not value.strip() for value in group):
            return False
    # Canonical IDs prevent episode 1 and "1" from appearing independent.
    return set(map(str, groups[0])).isdisjoint(map(str, groups[1]))


def _evidence(value, status):
    return (
        isinstance(value, dict)
        and value.get("status") == status
        and isinstance(value.get("source"), str)
        and bool(value["source"].strip())
    )


def _support(support):
    for source in _SOURCES:
        bounds = support.get(source, {})
        if not isinstance(bounds, dict):
            return False
        lower, upper = bounds.get("min"), bounds.get("max")
        if not _vector(lower, len(_JOINTS)) or not _vector(upper, len(_JOINTS)):
            return False
        if any(left > right for left, right in zip(lower, upper)):
            return False
    return True


def _limits(limits):
    for name in _JOINTS:
        bounds = limits.get(name)
        if not _vector(bounds, 2) or bounds[0] >= bounds[1]:
            return False
    return True


def _map_support(support, parameters, signs):
    result = {}
    for source in _SOURCES:
        result[source] = {}
        for index, name in enumerate(_JOINTS):
            lower, upper = support[source]["min"][index], support[source]["max"][index]
            if name == "gripper":
                scale, offset = parameters["gripper_slope"], parameters["gripper_offset"]
            else:
                scale = math.radians(signs[index])
                offset = parameters.get(f"{name}_offset", 0.0)
            result[source][name] = sorted((lower * scale + offset, upper * scale + offset))
    return result


def _covariance(values, errors):
    size = len(errors)
    if not isinstance(values, (list, tuple)) or len(values) != size:
        return None
    if not all(_vector(row, size) for row in values):
        return None
    matrix = np.asarray(values, dtype=float)
    diagonal = np.diag(matrix)
    if np.any(diagonal < 0) or not np.allclose(
        np.sqrt(diagonal), errors, rtol=_COVARIANCE_TOLERANCE, atol=0
    ):
        return None

    # Normalize mixed parameter units before checking symmetry and PSD.
    scales = np.asarray(errors)
    if np.any(matrix[scales == 0] != 0) or np.any(matrix[:, scales == 0] != 0):
        return None
    scales = np.where(scales > 0, scales, 1.0)
    correlation = matrix / scales[:, None] / scales[None, :]
    if not np.isfinite(correlation).all() or not np.allclose(
        correlation, correlation.T, rtol=0, atol=_COVARIANCE_TOLERANCE
    ):
        return None
    if np.linalg.eigvalsh((correlation + correlation.T) / 2).min() < -_COVARIANCE_TOLERANCE:
        return None
    return (matrix + matrix.T) / 2


def _gripper_error(names, support, errors, covariance):
    openings = [support[source][bound][-1] for source in _SOURCES for bound in ("min", "max")]
    indices = [names.index(name) for name in ("gripper_offset", "gripper_slope")]
    if covariance is None:
        # Without covariance, the sum bounds every possible correlation.
        error = errors[indices[0]] + max(map(abs, openings)) * errors[indices[1]]
        return math.degrees(error), "worst_correlation_bound"

    matrix = covariance[np.ix_(indices, indices)]
    # PSD makes angular variance convex in opening; endpoints bound the full range.
    variances = [
        matrix[0, 0] + 2 * value * matrix[0, 1] + value**2 * matrix[1, 1]
        for value in (min(openings), max(openings))
    ]
    return math.degrees(math.sqrt(max(0, *variances))), "propagated_covariance"


def evaluate(report, support, limits, metadata, thresholds=None):
    """Check a fit without exporting a calibration or modifying its status.

    Support contains state/action min/max vectors in policy coordinates. Limits
    map six joint names to [lower, upper] radians. Episode and provenance metadata
    must be supplied independently; a low reprojection error cannot prove them.
    Uncertainty thresholds are one standard error, not 95% interval widths.
    """
    if thresholds is not None and thresholds != _THRESHOLDS:
        raise ValueError("Acceptance thresholds are fixed before fitting")

    checks, failures = {}, []
    names, values = report.get("parameter_names", []), report.get("parameters", [])
    valid_names = (
        isinstance(names, list)
        and all(isinstance(name, str) for name in names)
        and len(set(names)) == len(names)
        and _PARAMETERS.issubset(names)
        and set(names).issubset(_PARAMETERS | {"frame_lag"})
    )
    valid_parameters = valid_names and _vector(values, len(names))
    parameters = dict(zip(names, values)) if valid_parameters else {}
    valid_parameters = valid_parameters and parameters["focal_px"] > 0
    signs = report.get("signs")
    valid_signs = _vector(signs, len(_JOINTS) - 1) and all(value in (-1, 1) for value in signs)
    _record(
        checks, failures, "parameters", valid_parameters, "Fit parameters are invalid or missing"
    )
    _record(checks, failures, "signs", valid_signs, "Five arm signs must each be -1 or +1")

    solver = report.get("solver") or {}
    active = solver.get("active_bounds")
    _record(checks, failures, "solver", solver.get("success") is True, "Solver did not succeed")
    _record(
        checks,
        failures,
        "bounds",
        _vector(active, len(names)) and not any(active),
        "Fit has active or missing solver bounds",
    )
    for split in ("fit", "validation"):
        metrics = report.get(split) or {}
        present = all(
            _finite(metrics.get(key)) and metrics[key] > 0 for key in ("frames", "points")
        )
        _record(
            checks,
            failures,
            f"{split}_depth",
            present and metrics.get("behind_camera_points") == 0,
            f"{split} must contain observations and zero behind-camera points",
        )
    metrics = report.get("validation") or {}
    for metric, threshold in (
        ("rms_px", "validation_rms_pixels"),
        ("max_px", "validation_max_pixels"),
    ):
        value = metrics.get(metric)
        _record(
            checks,
            failures,
            metric,
            _finite(value) and 0 <= value <= _THRESHOLDS[threshold],
            f"Held-out {metric} must be at most {_THRESHOLDS[threshold]} pixels",
        )

    _record(
        checks,
        failures,
        "episodes",
        _episodes(metadata),
        "Fit and validation need explicit, nonempty, disjoint episode IDs",
    )
    _record(
        checks,
        failures,
        "pan_gauge",
        report.get("pan_offset_rad") == 0 and metadata.get("pan_offset_gauge") == "fixed_zero",
        "Zero pan offset must be recorded as an imposed camera-pose gauge",
    )
    geometry = metadata.get("landmark_geometry") or {}
    _record(
        checks,
        failures,
        "geometry",
        _evidence(geometry, "proven")
        and geometry.get("independent") is True
        and geometry.get("uncertainty_included") is True,
        "Independent landmark geometry and its propagated uncertainty remain unproven",
    )
    _record(
        checks,
        failures,
        "sign_provenance",
        _evidence(metadata.get("sign_ambiguity"), "resolved"),
        "Competing joint-sign hypotheses have not been independently resolved",
    )

    valid_support, valid_limits = _support(support), _limits(limits)
    _record(
        checks,
        failures,
        "support",
        valid_support,
        "Full six-joint state and action support is required",
    )
    _record(
        checks, failures, "limits", valid_limits, "Six finite increasing URDF limits are required"
    )
    mapped = (
        _map_support(support, parameters, signs)
        if valid_support and valid_parameters and valid_signs
        else {}
    )
    for source in _SOURCES:
        for name in _JOINTS:
            bounds = mapped.get(source, {}).get(name)
            inside = (
                bool(bounds)
                and valid_limits
                and limits[name][0] <= bounds[0] <= bounds[1] <= limits[name][1]
            )
            _record(
                checks,
                failures,
                f"{source}.{name}",
                inside,
                f"{source}.{name} mapped support must lie inside its URDF limits",
            )

    uncertainty = report.get("uncertainty") or {}
    errors = uncertainty.get("standard_errors")
    finite_errors = (
        valid_names and _vector(errors, len(names)) and all(value >= 0 for value in errors)
    )
    _record(
        checks,
        failures,
        "rank",
        valid_names and uncertainty.get("jacobian_rank") == len(names),
        "Calibration Jacobian must have full parameter rank",
    )
    _record(
        checks,
        failures,
        "standard_errors",
        finite_errors,
        "Every fitted parameter needs a finite nonnegative standard error",
    )
    degrees = {}
    if finite_errors:
        by_name = dict(zip(names, errors))
        degrees = {name: math.degrees(by_name[name]) for name in _ARM_OFFSETS}
    for name in _ARM_OFFSETS:
        value = degrees.get(name)
        _record(
            checks,
            failures,
            name,
            value is not None and value <= _THRESHOLDS["joint_offset_std_degrees"],
            f"{name} standard error exceeds {_THRESHOLDS['joint_offset_std_degrees']} degrees"
            " or is unavailable",
        )

    supplied_covariance = uncertainty.get("covariance")
    covariance = _covariance(supplied_covariance, errors) if finite_errors else None
    valid_covariance = supplied_covariance is None or covariance is not None
    _record(
        checks,
        failures,
        "covariance",
        valid_covariance,
        "Supplied covariance must be finite, symmetric, PSD and match the standard errors",
    )
    gripper_error, gripper_method = None, "unavailable"
    if finite_errors and valid_support and valid_covariance:
        gripper_error, gripper_method = _gripper_error(names, support, errors, covariance)
    _record(
        checks,
        failures,
        "gripper_uncertainty",
        gripper_error is not None and gripper_error <= _THRESHOLDS["gripper_angle_std_degrees"],
        f"Gripper standard error exceeds {_THRESHOLDS['gripper_angle_std_degrees']} degrees"
        " over full support or is unavailable",
    )
    return {
        "status": "unverified" if failures else "verified",
        "failures": failures,
        "checks": checks,
        "thresholds": dict(_THRESHOLDS),
        "mapped_support_rad": mapped,
        "offset_standard_errors_degrees": degrees,
        "gripper_standard_error_bound_degrees": gripper_error,
        "gripper_uncertainty_method": gripper_method,
        "scope": "Observed state/action support only; pan zero is a gauge, not a measured offset",
    }
