"""Position planning for the standard SO101 new-calibration geometry.

Transforms are extracted from the bundled, pinned TheRobotStudio URDF. Inputs
are calibrated LeRobot degrees; the moving jaw is deliberately excluded.
The fixed gripper_frame_link is expressed in base_link metres. This is model
kinematics, not camera calibration, collision planning or contact verification.
"""

import math

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
MODEL = "so101-new-calib-3a65d2d35e68a8d2"
FRAME = "base_link"
TOLERANCE_M = 0.003
MAX_STEP_M = 0.02
# xyz, fixed rpy, lower/upper radians; axes are all local +Z.
CHAIN = (
    ((0.0388353, -8.97657e-9, 0.0624), (3.14159, 4.18253e-17, -3.14159), (-1.91986, 1.91986)),
    ((-0.0303992, -0.0182778, -0.0542), (-1.5708, -1.5708, 0.0), (-1.74533, 1.74533)),
    ((-0.11257, -0.028, 1.73763e-16), (-3.63608e-16, 8.74301e-16, 1.5708), (-1.69, 1.69)),
    ((-0.1349, 0.0052, 3.62355e-17), (4.02456e-15, 8.67362e-16, -1.5708), (-1.65806, 1.65806)),
    ((5.55112e-17, -0.0611, 0.0181), (1.5708, 0.0486795, 3.14159), (-2.74385, 2.84121)),
)
TIP = (-0.0079, -0.000218121, -0.0981274)
IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def multiply(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3)
    )


def vector(r, v):
    return tuple(sum(r[i][k] * v[k] for k in range(3)) for i in range(3))


def rotation(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (
        math.cos(r),
        math.sin(r),
        math.cos(p),
        math.sin(p),
        math.cos(y),
        math.sin(y),
    )
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


ORIGINS = tuple((xyz, rotation(rpy), limits) for xyz, rpy, limits in CHAIN)


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def difference(a, b):
    return tuple(x - y for x, y in zip(a, b, strict=True))


def forward_radians(q, *, jacobian=False):
    r, position, origins, axes = IDENTITY, (0.0, 0.0, 0.0), [], []
    for angle, (xyz, fixed, _) in zip(q, ORIGINS, strict=True):
        position = tuple(a + b for a, b in zip(position, vector(r, xyz), strict=True))
        r = multiply(r, fixed)
        origins.append(position)
        axes.append(vector(r, (0.0, 0.0, 1.0)))
        r = multiply(r, rotation((0.0, 0.0, angle)))
    position = tuple(a + b for a, b in zip(position, vector(r, TIP), strict=True))
    if not jacobian:
        return position
    columns = []
    for origin, axis in zip(origins, axes, strict=True):
        x, y, z = difference(position, origin)
        a, b, c = axis
        columns.append((b * z - c * y, c * x - a * z, a * y - b * x))
    return position, columns


def forward(joints):
    q = [math.radians(joints[j]) for j in JOINTS]
    if any(not math.isfinite(x) for x in q):
        raise ValueError("Gripper position needs finite calibrated joint angles")
    return list(forward_radians(q))


def linear_solve(matrix, values):
    rows = [list(row) + [value] for row, value in zip(matrix, values, strict=True)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda row: abs(rows[row][col]))
        rows[col], rows[pivot] = rows[pivot], rows[col]
        scale = rows[col][col]
        if abs(scale) < 1e-15:
            raise ValueError("Kinematics is singular at this pose")
        rows[col] = [x / scale for x in rows[col]]
        for row in range(3):
            if row != col:
                factor = rows[row][col]
                rows[row] = [x - factor * y for x, y in zip(rows[row], rows[col], strict=True)]
    return [row[3] for row in rows]


def bounds_for(limits):
    result = []
    for joint, (_, _, bounds) in zip(JOINTS, CHAIN, strict=True):
        low, high = limits[joint]
        if any(type(v) not in {int, float} or not math.isfinite(v) for v in (low, high)):
            raise ValueError("Gripper position needs finite calibration limits")
        low, high = max(bounds[0], math.radians(low)), min(bounds[1], math.radians(high))
        if low >= high:
            raise ValueError("Calibration and standard geometry have no common joint range")
        result.append((low, high))
    return result


def solve(target, measured, limits):
    if len(target) != 3 or any(type(v) not in {int, float} or not math.isfinite(v) for v in target):
        raise ValueError("A gripper target needs three finite coordinates in metres")
    if not -0.6 <= target[0] <= 0.6 or not -0.6 <= target[1] <= 0.6 or not 0 <= target[2] <= 0.6:
        raise ValueError(
            "Gripper target is outside the base-frame workspace (z must be at least zero)"
        )
    bounds = bounds_for(limits)
    current = [math.radians(measured[j]) for j in JOINTS]
    if any(
        not math.isfinite(v) or not lo <= v <= hi
        for v, (lo, hi) in zip(current, bounds, strict=True)
    ):
        raise ValueError("Current joint pose is outside the standard SO101 geometry limits")
    if forward_radians(current)[2] < 0:
        raise ValueError(
            "Current gripper estimate is below the base plane; check geometry and calibration"
        )
    seeds = [current, [0.0] * 5, [0.0, -0.6, 0.9, 0.6, 0.0], [0.0, 0.6, -0.9, -0.6, 0.0]]
    solutions = []
    for seed in seeds:
        q = [min(hi, max(lo, v)) for v, (lo, hi) in zip(seed, bounds, strict=True)]
        for _ in range(160):
            position, columns = forward_radians(q, jacobian=True)
            error = difference(target, position)
            if norm(error) <= 0.0005:
                # Reject a joint-space route that would cross the model's base plane.
                if all(
                    forward_radians(
                        [a + (b - a) * k / 32 for a, b in zip(current, q, strict=True)]
                    )[2]
                    >= 0
                    for k in range(33)
                ):
                    solutions.append(q)
                break
            matrix = [
                [
                    sum(c[i] * c[j] for c in columns) + (0.000025 if i == j else 0.0)
                    for j in range(3)
                ]
                for i in range(3)
            ]
            dual = linear_solve(matrix, error)
            delta = [sum(c[i] * dual[i] for i in range(3)) for c in columns]
            scale = min(1.0, 0.12 / max(max(abs(x) for x in delta), 1e-12))
            improved = False
            for _ in range(12):
                trial = [
                    min(hi, max(lo, a + scale * d))
                    for a, d, (lo, hi) in zip(q, delta, bounds, strict=True)
                ]
                if norm(difference(target, forward_radians(trial))) < norm(error) - 1e-10:
                    q, improved = trial, True
                    break
                scale *= 0.5
            if not improved:
                break
    if not solutions:
        raise ValueError(
            "No reachable gripper position within the calibrated joint limits and base-plane bounds"
        )
    return min(solutions, key=lambda q: sum((a - b) ** 2 for a, b in zip(q, current, strict=True)))


def waypoint(target, measured, limits, max_delta, duration):
    current = [math.radians(measured[j]) for j in JOINTS]
    initial = forward_radians(current)
    initial_error = norm(difference(target, initial))
    if initial_error <= TOLERANCE_M:
        raise ValueError(
            "Gripper estimate is already within 3 mm of the target; assess the result instead"
        )
    solved = solve(target, measured, limits)
    delta = difference(solved, current)
    alpha = 1.0
    for j, change in zip(JOINTS, delta, strict=True):
        allowed = min(max_delta[j], 10 * duration / 1.875, 20 * duration**2 / 5.774) * 0.9
        if abs(change) > 1e-12:
            alpha = min(alpha, math.radians(allowed) / abs(change))
    for _ in range(24):
        next_q = [a + alpha * d for a, d in zip(current, delta, strict=True)]
        next_position = forward_radians(next_q)
        remaining = norm(difference(target, next_position))
        if (
            norm(difference(next_position, initial)) <= MAX_STEP_M
            and next_position[2] >= 0
            and remaining < initial_error - 1e-7
        ):
            return {
                "targets": {j: math.degrees(q) for j, q in zip(JOINTS, next_q, strict=True)},
                "duration_seconds": duration,
            }, {
                "model": MODEL,
                "frame": FRAME,
                "units": "metres",
                "target_m": list(target),
                "from_m": list(initial),
                "next_m": list(next_position),
                "remaining_m": remaining,
                "tolerance_m": TOLERANCE_M,
                "final_waypoint": alpha >= 1.0 - 1e-9,
                "orientation_controlled": False,
                "evidence": "model_estimate",
            }
        alpha *= 0.5
    raise ValueError("No bounded progress to this target; choose an intermediate target position")
