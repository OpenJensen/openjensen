"""Model geometry and bounded planning; no robot connection or motion."""

import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from vla_platform.agent_control import kinematics as k

LIMITS = {j: [-90.0, 90.0] for j in k.JOINTS}
DELTAS = dict.fromkeys(k.JOINTS, 4.0)
ZERO = dict.fromkeys(k.JOINTS, 0.0)


def test_geometry_matches_the_bundled_pinned_urdf():
    root = Path(__file__).resolve().parents[1]
    model = ET.parse(root / "apps/web/public/robots/so101/so101_new_calib.urdf")
    for name, (xyz, rpy, bounds) in zip(k.JOINTS, k.CHAIN, strict=True):
        joint = model.find(f".//joint[@name='{name}']")
        assert tuple(map(float, joint.find("origin").get("xyz").split())) == xyz
        assert tuple(map(float, joint.find("origin").get("rpy").split())) == rpy
        assert joint.find("axis").get("xyz") == "0 0 1"
        assert tuple(float(joint.find("limit").get(key)) for key in ("lower", "upper")) == bounds
    tip = model.find(".//joint[@name='gripper_frame_joint']")
    assert tuple(map(float, tip.find("origin").get("xyz").split())) == k.TIP


def test_jacobian_matches_independent_position_differences():
    q = [0.2, -0.3, 0.4, 0.1, -0.2]
    position, columns = k.forward_radians(q, jacobian=True)
    for j, column in enumerate(columns):
        shifted = list(q)
        shifted[j] += 1e-6
        numerical = [
            (a - b) / 1e-6 for a, b in zip(k.forward_radians(shifted), position, strict=True)
        ]
        assert numerical == pytest.approx(column, abs=3e-7)


def test_repeated_reviewed_waypoints_reach_a_metric_goal_without_moving_the_gripper():
    measured = dict(ZERO, gripper=43.0)
    target = k.forward(dict(ZERO, shoulder_lift=20.0, elbow_flex=15.0))
    errors = []
    for _ in range(60):
        error = math.dist(k.forward(measured), target)
        errors.append(error)
        if error <= k.TOLERANCE_M:
            break
        segment, plan = k.waypoint(target, measured, LIMITS, DELTAS, 2.0)
        assert set(segment["targets"]) == set(k.JOINTS)
        assert "gripper" not in segment["targets"]
        assert math.dist(plan["from_m"], plan["next_m"]) <= k.MAX_STEP_M
        assert plan["remaining_m"] < error
        for j, value in segment["targets"].items():
            assert abs(value - measured[j]) <= 4.0
            assert LIMITS[j][0] <= value <= LIMITS[j][1]
        measured.update(segment["targets"])
        assert measured["gripper"] == 43.0
    assert errors[-1] <= k.TOLERANCE_M
    assert len(errors) > 1
    assert all(b < a for a, b in zip(errors, errors[1:]))


def test_reachable_targets_round_trip_across_normal_arm_configurations():
    rng = random.Random(101)
    checked = 0
    for _ in range(24):
        goal = {j: rng.uniform(-35, 35) for j in k.JOINTS}
        target = k.forward(goal)
        if target[2] < 0:
            continue
        solved = k.solve(target, ZERO, LIMITS)
        assert math.dist(k.forward_radians(solved), target) <= 0.0005
        checked += 1
    assert checked >= 12


@pytest.mark.parametrize(
    "target", [[0.6, 0.6, 0.6], [0, 0, -0.01], [float("nan"), 0, 0], [float("inf"), 0, 0]]
)
def test_unreachable_or_invalid_targets_never_produce_a_segment(target):
    with pytest.raises(ValueError):
        k.waypoint(target, ZERO, LIMITS, DELTAS, 2.0)


def test_joint_limits_and_short_duration_are_respected():
    target = k.forward(dict(ZERO, shoulder_lift=4.0))
    segment, _ = k.waypoint(target, ZERO, LIMITS, DELTAS, 0.5)
    for j, value in segment["targets"].items():
        delta = abs(value - ZERO[j])
        assert 1.875 * delta / 0.5 <= 10
        assert 5.774 * delta / 0.5**2 <= 20
    tight = {j: [-0.01, 0.01] for j in k.JOINTS}
    with pytest.raises(ValueError):
        k.solve(target, ZERO, tight)
