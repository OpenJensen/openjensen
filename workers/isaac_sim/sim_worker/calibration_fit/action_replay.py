"""Teacher-forced recorded-action audit; no simulator or policy inference."""

import argparse
import hashlib
import inspect
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from sim_worker.rollout.calibration import CalibrationUse, JointMap
from sim_worker.rollout.config import number
from sim_worker.rollout.experimental import MAX_SPEED_RAD_S, MotionGuard

_TIME_TOLERANCE_S = 1e-6
_LAG_FRAMES = tuple(range(-5, 6))
_SCHEMA = "simulation.recorded-action-audit/v1alpha1"


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _vector(value, joints, name):
    if not isinstance(value, (list, tuple)) or len(value) != len(joints):
        raise ValueError(f"{name} must contain one value per joint")
    return tuple(number(item, name) for item in value)


def _groups(rows, fps):
    groups, seen, previous_index = [], set(), None
    for row in rows:
        episode = _integer(row["episode_index"], "episode_index")
        frame = _integer(row["frame_index"], "frame_index")
        index = _integer(row["index"], "index")
        timestamp = number(row["timestamp"], "timestamp")
        if previous_index is not None and index <= previous_index:
            raise ValueError("Source indices must increase in recorded order")
        if not groups or groups[-1][0]["episode_index"] != episode:
            if episode in seen or frame != 0:
                raise ValueError("Each episode must start once at frame zero")
            groups.append([])
            seen.add(episode)
        group = groups[-1]
        if frame != len(group) or (group and index != previous_index + 1):
            raise ValueError("Frames and source indices must be contiguous within an episode")
        if abs(timestamp - frame / fps) > _TIME_TOLERANCE_S:
            raise ValueError("Timestamp must reset and follow the recorded control rate")
        group.append(row)
        previous_index = index
    if not groups:
        raise ValueError("Recorded actions are empty")
    return groups


def _stats(values):
    if not values:
        return {"count": 0, "mean_abs_rad": None, "rms_rad": None, "max_abs_rad": None}
    return {
        "count": len(values),
        "mean_abs_rad": sum(map(abs, values)) / len(values),
        "rms_rad": math.sqrt(sum(value * value for value in values) / len(values)),
        "max_abs_rad": max(map(abs, values)),
    }


def _difference(left, right):
    return tuple(a - b for a, b in zip(left, right, strict=True))


def _trace(group, mapping, guard, joints, limits, fps):
    states = [mapping.to_sim(_vector(row["observation.state"], joints, "state")) for row in group]
    trace = []
    for row, state in zip(group, states, strict=True):
        frame = row["frame_index"]
        action = _vector(row["action"], joints, "action")
        requested = mapping.to_sim(action)
        decision = guard.constrain(requested, state)
        next_state = states[frame + 1] if frame + 1 < len(states) else None
        trace.append(
            {
                "episode_index": row["episode_index"],
                "frame_index": frame,
                "source_index": row["index"],
                "episode_reset": frame == 0,
                "recorded_timestamp_s": row["timestamp"],
                "control_timestamp_s": frame / fps,
                "command_end_timestamp_s": (frame + 1) / fps,
                "timestamp_error_s": row["timestamp"] - frame / fps,
                "policy_action": action,
                "recorded_policy_state": tuple(row["observation.state"]),
                "recorded_state_rad": state,
                "raw_target_rad": requested,
                "guarded_target_rad": decision.target,
                "action_extrapolated": mapping.action_outside(action),
                "state_outside_limits": tuple(
                    not lower <= value <= upper
                    for value, (lower, upper) in zip(state, limits, strict=True)
                ),
                "joint_limit_clipped": decision.limit_clipped,
                "guard_offset_clipped": decision.speed_clipped,
                "guard_change_rad": _difference(decision.target, requested),
                "requested_minus_state_rad": _difference(requested, state),
                "guarded_minus_state_rad": _difference(decision.target, state),
                "recorded_next_state_rad": next_state,
                "requested_minus_next_state_rad": (
                    _difference(requested, next_state) if next_state is not None else None
                ),
            }
        )
    return trace


def _lag_profile(groups, joints, fps):
    profile = []
    for lag in _LAG_FRAMES:
        errors = []
        for group in groups:
            for frame, row in enumerate(group):
                shifted = frame + lag
                if not 0 <= shifted < len(group):
                    continue
                errors.append(
                    _difference(row["raw_target_rad"], group[shifted]["recorded_state_rad"])
                )
        per_joint = {
            name: _stats([error[index] for error in errors]) for index, name in enumerate(joints)
        }
        profile.append(
            {
                "state_offset_frames": lag,
                "state_offset_s": lag / fps,
                "pairs": len(errors),
                "per_joint": per_joint,
            }
        )
    best = {}
    for name in joints:
        supported = [item for item in profile if item["pairs"]]
        winner = min(supported, key=lambda item: item["per_joint"][name]["rms_rad"])
        best[name] = {
            "state_offset_frames": winner["state_offset_frames"],
            "rms_rad": winner["per_joint"][name]["rms_rad"],
            "at_search_boundary": winner["state_offset_frames"]
            in (_LAG_FRAMES[0], _LAG_FRAMES[-1]),
        }
    return {
        "definition": "Compare action[t] with recorded state[t + offset]; never cross episodes",
        "limitation": (
            "Descriptive alignment only; offsets, controller lag and sample support "
            "confound latency"
        ),
        "profile": profile,
        "minimum_rms_by_joint": best,
    }


def _summary(groups, joints, fps):
    rows = [row for group in groups for row in group]
    flags = (
        "action_extrapolated",
        "state_outside_limits",
        "joint_limit_clipped",
        "guard_offset_clipped",
    )
    metrics = (
        "guard_change_rad",
        "requested_minus_state_rad",
        "guarded_minus_state_rad",
        "requested_minus_next_state_rad",
    )
    per_joint = {}
    for index, name in enumerate(joints):
        per_joint[name] = {flag: sum(row[flag][index] for row in rows) for flag in flags}
        per_joint[name].update(
            {
                metric: _stats([row[metric][index] for row in rows if row[metric] is not None])
                for metric in metrics
            }
        )
    return {
        "frames": len(rows),
        "episodes": len(groups),
        "counts": {flag: sum(sum(row[flag]) for row in rows) for flag in flags},
        "frames_with_limit_clip": sum(any(row["joint_limit_clipped"]) for row in rows),
        "frames_with_guard_clip": sum(any(row["guard_offset_clipped"]) for row in rows),
        "max_timestamp_error_s": max(abs(row["timestamp_error_s"]) for row in rows),
        "unpaired_final_actions": len(groups),
        "per_joint": per_joint,
        "action_state_alignment": _lag_profile(groups, joints, fps),
    }


def audit_actions(rows, mapping: JointMap, joints, limits, fps):
    """Return (report, trace) using each recorded state as the guard's measured input.

    Trace targets are diagnostic, not a feed-forward command schedule. A later
    Isaac replay must reapply MotionGuard against the simulator's live state.
    """
    joints = tuple(joints)
    if not joints or len(joints) != len(set(joints)):
        raise ValueError("Joints must be nonempty and unique")
    limits = tuple(_vector(pair, ("lower", "upper"), "limits") for pair in limits)
    if len(limits) != len(joints) or any(lower > upper for lower, upper in limits):
        raise ValueError("Limits must match joint order and be increasing")
    guard = MotionGuard(lambda: limits, fps)
    groups = [_trace(group, mapping, guard, joints, limits, fps) for group in _groups(rows, fps)]
    report = {
        "api_version": _SCHEMA,
        "mode": "offline_teacher_forced",
        "calibration_status": mapping.status,
        "calibration_sha256": mapping.digest,
        "joints": joints,
        "limits_rad": limits,
        "fps": fps,
        "guard_max_offset_rad": MAX_SPEED_RAD_S / fps,
        "state_source": "observation.state from the same recorded row as action",
        "action_source": "action from the recorded dataset; never predicted or substituted state",
        "limitations": [
            "No simulation, actuator tracking, collision, grasp or closed-loop policy validation",
            "Every guard call uses the current recorded state, not the previous guarded target",
            "Recorded states can map outside limits; the guard may exceed its nominal offset "
            "to restore bounds",
            "Final action has no next recorded state; it is not compared to the next episode",
            "Dataset timestamps are shared row timestamps, not separate sensor and command clocks",
        ],
        "aggregate": _summary(groups, joints, fps),
        "episodes": {
            str(group[0]["episode_index"]): _summary([group], joints, fps) for group in groups
        },
    }
    return report, [row for group in groups for row in group]


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recorded_source(dataset, provenance):
    import pyarrow.parquet as pq

    metadata = json.loads(provenance.read_text())
    sources = [item for item in metadata["source_files"] if item["path"].startswith("data/")]
    if len(sources) != 1 or _digest(dataset) != sources[0]["sha256"]:
        raise ValueError("Dataset must match the single pinned data Parquet SHA256")
    info = metadata["confirmed_metadata"]
    joints = tuple(name.removesuffix(".pos") for name in info["joint_order"])
    rows = pq.read_table(
        dataset,
        columns=[
            "action",
            "observation.state",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
        ],
    ).to_pylist()
    if len(rows) != info["frames"]:
        raise ValueError("Dataset frame count differs from pinned provenance")
    offsets = {item["episode"]: item for item in metadata["episodes"]}
    grouped = _groups(rows, info["fps"])
    if len(grouped) != len(offsets):
        raise ValueError("Episode count differs from pinned provenance")
    for group in grouped:
        declared = offsets[group[0]["episode_index"]]
        expected = declared["data_rows_half_open"]
        if (
            len(group) != declared["frames"]
            or [group[0]["index"], group[-1]["index"] + 1] != expected
        ):
            raise ValueError("Episode boundaries differ from pinned provenance")
    return (
        rows,
        joints,
        info["fps"],
        {
            "dataset": metadata["dataset"],
            "recorded_file": sources[0],
            "local_dataset": str(dataset),
            "provenance_sha256": _digest(provenance),
            "episode_boundaries_verified": True,
        },
    )


def _scene_limits(path, joints):
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError("Cannot open the composed USD scene")
    found = {}
    for prim in stage.Traverse():
        name = prim.GetName()
        if name not in joints or not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        if name in found:
            raise ValueError(f"Ambiguous scene joint: {name}")
        joint = UsdPhysics.RevoluteJoint(prim)
        found[name] = tuple(
            math.radians(number(value, name))
            for value in (
                joint.GetLowerLimitAttr().Get(),
                joint.GetUpperLimitAttr().Get(),
            )
        )
    if set(found) != set(joints):
        raise ValueError("Scene does not contain every requested revolute joint")
    return tuple(found[name] for name in joints), {
        "kind": "composed_usd_revolute_limits",
        "path": str(path),
        "sha256": _digest(path),
        "composed_sha256": hashlib.sha256(stage.Flatten().ExportToString().encode()).hexdigest(),
    }


def _urdf_limits(path, joints):
    found = {}
    for joint in ET.parse(path).getroot().findall("joint"):
        name = joint.get("name")
        if name not in joints:
            continue
        if name in found or joint.get("type") != "revolute":
            raise ValueError(f"Ambiguous or unsupported URDF joint: {name}")
        limit = joint.find("limit")
        if limit is None:
            raise ValueError(f"URDF joint has no limits: {name}")
        found[name] = tuple(float(limit.attrib[key]) for key in ("lower", "upper"))
    if set(found) != set(joints):
        raise ValueError("URDF does not contain every requested revolute joint")
    return tuple(found[name] for name in joints), {
        "kind": "urdf_revolute_limits",
        "path": str(path),
        "sha256": _digest(path),
        "limitation": "URDF limits may differ from the composed simulation scene",
    }


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scene", type=Path)
    source.add_argument("--urdf", type=Path)
    parser.add_argument("--episode", type=int, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, joints, fps, provenance = _recorded_source(args.dataset, args.provenance)
    if args.episode:
        requested = set(args.episode)
        rows = [row for row in rows if row["episode_index"] in requested]
        if requested != {row["episode_index"] for row in rows}:
            raise ValueError("Requested episode is absent from the dataset")
    limits, origin = (
        _scene_limits(args.scene, joints) if args.scene else _urdf_limits(args.urdf, joints)
    )
    mapping = JointMap(args.calibration, joints, CalibrationUse.EXPERIMENTAL)
    report, trace = audit_actions(rows, mapping, joints, limits, fps)
    report["provenance"] = provenance
    report["limit_source"] = origin
    report["calibration_path"] = str(args.calibration)
    report["implementation_sha256"] = {
        "audit": _digest(Path(__file__)),
        "joint_map": _digest(Path(inspect.getfile(JointMap))),
        "motion_guard": _digest(Path(inspect.getfile(MotionGuard))),
    }
    report["future_replay"] = {
        "execution": "not_started",
        "action_field": "policy_action",
        "reset": "Reset per episode; review limits before initializing its recorded first state",
        "step": "Map the recorded action, guard against live Isaac state, then step once",
        "end": "Stop at episode end; never hold or repeat the last action to invent missing frames",
        "warning": "Do not feed guarded_target_rad as a fixed schedule; it used recorded states",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    trace_path = args.output / "command-trace.jsonl"
    trace_path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in trace))
    report["trace_sha256"] = _digest(trace_path)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "mode": report["mode"],
                "frames": len(trace),
                "counts": report["aggregate"]["counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    _main()
