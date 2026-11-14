"""Build two matched offline views of a sealed collection without modifying it."""

import argparse
import bisect
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

PIN = "d65f94cc0155d8908fd061a47b71b7370279be3f"
VERSION = "openjensen-isr/2.0"
MOTION_EPSILON = 0.15  # Encoder degrees or gripper percentage points over the interval.
MAX_TICK_GAP_NS = 100_000_000


def rows(bundle, name):
    return [json.loads(line) for line in bundle.read("evidence/" + name + ".jsonl").splitlines()]


def checked_episode(bundle):
    names = bundle.namelist()
    metadata = json.loads(bundle.read("evidence/recording.json"))
    if metadata["schema_version"] != 2 or "evidence/seal.json" not in names:
        raise ValueError("A sealed schema v2 episode is required; preserve older episodes as raw")
    hashes = json.loads(bundle.read("evidence/seal.json"))["sha256"]
    if len(set(names)) != len(names) or set(names) - {
        "openjensen/source.json",
        "openjensen/annotations.json",
    } != set(hashes) | {"evidence/seal.json"}:
        raise ValueError("Raw member inventory differs from seal")
    for name, expected in hashes.items():
        with bundle.open(name) as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != expected:
                raise ValueError("Raw checksum mismatch: " + name)
    return metadata


class Clocks:
    def __init__(self, trace):
        self.probes = sorted(
            (e["core_before_ns"], e["core_after_ns"], e["robot_ns"])
            for e in trace
            if e["type"] == "clock_probe"
        )
        self.worker_probes = [
            (int(e["worker_before_ns"]), int(e["worker_after_ns"]), e["core_ns"])
            for e in trace
            if e["type"] == "worker_clock_bound"
        ]
        if not self.worker_probes:
            raise ValueError("Worker clock calibration is unavailable")
        if len(self.probes) < 1:
            raise ValueError("Cannot align clocks")

    def robot(self, event):
        t = event["core_monotonic_ns"]
        error = 0
        if "worker_ns" in event:
            stamp = int(event["worker_ns"])
            left, right, core = min(
                self.worker_probes,
                key=lambda v: (v[1] - v[0]) / 2 + abs(stamp - (v[0] + v[1]) / 2) * 0.001,
            )
            t = stamp + core - (left + right) // 2
            error = (right - left + 1) // 2
        # Two-way probe midpoint, no symmetric transport assumption. Bound both endpoints
        # using neighboring probes; the range includes measured offset variation.
        i = bisect.bisect_left(self.probes, (t,))
        probes = self.probes[max(0, i - 1) : min(len(self.probes), i + 1)]
        lo = min(r - b for a, b, r in probes)
        hi = max(r - a for a, b, r in probes)
        return t + (lo + hi) // 2, error + (hi - lo + 1) // 2

    def report(self):
        a, b, r = self.probes[0]
        c, d, s = self.probes[-1]
        elapsed = (c + d - a - b) / 2
        left, right, k = self.worker_probes[0]
        u, v, last_core = self.worker_probes[-1]
        worker_elapsed = (u + v - left - right) / 2
        return {
            "core_probes": self.probes,
            "worker_probes": self.worker_probes,
            "worker_to_core_offset_ns": k - (left + right) // 2,
            "worker_core_uncertainty_ns": (right - left + 1) // 2,
            "worker_core_empirical_drift_ppm": ((last_core - k) / worker_elapsed - 1) * 1e6
            if worker_elapsed
            else None,
            "core_robot_empirical_drift_ppm": ((s - r) / elapsed - 1) * 1e6 if elapsed else None,
            "method": "bracketing host RTT bounds; hardware oscillator drift bounds unavailable",
        }


def intervals(events, clocks):
    starts, result = {}, []
    kinds = {
        "model_request": ("api", "start"),
        "model_response": ("api", "end"),
        "injected_delay_start": ("delay", "start"),
        "injected_delay_end": ("delay", "end"),
        "dwell_start": ("dwell", "start"),
        "dwell_end": ("dwell", "end"),
    }
    for e in events:
        if e["type"] not in kinds:
            continue
        kind, edge = kinds[e["type"]]
        key = kind, e.get("request_id", e.get("tool_call_id"))
        t, u = clocks.robot(e)
        if edge == "start":
            starts[key] = (t, u, e)
        elif key in starts:
            a, v, first = starts.pop(key)
            result.append(
                {
                    "kind": kind,
                    "id": key[1],
                    "event_ids": [first["event_id"], e["event_id"]],
                    "start_robot_ns": a,
                    "end_robot_ns": t,
                    "start_uncertainty_ns": v,
                    "end_uncertainty_ns": u,
                    "reason": first.get("reason"),
                    "planned_seconds": first.get("duration_seconds"),
                    "elapsed_ns": int(e["worker_ns"]) - int(first["worker_ns"])
                    if "worker_ns" in e
                    else t - a,
                }
            )
    for (kind, ident), (a, u, _) in starts.items():
        result.append(
            {
                "kind": kind,
                "id": ident,
                "start_robot_ns": a,
                "start_uncertainty_ns": u,
                "end_robot_ns": None,
            }
        )
    for span in result:
        if span["kind"] == "api" and span["end_robot_ns"] is not None:
            injected = sum(
                x["elapsed_ns"]
                for x in result
                if x["kind"] == "delay" and x["id"] == span["id"] and x["end_robot_ns"] is not None
            )
            span.update(
                injected_ns=injected, elapsed_excluding_injection_ns=span["elapsed_ns"] - injected
            )
    return result


def join(metadata, frames, control, spans, annotations):
    from vla_platform.agent_control import kinematics

    if (
        metadata["kinematics"]["model"] != kinematics.MODEL
        or metadata["kinematics"]["chain"] != json.loads(json.dumps(kinematics.CHAIN))
        or metadata["kinematics"]["tip_offset_m"] != list(kinematics.TIP)
    ):
        raise ValueError("Unknown FK model")
    ticks = {s["sample_id"]: s for s in control}
    dispatch = [s["dispatch_begin_monotonic_ns"] for s in control]
    if any(y <= x for x, y in zip(dispatch, dispatch[1:])):
        raise ValueError("Non-monotonic control journal")
    samples = []
    if not dispatch:
        raise ValueError("No actual writes recorded")
    previous = None
    segment = 0
    for f in frames:
        idx = f["frame_index"]
        t = f["pair_available_robot_ns"]
        if f["pair_skew_ns"] > 150_000_000 or any(
            c["age_ns"] > 500_000_000 for c in f["cameras"].values()
        ):
            previous = None
            continue
        i = bisect.bisect_left(dispatch, t)
        measured = ticks[f["controller_sample_id"]]
        # Images and measured input must all exist before the chosen *next* write.
        if i == len(dispatch) or dispatch[i] - t > MAX_TICK_GAP_NS:
            previous = None
            continue
        act = control[i]
        j = bisect.bisect_left(dispatch, previous["available_robot_ns"]) if previous else i
        gap = (
            previous is None
            or idx != previous["raw_index"] + 1
            or (
                f["missed_schedule_slots_before"] != previous["missed_slots"]
                or t - previous["available_robot_ns"] > 1.75e9 / metadata["run_parameters"]["fps"]
                or any(
                    b - a > MAX_TICK_GAP_NS for a, b in zip(dispatch[j:i], dispatch[j + 1 : i + 1])
                )
            )
        )
        invalid = any(
            e["label"] in ("intervention", "reset", "failure")
            and e["first_frame"] <= idx <= e["last_frame"]
            for e in annotations
        )
        if any(
            span["end_robot_ns"] is None
            and t >= span["start_robot_ns"] - span["start_uncertainty_ns"]
            for span in spans
        ):
            invalid = True
        if invalid:
            previous = None
            continue
        if gap:
            segment += 1
        labels = []
        for span in spans:
            end = span["end_robot_ns"]
            if end is None:
                if t >= span["start_robot_ns"] - span["start_uncertainty_ns"]:
                    labels.append(["ambiguous", span["id"]])
                continue
            if (
                not span["start_robot_ns"] - span["start_uncertainty_ns"]
                <= t
                <= end + span["end_uncertainty_ns"]
            ):
                continue
            sure_start = span["start_robot_ns"] + span["start_uncertainty_ns"]
            sure_end = end - span["end_uncertainty_ns"]
            if t < sure_start or t > sure_end:
                labels.append(["ambiguous", span["id"]])
                continue
            if span["kind"] == "dwell":
                label = (
                    "protected_dwell"
                    if span["reason"] in ("grasp_settling", "release_settling")
                    and act["controller_state"] == "holding"
                    else "ambiguous"
                )
                labels.append([label, span["id"]])
            elif span["kind"] == "api":
                a = max(bisect.bisect_left(dispatch, max(sure_start, t - 100_000_000)) - 1, 0)
                near = control[a : i + 1]
                motion = any(
                    max(x["measured"][joint] for x in near)
                    - min(x["measured"][joint] for x in near)
                    > MOTION_EPSILON
                    for joint in metadata["joint_order"]
                )
                active = any(x["controller_state"] == "executing" for x in near)
                continuous = len(near) >= 3 and all(
                    b - a <= MAX_TICK_GAP_NS for a, b in zip(dispatch[a:i], dispatch[a + 1 : i + 1])
                )
                label = (
                    "api_motion"
                    if motion
                    else "stable_wait"
                    if continuous and not active
                    else "ambiguous"
                )
                labels.append([label, span["id"]])
        images = {
            role: {
                "id": c["frame_id"],
                "path": f"raw/observation.images.{role}/{idx:06d}.png",
                "read_complete_robot_ns": c["read_complete_robot_ns"],
                "available_robot_ns": c["available_robot_ns"],
                "delta_to_dispatch_ns": dispatch[i] - c["available_robot_ns"],
            }
            for role, c in f["cameras"].items()
        }
        if (
            set(images) != {"wrist", "side"}
            or min(x["delta_to_dispatch_ns"] for x in images.values()) < 0
        ):
            raise ValueError("Invalid paired observation/action join")
        state = measured["measured"]
        row = {
            "raw_index": idx,
            "segment": segment,
            "images": images,
            "available_robot_ns": t,
            "original_elapsed_ns": t - metadata["origin_monotonic_ns"],
            "state_sample_id": measured["sample_id"],
            "state_robot_ns": measured["state_read_monotonic_ns"],
            "action_sample_id": act["sample_id"],
            "command_id": act["command_id"],
            "action_robot_ns": dispatch[i],
            "write_returned_ns": act["write_returned_monotonic_ns"],
            "action_delta_ns": dispatch[i] - t,
            "sent": act["sent"],
            "sent_raw": act["sent_raw"],
            "state": state,
            "xyz_m": kinematics.forward(state),
            "gripper_fraction": state["gripper"] / 100,
            "missed_slots": f["missed_schedule_slots_before"],
            "masks": labels,
        }
        samples.append(row)
        previous = row
    return samples


def selections(samples, fps, d_target, budget):
    from itertools import groupby

    import numpy as np
    from isr.resample_trajectory import isr_resample_trajectory

    indices, protected, ordinary = set(), set(), []
    for _, group in groupby(range(len(samples)), key=lambda i: samples[i]["segment"]):
        group = list(group)
        chosen = isr_resample_trajectory(
            [samples[i]["xyz_m"] for i in group],
            times=np.array([samples[i]["original_elapsed_ns"] / 1e9 for i in group]),
            gripper=np.array([samples[i]["gripper_fraction"] for i in group]),
            d_target=d_target,
        )
        indices.update(group[j] for j in chosen)
        ordinary.extend(group)
    protected = {i for i in ordinary if any(k == "protected_dwell" for k, _ in samples[i]["masks"])}
    required = indices | protected
    size = len(required) if budget is None else budget
    if not len(required) <= size <= len(ordinary) or not ordinary:
        raise ValueError(
            f"Requested retained budget must be between {len(required)} and {len(ordinary)}"
        )

    def fill(selected, allow):
        selected = set(selected)
        # Deterministic time coverage within each continuous segment. Preserve every upstream guard.
        candidates = set(allow) - selected
        by_segment = {}
        for j in sorted(selected):
            by_segment.setdefault(samples[j]["segment"], []).append(
                samples[j]["original_elapsed_ns"]
            )

        def initial_distance(i):
            anchors = by_segment[samples[i]["segment"]]
            t = samples[i]["original_elapsed_ns"]
            j = bisect.bisect_left(anchors, t)
            return min(abs(t - a) for a in anchors[max(0, j - 1) : j + 1])

        distances = {i: initial_distance(i) for i in candidates}
        while len(selected) < size and distances:
            chosen = max(distances, key=lambda i: (distances[i], -i))
            selected.add(chosen)
            del distances[chosen]
            for i in distances:
                if samples[i]["segment"] == samples[chosen]["segment"]:
                    distances[i] = min(
                        distances[i],
                        abs(
                            samples[i]["original_elapsed_ns"]
                            - samples[chosen]["original_elapsed_ns"]
                        ),
                    )
        return selected

    blind = fill(indices, ordinary)
    preferred = [i for i in ordinary if not any(k == "stable_wait" for k, _ in samples[i]["masks"])]
    assisted = fill(fill(required, preferred), ordinary)

    def mapping(selected):
        return [
            {
                "raw_index": samples[i]["raw_index"],
                "segment": samples[i]["segment"],
                "derived_elapsed_ns": round(n * 1e9 / fps),
                "original_elapsed_ns": samples[i]["original_elapsed_ns"],
                "action_sample_id": samples[i]["action_sample_id"],
                "command_id": samples[i]["command_id"],
                "camera_ids": {k: v["id"] for k, v in samples[i]["images"].items()},
            }
            for n, i in enumerate(sorted(selected))
        ]

    return {"blind": mapping(blind), "assisted": mapping(assisted)}, len(indices)


def build(archive, upstream, output, budget=None, d_target=0.05, reviews=()):
    if (
        subprocess.check_output(["git", "-C", upstream, "rev-parse", "HEAD"], text=True).strip()
        != PIN
        or subprocess.check_output(
            ["git", "-C", upstream, "status", "--porcelain", "--", "src/isr"], text=True
        ).strip()
    ):
        raise ValueError("Use the specified clean upstream ISR revision")
    sys.path.insert(0, str(Path(upstream).resolve() / "src"))
    with zipfile.ZipFile(archive) as bundle:
        m = checked_episode(bundle)
        if m["status"] != "saved" and not reviews:
            raise ValueError(
                "Failed or partial capture requires evidence review before deriving views"
            )
        frames, control, events = [rows(bundle, n) for n in ("frames", "controller", "trace")]
        clocks = Clocks(events)
        spans = intervals(events, clocks)
        approvals = {e["command_id"]: e for e in events if e["type"] == "command_requested"}
        commands = []
        for name in bundle.namelist():
            if not name.startswith("evidence/executions/"):
                continue
            receipt = json.loads(bundle.read(name))
            if receipt["operation"] != "execute":
                continue
            cid = receipt["command_id"]
            writes = [s for s in control if s["command_id"] == cid]
            req = approvals.get(cid, {})
            commands.append(
                {
                    "command_id": cid,
                    "tool_call_id": req.get("tool_call_id"),
                    "model_observation_id": req.get("model_observation_id"),
                    "requested": req.get("requested"),
                    "accepted_ns": receipt["created_monotonic_ns"],
                    "dispatch_ack_ns": receipt.get("dispatch_ack_monotonic_ns"),
                    "admitted_start_ns": receipt.get("started_monotonic_ns"),
                    "first_dispatch_ns": writes[0]["dispatch_begin_monotonic_ns"]
                    if writes
                    else None,
                    "last_write_returned_ns": writes[-1]["write_returned_monotonic_ns"]
                    if writes
                    else None,
                    "encoder_target_reached_ns": receipt.get("completed_monotonic_ns")
                    if receipt.get("measured_completion_verified")
                    else None,
                    "abort_ns": receipt.get("completed_monotonic_ns")
                    if receipt["status"] == "stopped"
                    else None,
                    "error_ns": receipt.get("error_monotonic_ns"),
                    "settled_verified_ns": None,
                    "motor_ack_ns": None,
                    "status": receipt["status"],
                    "gripper_direction": (
                        (
                            "opening"
                            if writes[-1]["sent"]["gripper"] > writes[0]["measured"]["gripper"]
                            else "closing"
                            if writes[-1]["sent"]["gripper"] < writes[0]["measured"]["gripper"]
                            else "holding"
                        )
                        if "gripper" in req.get("requested", {}).get("targets", {}) and writes
                        else None
                    ),
                }
            )
        annotations = [e for e in events if e["type"] == "annotation"] + list(reviews)
        if any(e["episode_id"] != m["episode_id"] for e in annotations):
            raise ValueError("Review belongs to another episode")
        samples = join(m, frames, control, spans, annotations)
    fps = m["run_parameters"]["fps"]
    views, count = selections(samples, fps, d_target, budget)
    result = {
        "version": VERSION,
        "upstream": PIN,
        "source_family_id": m["source_family_id"],
        "episode_id": m["episode_id"],
        "configuration_id": m["experiment"]["configuration_id"],
        "clocks": clocks.report(),
        "intervals": spans,
        "annotations": annotations,
        "commands": commands,
        "samples": samples,
        "views": views,
        "eligible": len(samples),
        "raw_frames": len(frames),
        "budget_source": "fixture minimum; specify an independent --budget for comparison"
        if budget is None
        else "specified",
        "upstream_retained": count,
        "retained_per_arm": len(views["blind"]),
        "parameters": {
            "fps": fps,
            "d_target": d_target,
            "lambda_dist": 1.0,
            "lambda_acc": 0.01,
            "gripper_expand_ranges": True,
            "gripper_expand_max_gap": 20,
            "gripper_threshold": 0.05,
            "motion_epsilon": MOTION_EPSILON,
            "max_tick_gap_ns": MAX_TICK_GAP_NS,
        },
        "confidence": {
            "stable_wait": "encoder-stable within tolerance; no force evidence",
            "api_motion": "encoder change above tolerance",
            "protected_dwell": "declared purpose and host timer; mechanical settling unverified",
            "ambiguous": "insufficient or overlapping timing evidence",
        },
        "mask_source": "Provider and controller timing; no contact measurement",
        "student_input_fields": ["wrist_rgb", "side_rgb", "state", "task"],
    }
    # Read-only sidecars. Models receive RGB, proprioception and task only. Joined IDs and
    # masks remain in the analyst manifest. A chunk is permitted within one segment only.
    for name, selected in views.items():
        lookup = {r["raw_index"]: r for r in samples}
        segments = []
        for item in selected:
            row = lookup[item["raw_index"]]
            if not segments or segments[-1]["segment"] != row["segment"]:
                segments.append({"segment": row["segment"], "steps": []})
            segments[-1]["steps"].append(
                {
                    "inputs": {
                        "wrist_rgb": row["images"]["wrist"]["path"],
                        "side_rgb": row["images"]["side"]["path"],
                        "state": row["state"],
                        "task": m["task"],
                    },
                    "target": row["sent"],
                }
            )
        Path(output).with_suffix("." + name + ".json").write_text(json.dumps(segments) + "\n")
    Path(output).write_text(json.dumps(result, allow_nan=False, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("upstream", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--budget", type=int)
    p.add_argument("--d-target", type=float, default=0.05)
    p.add_argument("--reviews", type=Path)
    a = p.parse_args()
    build(
        a.archive,
        a.upstream,
        a.output,
        a.budget,
        a.d_target,
        json.loads(a.reviews.read_text()) if a.reviews else (),
    )
