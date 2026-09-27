"""Strict immutable observation/identity contracts; this module never imports ML."""

import math
import os

from firebird_quant.native_package import (
    RUNTIME,
    SHA,
    canonical,
    checked_path,
    decode,
    inspect_policy,
    read,
    sha,
)

MAX_SAMPLES = 32
MAX_RGB_BYTES = 1920 * 1920 * 3
MAX_TOTAL_BYTES = 128 * 1024**2


def keys(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("Unexpected or missing native replay fields")


def integer(value, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError("Expected a bounded integer")
    return value


def text(value, limit=256):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError("Expected bounded nonempty text")
    return value


def hash_value(value):
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ValueError("Expected SHA256")
    return value


def number(value, lower=-3.4e38, upper=3.4e38):
    if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError("Expected a finite bounded number")
    return float(value)


def request(value):
    keys(
        value,
        {
            "schema_version",
            "job_id",
            "operation",
            "source",
            "observations",
            "output_dir",
            "timeout_seconds",
        },
    )
    integer(value["schema_version"], 1, 1)
    text(value["job_id"])
    if value["operation"] != "policy.run":
        raise ValueError("Native replay supports only policy.run")
    integer(value["timeout_seconds"], 1, 600)
    source = value["source"]
    keys(source, {"path", "files", "model_id", "artifact_id", "artifact_manifest_sha256"})
    text(source["artifact_id"])
    hash_value(source["artifact_manifest_sha256"])
    model_id = source["model_id"]
    if not isinstance(model_id, str) or not model_id.startswith("sha256:"):
        raise ValueError("Expected native model identity")
    hash_value(model_id[7:])
    keys(value["observations"], {"path", "manifest_sha256"})
    hash_value(value["observations"]["manifest_sha256"])
    roots = [
        checked_path(source["path"]),
        checked_path(value["observations"]["path"]),
        checked_path(value["output_dir"]),
    ]
    for i, left in enumerate(roots):
        for right in roots[i + 1 :]:
            a, b = left.resolve(), right.resolve()
            if a == b or a.is_relative_to(b) or b.is_relative_to(a):
                raise ValueError("Policy, observations and output must be separate")
    if os.path.lexists(roots[2] / "native-run"):
        raise FileExistsError("Native replay artifact already exists")
    return roots


def policy(root, source):
    info = inspect_policy(root)
    if info["files"] != source["files"] or info["model_id"] != source["model_id"]:
        raise ValueError("Packed source differs from its admitted bytes or model identity")
    raw = read(root / "config.json")
    if sha(raw) != info["files"]["config.json"]["sha256"]:
        raise ValueError("Policy config changed during admission")
    return {**info, "config": decode(raw)}


def observations(root, expected_sha, config):
    """Read and hash each original RGB snapshot; return bounded JSON plus file inventory."""
    checked_path(str(root))
    raw = read(root / "manifest.json")
    if sha(raw) != expected_sha:
        raise ValueError("Observation manifest identity changed")
    doc = decode(raw)
    keys(doc, {"schema_version", "format", "source", "camera_key", "semantics", "samples"})
    integer(doc["schema_version"], 1, 1)
    if doc["format"] != "native-policy-observations-v1":
        raise ValueError("Unsupported observation format")
    cameras = [name for name in config["input_features"] if name.startswith("observation.images.")]
    if cameras != [doc["camera_key"]]:
        raise ValueError("Observation camera differs from policy")
    shape = config["input_features"][doc["camera_key"]]["shape"]
    source = doc["source"]
    keys(source, {"kind", "identity", "manifest_sha256"})
    if not isinstance(source["kind"], str) or source["kind"] not in {
        "lerobot_snapshot",
        "generated_fixture",
    }:
        raise ValueError("Observation source kind must be explicit")
    text(source["identity"])
    hash_value(source["manifest_sha256"])
    semantics = doc["semantics"]
    keys(semantics, {"state_names", "action_names", "units", "compatibility"})
    for field in ("state_names", "action_names", "units"):
        if not isinstance(semantics[field], list) or len(semantics[field]) != 6:
            raise ValueError("Six explicit coordinate names/units are required")
        for value in semantics[field]:
            text(value)
    if (
        len(set(semantics["state_names"])) != 6
        or semantics["state_names"] != semantics["action_names"]
    ):
        raise ValueError("This ACT adapter requires matching unique state/action coordinate order")
    expected = (
        "generated_fixture"
        if source["kind"] == "generated_fixture"
        else "operator_attested_policy_recorded_coordinates"
    )
    if semantics["compatibility"] != expected:
        raise ValueError("Policy coordinate compatibility must be explicitly attested")
    samples = doc["samples"]
    if not isinstance(samples, list) or not 1 <= len(samples) <= MAX_SAMPLES:
        raise ValueError("Native replay requires1..32 selected observations")
    files, seen, total = {"manifest.json": {"sha256": sha(raw), "bytes": len(raw)}}, set(), 0
    for i, sample in enumerate(samples):
        keys(
            sample,
            {
                "episode_index",
                "frame_index",
                "timestamp_seconds",
                "task",
                "state",
                "origin",
                "lineage_group",
                "image",
            },
        )
        episode = integer(sample["episode_index"], 0, 2**31 - 1)
        frame = integer(sample["frame_index"], 0, 10**7)
        if (episode, frame) in seen:
            raise ValueError("Duplicate selected observation")
        seen.add((episode, frame))
        number(sample["timestamp_seconds"], 0, 1e7)
        text(sample["task"], 4096)
        if not isinstance(sample["origin"], str) or sample["origin"] not in {
            "recorded",
            "imported",
            "augmented",
            "synthetic",
        }:
            raise ValueError("Explicit observation origin required")
        if (sample["origin"] == "synthetic") != (source["kind"] == "generated_fixture"):
            raise ValueError("Generated observation origins cannot be relabeled as recorded")
        if sample["lineage_group"] is not None:
            text(sample["lineage_group"], 160)
        if not isinstance(sample["state"], list) or len(sample["state"]) != 6:
            raise ValueError("Expected six raw state coordinates")
        for value in sample["state"]:
            number(value)
        image = sample["image"]
        keys(image, {"file", "width", "height", "sha256", "bytes"})
        if image["file"] != f"frame-{i:06d}.rgb":
            raise ValueError("RGB filenames must be canonical and ordered")
        width, height = integer(image["width"], 2, 1920), integer(image["height"], 2, 1920)
        if shape != [3, height, width]:
            raise ValueError(
                "Recorded RGB resolution must exactly match policy; no implicit resize"
            )
        integer(image["bytes"], 1, MAX_RGB_BYTES)
        hash_value(image["sha256"])
        if image["bytes"] != width * height * 3:
            raise ValueError("RGB declared size differs from dimensions")
        total += image["bytes"]
        if total > MAX_TOTAL_BYTES:
            raise ValueError("Observation corpus exceeds128MiB")
        files[image["file"]] = {"sha256": image["sha256"], "bytes": image["bytes"]}
    # Enforce sample and aggregate budgets BEFORE any expensive RGB file IO.
    for name, identity in files.items():
        if name == "manifest.json":
            continue
        rgb = read(root / name, MAX_RGB_BYTES)
        if len(rgb) != identity["bytes"] or sha(rgb) != identity["sha256"]:
            raise ValueError("RGB bytes differ from the admitted observation")
    if set(p.name for p in root.iterdir()) != set(files):
        raise ValueError("Observation bundle contains unexpected files")
    return doc, files


def publish(stage, destination):
    """No-replace publication with file and directory sync; success follows persistence."""
    from firebird_act.bundle import publish_new_directory

    for file in stage.iterdir():
        with file.open("rb") as stream:
            os.fsync(stream.fileno())
    sync_directory(stage)
    publish_new_directory(stage, destination)
    sync_directory(destination.parent)


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_result(value, source, doc):
    keys(
        value,
        {
            "schema_version",
            "model_id",
            "versions",
            "device",
            "mode",
            "records",
            "server_closed",
            "external_network_disabled",
            "floating_master_reads_blocked",
            "task_success",
            "quality_verified",
            "calibration_verified",
            "speedup_verified",
            "isaac_runtime_verified",
        },
    )
    integer(value["schema_version"], 1, 1)
    if (
        value["model_id"] != source["model_id"]
        or value["device"] != "cpu"
        or value["mode"] != "independent_observation_replay"
    ):
        raise ValueError("Replay runtime identity differs")
    for name in ("server_closed", "external_network_disabled", "floating_master_reads_blocked"):
        if value[name] is not True:
            raise ValueError("Replay lifecycle evidence is incomplete")
    for name in (
        "quality_verified",
        "calibration_verified",
        "speedup_verified",
        "isaac_runtime_verified",
    ):
        if value[name] is not False:
            raise ValueError("Replay cannot promote robotics quality claims")
    if value["task_success"] is not None:
        raise ValueError("Runtime replay does not measure task success")
    keys(value["versions"], RUNTIME)
    for key, expected in RUNTIME.items():
        version = value["versions"][key]
        if not isinstance(version, str) or version.split("+")[0] != expected:
            raise ValueError("Replay runtime pin differs")
    records = value["records"]
    if not isinstance(records, list) or len(records) != len(doc["samples"]):
        raise ValueError("Incomplete replay observation coverage")
    for i, (row, sample) in enumerate(zip(records, doc["samples"])):
        keys(
            row,
            {
                "sample_index",
                "episode_index",
                "frame_index",
                "timestamp_seconds",
                "input_rgb_sha256",
                "input_state_sha256",
                "actions",
                "reset_repeat_exact",
                "requests",
            },
        )
        integer(row["sample_index"], i, i)
        integer(row["episode_index"], sample["episode_index"], sample["episode_index"])
        integer(row["frame_index"], sample["frame_index"], sample["frame_index"])
        if (
            number(row["timestamp_seconds"], 0, 1e7) != sample["timestamp_seconds"]
            or row["input_rgb_sha256"] != sample["image"]["sha256"]
            or row["input_state_sha256"] != sha(canonical(sample["state"]))
            or row["reset_repeat_exact"] is not True
        ):
            raise ValueError("Replay result differs from exact selected input")
        actions = row["actions"]
        if not isinstance(actions, list) or len(actions) != 100:
            raise ValueError("Replay requires full100x6 output")
        for action in actions:
            if not isinstance(action, list) or len(action) != 6:
                raise ValueError("Replay requires full100x6 output")
            for coordinate in action:
                number(coordinate)
        if not isinstance(row["requests"], list) or len(row["requests"]) != 2:
            raise ValueError("Two reset comparisons are required")
        for request_receipt in row["requests"]:
            keys(request_receipt, {"seconds", "actions_sha256"})
            number(request_receipt["seconds"], 0, 600)
            if request_receipt["actions_sha256"] != sha(canonical(actions)):
                raise ValueError("Reset output hashes differ from recorded predictions")
