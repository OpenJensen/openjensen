"""Derive simulator semantics only from an immutable, verified local snapshot."""

import hashlib
import json
from pathlib import Path

from .control_schema import FILE, SCOPE, canonical, optional, validate


def derive(root, manifest, pointer, recipe):
    # The caller verifies the complete snapshot first. Bind these exact reads too.
    from .local_dataset import _open

    files = {item["path"]: item for item in manifest["files"]}
    name = "meta/firebird-demonstrations.json"
    if name not in files:
        return None

    def read(name):
        with _open(root, name) as stream:
            raw = stream.read(1024 * 1024 + 1)
        entry = files.get(name, {})
        if (
            len(raw) > 1024 * 1024
            or len(raw) != entry.get("size")
            or hashlib.sha256(raw).hexdigest() != entry.get("sha256")
        ):
            raise ValueError("Simulator metadata differs from verified dataset snapshot")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("Simulator metadata must be an object")
        return value

    provenance, info = read(name), read("meta/info.json")
    if manifest.get("lineage_validated") is not True:
        raise ValueError("Simulator policy requires verified episode lineage")
    constants = {
        "schema_version": 1,
        "controller": "joint_position_targets",
        "state_units": "radians",
        "action_units": "radians",
        "timebase": "simulation_seconds",
        "action_column": "action",
        "requested_action_column": "teaching.requested_action",
        "scene_hash_scope": SCOPE,
        "task_success_verified": False,
    }
    if any(
        type(provenance.get(k)) is not type(v) or provenance.get(k) != v
        for k, v in constants.items()
    ):
        raise ValueError("Unsupported simulator recording coordinate provenance")
    features = manifest["features"]
    if canonical(info.get("features")) != canonical(features):
        raise ValueError("Simulator dataset feature metadata differs from snapshot")
    joints = provenance.get("joint_order")
    for key in ("observation.state", "action"):
        feature = features.get(key, {})
        if (
            feature.get("dtype") != "float32"
            or feature.get("names") != joints
            or not isinstance(joints, list)
            or feature.get("shape") != [len(joints)]
        ):
            raise ValueError("Simulator state/action feature names differ from joint order")
    cameras = recipe.get("camera_keys")
    if not isinstance(cameras, list) or len(cameras) != 1:
        raise ValueError("Simulator policy requires exactly one selected camera")
    camera = features.get(cameras[0], {})
    shape = camera.get("shape")
    if (
        camera.get("dtype") not in {"video", "image"}
        or not isinstance(shape, list)
        or len(shape) != 3
        or shape[2] != 3
    ):
        raise ValueError("Simulator policy requires an RGB dataset camera")
    sources, episodes = provenance.get("sources"), provenance.get("episodes")
    if not isinstance(sources, list) or not sources or not isinstance(episodes, list):
        raise ValueError("Simulator recording source lineage is missing")
    sessions = {}
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("source_session_id"), str):
            raise ValueError("Invalid simulator recording source")
        if not isinstance(source.get("scene_sha256"), str) or source.get("origin") not in (
            "recorded",
            "synthetic",
        ):
            raise ValueError("Invalid simulator source origin or scene hash")
        key = source["source_session_id"]
        if key in sessions:
            raise ValueError("Duplicate simulator source session")
        sessions[key] = source
    lineage = {row["episode_index"]: row for row in manifest["lineage"]}
    indices = []
    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError("Invalid simulator recording episode")
        index, session = episode.get("episode_index"), episode.get("source_session_id")
        source = sessions.get(session)
        if (
            type(index) is not int
            or index not in lineage
            or source is None
            or lineage[index].get("origin") != source.get("origin")
            or lineage[index].get("lineage_group") != source.get("lineage_group")
        ):
            raise ValueError("Simulator source episode differs from snapshot lineage")
        indices.append(index)
    if len(indices) != len(set(indices)) or set(indices) != set(lineage):
        raise ValueError("Simulator provenance must cover every snapshot episode exactly once")
    record = {
        "schema_version": 1,
        "kind": "simulator_joint_position",
        "controller": "joint_position_targets",
        "state_key": "observation.state",
        "action_key": "action",
        "state_units": "radians",
        "action_units": "radians",
        "timebase": "simulation_seconds",
        "joint_order": joints,
        "camera": {
            "key": cameras[0],
            "height": shape[0],
            "width": shape[1],
            "prim": provenance.get("camera_prim"),
        },
        "action_fps": info.get("fps"),
        "source": {
            "dataset_snapshot_id": pointer["id"],
            "dataset_manifest_sha256": pointer["manifest_sha256"],
            "demonstrations_sha256": files[name]["sha256"],
            "scene_sha256": sorted({s.get("scene_sha256") for s in sources}),
            "scene_hash_scope": SCOPE,
            "origins": sorted({s.get("origin") for s in sources}),
        },
        "physical_calibration_verified": False,
        "task_success_verified": False,
    }
    return validate(record)


def check_resolved(record, config, fps):
    if record is None:
        return

    def feature(value):
        kind = value.type.value if hasattr(value.type, "value") else value.type
        return {"type": kind, "shape": list(value.shape)}

    validate(
        record,
        {
            "type": config.type,
            "input_features": {k: feature(v) for k, v in config.input_features.items()},
            "output_features": {k: feature(v) for k, v in config.output_features.items()},
        },
    )
    if type(fps) not in (int, float) or fps != record["action_fps"]:
        raise ValueError("Resolved dataset FPS differs from simulator control contract")


def check_checkpoint(root, recipe, config=None):
    """Check both resume envelope and stand-alone policy; missing claims fail closed."""
    expected = recipe.get("control_contract")
    record, digest = optional(root, config)
    policy, policy_digest = optional(Path(root) / "pretrained_model", config)
    if canonical(record) != canonical(expected) or record != policy or digest != policy_digest:
        raise ValueError("Checkpoint simulator control contract is missing or changed")
    return record, digest


def save(root, record):
    if record is not None:
        validate(record)
        (Path(root) / FILE).write_bytes(canonical(record))
