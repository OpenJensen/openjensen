"""Exact inherited inference semantics, without importing Torch or a simulator."""

from firebird_act.bundle import (
    canonical, control_files, safe_file, temporal_dimensions, temporal_files,
    validate_processors,
)
from firebird_act.control_schema import optional as control_optional


def policy_metadata(root, config):
    # Validate both independently; control_files also checks their common cadence.
    temporal = temporal_files(root, config)
    control_files(root, config)
    record, control_sha = control_optional(root, config)
    from .contracts import digest

    return {
        **temporal_dimensions(config),
        "temporal_contract_sha256": (
            digest(safe_file(root / "temporal-contract.json", 1024**2)) if temporal else None
        ),
        "control_contract": record,
        "control_contract_sha256": control_sha,
    }


def action_fps(root, metadata):
    from .contracts import read

    if metadata["temporal_contract_sha256"] is not None:
        return read(root / "temporal-contract.json")["action_fps"]
    record = metadata["control_contract"]
    return record["action_fps"] if record is not None else None


def inherited_files(root, config):
    return (
        {"policy_preprocessor.json", "policy_postprocessor.json"}
        | validate_processors(root, config)
        | temporal_files(root, config)
        | control_files(root, config)
    )


def check_snapshot_control(root, manifest, pointer, camera, metadata, expected_fps):
    """Called only after the complete immutable snapshot verifier has succeeded."""
    if expected_fps is not None and manifest.get("fps") != expected_fps:
        raise ValueError("Dataset FPS differs from the teacher sampling contract")
    record = metadata["control_contract"]
    if record is None:
        return
    source = record["source"]
    if (
        pointer.get("id") != source["dataset_snapshot_id"]
        or pointer.get("manifest_sha256") != source["dataset_manifest_sha256"]
    ):
        raise ValueError("Simulator distillation requires the exact teacher source snapshot")
    from firebird_vla.control_contract import derive

    actual = derive(root, manifest, pointer, {"camera_keys": [camera]})
    if canonical(actual) != canonical(record):
        raise ValueError("Dataset simulator joints, camera, cadence or provenance differ")


def check_corpus_metadata(doc, config, metadata, expected_fps):
    from .contracts import integer

    dimensions = temporal_dimensions(config)
    integer(doc.get("chunk_size"), 1, 1024)
    if not isinstance(doc.get("source"), dict) or not isinstance(doc.get("semantics"), dict):
        raise ValueError("Corpus source and coordinate semantics must be objects")
    if doc["chunk_size"] != dimensions["prediction_horizon"]:
        raise ValueError("Corpus prediction horizon differs from teacher")
    fields = {
        "execution_horizon", "action_fps", "temporal_contract_sha256",
        "control_contract", "control_contract_sha256",
    }
    present = fields.intersection(doc)
    if not present:
        if dimensions != {"prediction_horizon": 100, "execution_horizon": 100} or any(
            metadata[k] is not None for k in (
                "temporal_contract_sha256", "control_contract", "control_contract_sha256"
            )
        ):
            raise ValueError("Legacy corpus cannot omit nondefault teacher semantics")
        return fields
    if present != fields:
        raise ValueError("Incomplete corpus timing/control metadata")
    integer(doc["execution_horizon"], 1, dimensions["prediction_horizon"])
    if doc["execution_horizon"] != dimensions["execution_horizon"]:
        raise ValueError("Corpus execution horizon differs from teacher")
    import math

    fps = doc["action_fps"]
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Corpus action FPS must be positive and finite")
    if expected_fps is not None and fps != expected_fps:
        raise ValueError("Corpus FPS differs from teacher sampling contract")
    for key in fields - {"execution_horizon", "action_fps"}:
        if canonical(doc[key]) != canonical(metadata[key]):
            raise ValueError("Corpus timing/control provenance differs from teacher")
    record = metadata["control_contract"]
    if record is not None:
        source, semantics = doc["source"], doc["semantics"]
        if (
            source.get("identity") != record["source"]["dataset_snapshot_id"]
            or source.get("revision") != record["source"]["dataset_manifest_sha256"]
            or semantics.get("state_names") != record["joint_order"]
            or semantics.get("action_names") != record["joint_order"]
            or semantics.get("units") != ["radians"] * len(record["joint_order"])
            or doc.get("camera") != record["camera"]["key"]
            or fps != record["action_fps"]
        ):
            raise ValueError("Corpus source or coordinates differ from simulator contract")
        # The exact source snapshot may contain both origins while this selected
        # corpus is homogeneous. prepare() and core admission reject mixed selection.
        origins = record["source"]["origins"]
        selected_origin = {"generated_fixture": "synthetic", "lerobot": "recorded"}.get(
            source.get("kind") if isinstance(source.get("kind"), str) else ""
        )
        if selected_origin not in origins:
            raise ValueError("Corpus origin differs from simulator recording provenance")
    return fields
