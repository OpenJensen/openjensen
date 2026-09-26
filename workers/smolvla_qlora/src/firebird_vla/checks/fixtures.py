"""Safe, portable native-preprocessed batches; no arbitrary pickle or generated robot data."""

import json
import re
from pathlib import Path

from firebird_vla.checkpoint import sha256, write_json


def validate_fixture(manifest, entry):
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "native_training_batch":
        raise ValueError("Expected a native training-batch fixture")
    if manifest.get("model_id") != entry["id"] or manifest.get("checkpoint") != entry["checkpoint"]:
        raise ValueError("Fixture is bound to another model/checkpoint/subdirectory")
    if manifest.get("suite") != "libero_spatial":
        raise ValueError("SO-101 or other suite data cannot stand in for LIBERO-Spatial")
    provenance = manifest.get("provenance", {})
    for key in ("source", "native_code_commit", "processor_description"):
        if not provenance.get(key):
            raise ValueError(f"Missing fixture provenance: {key}")
    if not re.fullmatch(r"[0-9a-f]{40}", provenance["native_code_commit"]):
        raise ValueError("Fixture native_code_commit must be an immutable SHA")
    if provenance.get("kind") not in ("recorded", "synthetic"):
        raise ValueError("Fixture provenance kind must be recorded or explicitly synthetic")
    if provenance.get("split") not in ("train", "diagnostic"):
        raise ValueError("QLoRA checks must not train on search or final-evaluation observations")
    if entry["id"] == "gr00t_n17" and provenance.get("embodiment") != "LIBERO_PANDA":
        raise ValueError("GR00T fixture must preserve the LIBERO_PANDA embodiment")


def save_fixture(directory, entry, batch, provenance):
    """Call immediately after the model's native processor/collator, before moving to CUDA."""
    import torch
    from safetensors.torch import save_file

    tensors = {}

    def encode(value):
        if isinstance(value, torch.Tensor):
            name = f"tensor_{len(tensors)}"
            tensors[name] = value.detach().cpu().contiguous().clone()
            return {"tensor": name}
        if isinstance(value, dict):
            return {"dict": {k: encode(v) for k, v in value.items()}}
        if isinstance(value, (list, tuple)):
            return {"list": [encode(v) for v in value]}
        if value is None or type(value) in (str, bool, int, float):
            return {"value": value}
        raise TypeError(f"Convert native custom containers into dicts before export: {type(value)}")

    manifest = {
        "schema_version": 1,
        "kind": "native_training_batch",
        "model_id": entry["id"],
        "checkpoint": entry["checkpoint"],
        "suite": "libero_spatial",
        "provenance": provenance,
        "batch": encode(batch),
    }
    validate_fixture(manifest, entry)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    save_file(tensors, str(directory / "batch.safetensors"))
    manifest["batch_sha256"] = sha256(directory / "batch.safetensors")
    write_json(directory / "fixture.json", manifest)


def load_fixture(directory, entry):
    from safetensors.torch import load_file

    directory = Path(directory)
    manifest = json.loads((directory / "fixture.json").read_text())
    validate_fixture(manifest, entry)
    if sha256(directory / "batch.safetensors") != manifest["batch_sha256"]:
        raise ValueError("Fixture tensor hash mismatch")
    tensors = load_file(str(directory / "batch.safetensors"), device="cuda:0")

    def decode(node):
        if set(node) == {"tensor"}:
            return tensors[node["tensor"]]
        if set(node) == {"dict"}:
            return {k: decode(v) for k, v in node["dict"].items()}
        if set(node) == {"list"}:
            return [decode(v) for v in node["list"]]
        if set(node) == {"value"}:
            return node["value"]
        raise ValueError("Invalid fixture tree")

    return decode(manifest["batch"]), manifest
