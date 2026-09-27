"""Inspect exported policies without importing the inference runtime."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

_EXPORT_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)
_PROCESSOR_FILES = ("policy_preprocessor.json", "policy_postprocessor.json")
_HASH_BLOCK_SIZE = 1024 * 1024
_HASH_FORMAT = b"sim-policy-checkpoint-v1\0"
_STATE_KEY = "observation.state"
_RGB_CHANNELS = 3
_IDENTITY = "IDENTITY"


@dataclass(frozen=True)
class Checkpoint:
    model_id: str
    policy_type: str
    camera_key: str
    width: int
    height: int
    state_dim: int
    action_dim: int
    chunk_size: int
    action_steps: int


def inspect_checkpoint(path: Path) -> Checkpoint:
    """Validate metadata and fingerprint weights plus their saved processors."""
    root = path.resolve()
    files = set(_EXPORT_FILES)
    for name in _EXPORT_FILES:
        _file(root, name)
    config = _json(root / "config.json")
    policy_type = config.get("type")
    if policy_type not in {"act", "smolvla"}:
        raise ValueError("Only ACT and SmolVLA checkpoint exports are supported")
    if config.get("n_obs_steps") != 1 or config.get("use_peft", False):
        raise ValueError("Checkpoint must be a full export with one observation step")
    if policy_type == "act" and config.get("temporal_ensemble_coeff") is not None:
        raise ValueError("ACT temporal ensembling is not supported by chunk inference")
    inputs = config.get("input_features", {})
    outputs = config.get("output_features", {})
    if not isinstance(inputs, dict) or not isinstance(outputs, dict):
        raise ValueError("Checkpoint features must be dictionaries")
    cameras = [name for name in inputs if name.startswith("observation.images.")]
    if len(cameras) != 1 or set(inputs) != {_STATE_KEY, cameras[0]}:
        raise ValueError("Checkpoint must expect one camera and observation.state")
    if set(outputs) != {"action"}:
        raise ValueError("Checkpoint must output only joint actions")
    state = _shape(inputs[_STATE_KEY], "STATE", 1)
    action = _shape(outputs["action"], "ACTION", 1)
    image = _shape(inputs[cameras[0]], "VISUAL", 3)
    if image[0] != _RGB_CHANNELS:
        raise ValueError("Checkpoint camera must have RGB channels")
    chunk_size = _positive(config.get("chunk_size"))
    action_steps = _positive(config.get("n_action_steps", chunk_size))
    if action_steps > chunk_size:
        raise ValueError("Checkpoint n_action_steps exceeds its chunk size")

    # Saved statistics affect policy behavior and belong in its identity.
    for name in _PROCESSOR_FILES:
        files.update(_state_files(root, _json(root / name)))
    digest = hashlib.sha256(_HASH_FORMAT)
    for name in sorted(files):
        item = _file(root, name)
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(item.stat().st_size.to_bytes(8, "big"))
        with item.open("rb") as stream:
            while block := stream.read(_HASH_BLOCK_SIZE):
                digest.update(block)
    return Checkpoint(
        f"sha256:{digest.hexdigest()}",
        policy_type,
        cameras[0],
        image[2],
        image[1],
        state[0],
        action[0],
        chunk_size,
        action_steps,
    )


def _state_files(root, processor):
    steps = processor.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("Checkpoint processors must contain saved steps")
    result = set()
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError("Invalid checkpoint processor step")
        filename = step.get("state_file")
        registry = step.get("registry_name")
        if registry in {"normalizer_processor", "unnormalizer_processor"}:
            config = step.get("config", {})
            norm_map = config.get("norm_map", {})
            if any(mode != _IDENTITY for mode in norm_map.values()) and not filename:
                raise ValueError("Checkpoint normalization statistics are missing")
        if filename is None:
            continue
        _file(root, filename)
        result.add(filename)
    return result


def _file(root, name):
    if not isinstance(name, str) or not name:
        raise ValueError("Invalid checkpoint filename")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Checkpoint files must stay inside the export directory")
    result = root / relative
    if not result.resolve().is_relative_to(root):
        raise ValueError("Checkpoint files must stay inside the export directory")
    if not result.is_file() or result.stat().st_size == 0:
        raise ValueError(f"Checkpoint export is missing {name}")
    return result


def _json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, ValueError) as error:
        raise ValueError(f"Invalid checkpoint JSON: {path.name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Invalid checkpoint JSON: {path.name}")
    return value


def _shape(feature, kind, dimensions):
    if not isinstance(feature, dict) or feature.get("type") != kind:
        raise ValueError(f"Checkpoint feature must have type {kind}")
    shape = feature.get("shape")
    if not isinstance(shape, list) or len(shape) != dimensions:
        raise ValueError("Invalid checkpoint feature dimensions")
    return [_positive(value) for value in shape]


def _positive(value):
    if type(value) is not int or value < 1:
        raise ValueError("Checkpoint dimensions must be positive integers")
    return value
