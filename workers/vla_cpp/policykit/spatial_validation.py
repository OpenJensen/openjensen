"""Static Spatial layout/statistics checks, shared by preflight and evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

DIMENSIONS = {
    "chunk_size": 50,
    "max_action_dim": 32,
    "real_action_dim": 7,
    "image_size": 512,
    "real_state_dim": 8,
    "max_state_dim": 32,
    "num_steps": 10,
}
NORMALIZERS = (
    (
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "state",
        "observation.state",
        8,
    ),
    ("policy_postprocessor_step_0_unnormalizer_processor.safetensors", "action", "action", 7),
)
MAX_NORMALIZER_BYTES = 1024 * 1024
MAX_NORMALIZER_HEADER_BYTES = 64 * 1024


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate normalization header key")
        result[key] = value
    return result


def normalizer_snapshot(path, expected_sha256=None):
    """Bound the immutable bytes before parsing metadata or allocating tensors."""
    with path.open("rb") as stream:
        payload = stream.read(MAX_NORMALIZER_BYTES + 1)
    if not 8 < len(payload) <= MAX_NORMALIZER_BYTES:
        raise ValueError("Normalization file is missing or exceeds the 1 MiB bound")
    if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Normalization snapshot SHA256 mismatch")
    size = int.from_bytes(payload[:8], "little")
    if not 2 <= size <= MAX_NORMALIZER_HEADER_BYTES or 8 + size > len(payload):
        raise ValueError("Normalization header exceeds its bound")
    header = json.loads(payload[8 : 8 + size], object_pairs_hook=unique_object)
    if not isinstance(header, dict):
        raise ValueError("Invalid normalization header")
    return payload, header


def validate_spatial_layout(reader, bundle, *, files=None):
    """Check layout and exact small stats; never load or execute native policy weights."""
    import gguf
    import numpy as np
    from safetensors.numpy import load

    architecture = reader.fields.get("general.architecture")
    if architecture is None or architecture.contents() != "smolvla":
        raise ValueError("Only a SmolVLA GGUF can pass Spatial static checks")
    for name, expected in DIMENSIONS.items():
        field = reader.fields.get("smolvla." + name)
        value = field.contents() if field is not None else None
        if type(value) is not int or value != expected:
            raise ValueError("GGUF lacks the verified Spatial state/action contract: " + name)
    tensors = {}
    for tensor in reader.tensors:
        if tensor.name in tensors:
            raise ValueError("Duplicate GGUF tensor: " + tensor.name)
        tensors[tensor.name] = tensor
    for filename, prefix, feature, dimension in NORMALIZERS:
        path = Path(bundle) / "policy" / filename
        payload, header = normalizer_snapshot(
            path, files["policy/" + filename] if files is not None else None
        )
        for suffix in ("mean", "std"):
            name = feature + "." + suffix
            entry = header.get(name)
            tensor = tensors.get(prefix + "_" + suffix)
            if not isinstance(entry, dict) or tensor is None:
                raise ValueError("Missing serialized normalization statistics")
            if (
                entry.get("shape") != [dimension]
                or entry.get("dtype") != "F32"
                or tuple(tensor.shape) != (dimension,)
                or tensor.tensor_type != gguf.GGMLQuantizationType.F32
            ):
                raise ValueError("Unsupported normalization shape or dtype")
        # The entire serialized file is bounded; only required statistics are used.
        # Additional processor statistics remain compatible with the pinned format.
        saved = load(payload)
        for suffix in ("mean", "std"):
            native = saved[feature + "." + suffix]
            converted = tensors[prefix + "_" + suffix].data.reshape(-1)
            if (
                not np.isfinite(native).all()
                or not np.array_equal(native, converted)
                or (suffix == "std" and np.any(native < 0))
            ):
                raise ValueError("GGUF normalization differs or contains invalid statistics")
    return {**DIMENSIONS, "action_length": 50 * 32, "tensor_count": len(tensors)}
