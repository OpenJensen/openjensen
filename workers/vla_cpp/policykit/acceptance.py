"""Fail-closed acceptance checks; these inspect artifacts, not policy quality."""

from __future__ import annotations

import json
import math
from pathlib import Path


def artifact_contract(directory: Path, manifest: dict) -> dict:
    import gguf

    from .quantization import tensor_quantization

    reader = gguf.GGUFReader(directory / "model.gguf")
    if reader.fields["general.architecture"].contents() != "smolvla":
        raise ValueError("Only the SmolVLA native artifact contract is supported")
    dimensions = {}
    for name in ("chunk_size", "max_action_dim", "real_action_dim", "image_size"):
        field = reader.fields.get("smolvla." + name)
        value = field.contents() if field else None
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("Missing or invalid GGUF dimension: " + name)
        dimensions[name] = value
    if not (
        dimensions["chunk_size"] <= 50
        and dimensions["real_action_dim"] <= dimensions["max_action_dim"] <= 32
        and dimensions["image_size"] == 512
    ):
        raise ValueError("Unsupported dimensions for the current native probe and benchmark")
    if manifest["metadata"].get("action_dim") != dimensions["real_action_dim"]:
        raise ValueError("Artifact action dimension disagrees with GGUF metadata")
    precision = manifest["metadata"].get("precision")
    if precision == "float":
        language = vision = None
    elif (
        isinstance(precision, dict)
        and precision.get("language") in {"Q4_0", "Q8_0"}
        and precision.get("vision") in {None, "Q8_0"}
    ):
        language, vision = precision["language"], precision.get("vision")
    else:
        raise ValueError("Unsupported artifact precision declaration")
    inventory, packed = {}, {"language": 0, "vision": 0}
    packed_names = []
    for tensor in reader.tensors:
        if tensor.name in inventory:
            raise ValueError("Duplicate GGUF tensor: " + tensor.name)
        actual = tensor.tensor_type.name
        inventory[tensor.name] = actual
        expected = tensor_quantization(tensor.name, tensor.shape, language, vision)
        if expected:
            if actual != expected:
                raise ValueError("GGUF packing disagrees with recipe: " + tensor.name)
            component = "language" if tensor.name.startswith("vlm.") else "vision"
            packed[component] += 1
            packed_names.append(tensor.name)
        elif actual not in {"F32", "BF16"}:
            raise ValueError("Unsupported protected tensor precision: " + tensor.name)
    if not inventory or (language and not packed["language"]) or (vision and not packed["vision"]):
        raise ValueError("GGUF has no tensors for the declared recipe")
    if language:
        audit_path = directory / "conversion-manifest.json"
        if not audit_path.is_file():
            raise ValueError("Packed artifact requires its conversion precision audit")
        audit = json.loads(audit_path.read_text())
        if audit.get("tensor_types") != inventory or sorted(
            audit.get("quantized_tensors", [])
        ) != sorted(packed_names):
            raise ValueError("Conversion audit disagrees with GGUF tensor inventory")
    return {
        **dimensions,
        "action_length": dimensions["chunk_size"] * dimensions["max_action_dim"],
        "packed_matrices": packed,
        "tensor_count": len(inventory),
    }


def memory_coverage(measurements: dict, device: str) -> dict:
    """A complete peak needs samples from reload, timing, and every rollout."""
    key, count_key = (
        ("sampled_peak_device_used_mib", "gpu_samples")
        if device == "cuda"
        else ("sampled_peak_rss_mib", "rss_samples")
    )
    missing = {"reload", "timing"} - measurements.keys()
    peaks = []
    for stage, measurement in measurements.items():
        value, samples = measurement.get(key), measurement.get(count_key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
            or not isinstance(samples, int)
            or isinstance(samples, bool)
            or samples <= 0
        ):
            missing.add(stage)
        else:
            peaks.append(value)
    complete = not missing and bool(peaks)
    return {
        "peak_device_mib": max(peaks) if complete else None,
        "memory_coverage": {
            "complete": complete,
            "required_stages": sorted(set(measurements) | {"reload", "timing"}),
            "unmeasured_stages": sorted(missing),
            "metric": key,
            "sampling_limit": "Sampled peak; transients between samples may be missed",
        },
    }


def sanitize_measurement(measurement: dict) -> dict:
    """Keep invalid telemetry unqualified without emitting invalid JSON numbers."""
    invalid = [
        key
        for key, value in measurement.items()
        if isinstance(value, float) and not math.isfinite(value)
    ]
    if not invalid:
        return measurement
    return {
        **{key: None if key in invalid else value for key, value in measurement.items()},
        "invalid_nonfinite_fields": sorted(invalid),
    }


def satisfies_limits(report: dict, limits: dict) -> bool:
    def finite(name, positive=False):
        value = report.get(name)
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and (value > 0 if positive else 0 <= value <= 1)
        )

    return (
        finite("success_rate")
        and finite("p95_ms", positive=True)
        and finite("peak_device_mib", positive=True)
        and report.get("memory_coverage", {}).get("complete") is True
        and report["success_rate"] >= limits["min_success_rate"]
        and report["p95_ms"] <= limits["max_p95_ms"]
        and report["peak_device_mib"] <= limits["max_peak_device_mib"]
    )
