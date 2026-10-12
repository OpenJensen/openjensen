"""Bounded local checkpoint parsing and byte-preserving ACT tensor extraction."""

import ctypes
import hashlib
import json
import math
import os
import platform
import re
import stat
import struct
from pathlib import Path
from typing import Any, cast

from .control_schema import FILE as CONTROL_FILE
from .control_schema import optional as control_optional

JSON_LIMIT = 1024 * 1024
WEIGHT_LIMIT = 512 * 1024 * 1024
CORE_FILES = {
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
}
NORM_MAP = {"VISUAL": "MEAN_STD", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def decode(data: bytes) -> dict[str, Any]:
    value = json.loads(data, object_pairs_hook=_pairs, parse_constant=_bad_constant)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value


def _bad_constant(value: str) -> Any:
    raise ValueError(f"Non-finite JSON: {value}")


def safe_file(path: Path, limit: int = WEIGHT_LIMIT) -> bytes:
    """Read one bounded regular file without following a final symlink."""
    if path.is_symlink():
        raise ValueError(f"Symlink is not supported: {path.name}")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
        os.close(fd)
        raise ValueError(f"Invalid file or size: {path.name}")
    with os.fdopen(fd, "rb") as stream:
        data = stream.read(limit + 1)
        if len(data) != info.st_size:
            raise ValueError(f"File changed while reading: {path.name}")
        return data


def read_json(path: Path) -> dict[str, Any]:
    return decode(safe_file(path, JSON_LIMIT))


def file_identity(path: Path) -> dict[str, Any]:
    data = safe_file(path)
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def inventory(root: Path) -> dict[str, dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Checkpoint must be a nonsymlink directory")
    entries = list(root.iterdir())
    if not 1 <= len(entries) <= 16:
        raise ValueError("Checkpoint must contain at most 16 files")
    return {p.name: file_identity(p) for p in sorted(entries)}


def temporal_dimensions(config: dict[str, Any]) -> dict[str, int]:
    prediction, execution = config.get("chunk_size"), config.get("n_action_steps")
    if (
        type(prediction) is not int
        or type(execution) is not int
        or not 1 <= execution <= prediction <= 1024
    ):
        raise ValueError("ACT horizons require 1 <= execution <= prediction <= 1024")
    return {"prediction_horizon": prediction, "execution_horizon": execution}


def validate_temporal_contract(config: dict[str, Any], record: dict[str, Any]) -> None:
    """Bind optional native training sampling provenance to the exact saved ACT config."""
    horizons = temporal_dimensions(config)
    fps = cast(int | float, record.get("action_fps"))
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Temporal contract requires positive finite action FPS")
    observation = record.get("observation_delta_indices")
    if observation not in (None, [0]) or (
        observation is not None and any(type(i) is not int for i in observation)
    ):
        raise ValueError("Unsupported ACT observation indices")
    expected = {
        "schema_version": 1,
        "family": "act",
        "action_fps": fps,
        "policy_fields": {
            "chunk_size": horizons["prediction_horizon"],
            "n_action_steps": horizons["execution_horizon"],
            "n_obs_steps": 1,
        },
        "action_delta_indices": list(range(horizons["prediction_horizon"])),
        "observation_delta_indices": observation,
        "action_delta_timestamps": [i / fps for i in range(horizons["prediction_horizon"])],
        "observation_delta_timestamps": None if observation is None else [0.0],
        "observation_history": 1,
        "frame_stride": 1,
        **horizons,
    }
    # Canonical JSON distinguishes bool/int and int/float in all identity fields.
    if canonical(record) != canonical(expected):
        raise ValueError("Temporal contract differs from the saved ACT configuration")


def control_files(root: Path, config: dict[str, Any]) -> set[str]:
    record, _ = control_optional(root, config)
    if record is not None and (root / "temporal-contract.json").exists():
        if read_json(root / "temporal-contract.json").get("action_fps") != record["action_fps"]:
            raise ValueError("Temporal action FPS differs from simulator control contract")
    return {CONTROL_FILE} if record is not None else set()


def temporal_files(root: Path, config: dict[str, Any]) -> set[str]:
    name = "temporal-contract.json"
    if not os.path.lexists(root / name):
        return set()
    validate_temporal_contract(config, read_json(root / name))
    return {name}


def validate_config(config: dict[str, Any], *, source: bool) -> None:
    temporal_dimensions(config)
    dilation = config.get("replace_final_stride_with_dilation")
    if not (
        (type(dilation) is bool and dilation is False) or (type(dilation) is int and dilation == 0)
    ):
        raise ValueError("Unsupported ACT config: replace_final_stride_with_dilation")
    required = {
        "type": "act",
        "n_obs_steps": 1,
        "use_vae": source,
        "use_peft": False,
        "use_amp": False,
        "temporal_ensemble_coeff": None,
        "vision_backbone": "resnet18",
        "pre_norm": False,
        "feedforward_activation": "relu",
        "normalization_mapping": NORM_MAP,
    }
    for key, value in required.items():
        if type(config.get(key)) is not type(value) or config[key] != value:
            raise ValueError(f"Unsupported ACT config: {key}")
    inputs = config.get("input_features")
    outputs = config.get("output_features")
    if not isinstance(inputs, dict) or not isinstance(outputs, dict):
        raise ValueError("Missing ACT features")
    cameras = [k for k in inputs if k.startswith("observation.images.")]
    if len(cameras) != 1 or set(inputs) != {"observation.state", cameras[0]}:
        raise ValueError("ACT requires one camera and six state coordinates")
    if inputs["observation.state"] != {"type": "STATE", "shape": [6]} or outputs != {
        "action": {"type": "ACTION", "shape": [6]}
    }:
        raise ValueError("ACT requires six state/action coordinates")
    image = inputs[cameras[0]]
    shape = image.get("shape") if isinstance(image, dict) else None
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or type(shape[0]) is not int
        or shape[0] != 3
        or image.get("type") != "VISUAL"
        or any(type(n) is not int or not 32 <= n <= 2048 for n in shape[1:])
        or shape[1] * shape[2] > 1920 * 1080
    ):
        raise ValueError("Unsupported RGB image shape")
    for key, low, high in (
        ("dim_model", 32, 512),
        ("n_heads", 1, 8),
        ("dim_feedforward", 64, 3200),
        ("n_encoder_layers", 1, 4),
        ("n_decoder_layers", 1, 4),
        ("n_vae_encoder_layers", 1, 4),
        ("latent_dim", 1, 32),
    ):
        value = config.get(key)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"Unsupported ACT dimension: {key}")
    if config["dim_model"] % 4 or config["dim_model"] % config["n_heads"]:
        raise ValueError("ACT model dimension must divide positional/attention heads")


def validate_processors(root: Path, config: dict[str, Any]) -> set[str]:
    """Accept only the saved built-in processor recipe used by this ACT slice."""
    files: set[str] = set()
    features = config["input_features"] | config["output_features"]
    for filename, registries in (
        (
            "policy_preprocessor.json",
            [
                "rename_observations_processor",
                "to_batch_processor",
                "device_processor",
                "normalizer_processor",
            ],
        ),
        ("policy_postprocessor.json", ["unnormalizer_processor", "device_processor"]),
    ):
        document = read_json(root / filename)
        if set(document) != {"name", "steps"} or document["name"] != filename[:-5]:
            raise ValueError("Unsupported processor document")
        steps = document["steps"]
        if (
            not isinstance(steps, list)
            or [s.get("registry_name") for s in steps if isinstance(s, dict)] != registries
        ):
            raise ValueError("Unsupported processor steps")
        for step, registry in zip(steps, registries, strict=True):
            cfg = step.get("config")
            if registry in {"normalizer_processor", "unnormalizer_processor"}:
                expected = (
                    features if registry == "normalizer_processor" else config["output_features"]
                )
                if cfg != {"eps": 1e-8, "features": expected, "norm_map": NORM_MAP}:
                    raise ValueError("Processor features/normalization differ from ACT config")
                name = step.get("state_file")
                if not isinstance(name, str) or not re.fullmatch(
                    r"[A-Za-z0-9_-]+\.safetensors", name
                ):
                    raise ValueError("Processor state must be a local safetensors filename")
                if name in CORE_FILES or name in files:
                    raise ValueError("Processor state files must be distinct")
                validate_statistics(safe_file(root / name, JSON_LIMIT), expected, features)
                files.add(name)
                if set(step) != {"registry_name", "config", "state_file"}:
                    raise ValueError("Unsupported processor step fields")
            else:
                if set(step) != {"registry_name", "config"}:
                    raise ValueError("Unsupported processor step fields")
                if registry == "rename_observations_processor" and cfg != {"rename_map": {}}:
                    raise ValueError("Observation renaming is unsupported")
                if registry == "to_batch_processor" and cfg != {}:
                    raise ValueError("Unsupported batching config")
                if registry == "device_processor" and (
                    not isinstance(cfg, dict)
                    or set(cfg) != {"device", "float_dtype"}
                    or cfg["device"] not in {"cpu", "cuda"}
                    or cfg["float_dtype"] is not None
                ):
                    raise ValueError("Unsupported processor device/dtype")
    return files


def validate_statistics(data: bytes, required: dict[str, Any], configured: dict[str, Any]) -> None:
    """Reject LeRobot's silent identity fallback when MEAN_STD stats are absent."""
    header, offset = tensor_header(data)
    required_keys = {f"{name}.{statistic}" for name in required for statistic in ("mean", "std")}
    if not required_keys.issubset(header):
        raise ValueError("Missing required feature normalization statistics")
    allowed_shapes = {
        name: [3, 1, 1] if feature["type"] == "VISUAL" else feature["shape"]
        for name, feature in configured.items()
    }
    allowed_shapes.update(
        {name: [1] for name in ("episode_index", "frame_index", "index", "task_index", "timestamp")}
    )
    allowed_stats = {"mean", "std", "min", "max", "count", "q01", "q10", "q50", "q90", "q99"}
    values: dict[str, dict[str, tuple[float, ...]]] = {}
    for key, item in header.items():
        if key == "__metadata__":
            continue
        feature, _, statistic = key.rpartition(".")
        if feature not in allowed_shapes or statistic not in allowed_stats:
            raise ValueError("Unsupported auxiliary processor statistics")
        expected = allowed_shapes[feature]
        if statistic == "count":
            shapes = [[1], [1, 1, 1]] if expected == [3, 1, 1] else [[1]]
            if item["shape"] not in shapes:
                raise ValueError("Invalid processor count statistics shape")
        elif item["shape"] != expected:
            raise ValueError("Processor statistics shape differs from configured feature")
        start, end = item["data_offsets"]
        numbers = struct.unpack(f"<{(end - start) // 4}f", data[offset + start : offset + end])
        if not all(math.isfinite(number) for number in numbers):
            raise ValueError("Non-finite processor statistics")
        if statistic == "std" and any(number < 0 for number in numbers):
            raise ValueError("Negative processor standard deviation statistics")
        if statistic == "count" and any(
            number < 1 or not number.is_integer() for number in numbers
        ):
            raise ValueError("Processor count statistics must be positive integers")
        values.setdefault(feature, {})[statistic] = numbers
    for stats in values.values():
        ordered = [
            stats[k] for k in ("min", "q01", "q10", "q50", "q90", "q99", "max") if k in stats
        ]
        if any(
            any(
                a > b and not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7)
                for a, b in zip(left, right, strict=True)
            )
            for left, right in zip(ordered, ordered[1:])
        ):
            raise ValueError("Processor range/quantile statistics are inconsistent")


def tensor_header(data: bytes) -> tuple[dict[str, Any], int]:
    if len(data) < 8:
        raise ValueError("Truncated safetensors file")
    size = struct.unpack("<Q", data[:8])[0]
    if not 0 < size <= JSON_LIMIT or size % 8 or 8 + size > len(data):
        raise ValueError("Invalid safetensors header length")
    header = decode(data[8 : 8 + size])
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    if not 1 <= len(tensors) <= 4096:
        raise ValueError("Invalid tensor count")
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(v, str) for v in metadata.values()):
        raise ValueError("Invalid tensor metadata")
    for key, item in tensors.items():
        if not isinstance(item, dict) or set(item) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"Invalid tensor record: {key}")
        shape, offsets = item["shape"], item["data_offsets"]
        if (
            item["dtype"] != "F32"
            or not isinstance(shape, list)
            or len(shape) > 8
            or any(type(n) is not int or n < 1 for n in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(n) is not int or n < 0 for n in offsets)
            or offsets[1] - offsets[0] != math.prod(shape) * 4
        ):
            raise ValueError(f"Only valid dense FP32 tensors are supported: {key}")
    end = 0
    for item in sorted(tensors.values(), key=lambda v: v["data_offsets"][0]):
        start, stop = item["data_offsets"]
        if start != end:
            raise ValueError("Tensor data must cover the payload without gaps/overlap")
        end = stop
    if 8 + size + end != len(data):
        raise ValueError("Tensor payload size differs from header")
    return header, 8 + size


def removal_keys(config: dict[str, Any]) -> set[str]:
    suffixes = (
        "linear1.bias",
        "linear1.weight",
        "linear2.bias",
        "linear2.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
        "self_attn.in_proj_bias",
        "self_attn.in_proj_weight",
        "self_attn.out_proj.bias",
        "self_attn.out_proj.weight",
    )
    keys = {
        f"model.vae_encoder.layers.{i}.{suffix}"
        for i in range(config["n_vae_encoder_layers"])
        for suffix in suffixes
    }
    keys |= {
        f"model.vae_encoder_{name}.{part}"
        for name in ("action_input_proj", "latent_output_proj", "robot_state_input_proj")
        for part in ("bias", "weight")
    }
    return keys | {"model.vae_encoder_cls_embed.weight", "model.vae_encoder_pos_enc"}


def strip_vae(source: Path, output: Path, config: dict[str, Any]) -> dict[str, Any]:
    data = safe_file(source)
    header, base = tensor_header(data)
    removed = removal_keys(config)
    if {k for k in header if k.startswith("model.vae_encoder")} != removed:
        raise ValueError("VAE tensor inventory does not match the supported config")
    result: dict[str, Any] = {}
    if "__metadata__" in header:
        result["__metadata__"] = header["__metadata__"]
    chunks: list[bytes] = []
    offset = 0
    kept: dict[str, str] = {}
    for key, item in header.items():
        if key == "__metadata__" or key in removed:
            continue
        start, stop = item["data_offsets"]
        chunk = data[base + start : base + stop]
        kept[key] = hashlib.sha256(chunk).hexdigest()
        result[key] = item | {"data_offsets": [offset, offset + len(chunk)]}
        chunks.append(chunk)
        offset += len(chunk)
    raw = json.dumps(result, separators=(",", ":"), allow_nan=False).encode()
    raw += b" " * (-len(raw) % 8)
    with output.open("xb") as stream:
        stream.write(struct.pack("<Q", len(raw)))
        stream.write(raw)
        for chunk in chunks:
            stream.write(chunk)
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "removed_keys": sorted(removed),
        "retained_tensor_sha256": kept,
        "original_tensor_payload_bytes": len(data) - base,
        "export_tensor_payload_bytes": offset,
        "stored_dtype": "F32",
        "runtime_dtype": "float32",
    }


def publish_new_directory(staging: Path, output: Path) -> None:
    """Use OS no-replace rename, including a raced empty destination."""
    args: tuple[Any, ...]
    system = platform.system()
    if system == "Windows":
        os.rename(staging, output)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if system == "Linux" and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        args = (-100, os.fsencode(staging), -100, os.fsencode(output), 1)
    elif system == "Darwin" and hasattr(libc, "renamex_np"):
        rename = libc.renamex_np
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        args = (os.fsencode(staging), os.fsencode(output), 4)
    else:
        raise OSError("Atomic no-replace publication is unavailable on this platform")
    rename.restype = ctypes.c_int
    if rename(*args) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(output))


def verify_export(root: Path) -> dict[str, Any]:
    """Verify an exported package's complete inventory before accepting its weights."""
    manifest = read_json(root / "manifest.json")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("recipe") != "act-vae-removal-fp32-v1"
        or manifest.get("inference_only") is not True
        or manifest.get("family") != "act"
    ):
        raise ValueError("Unsupported ACT export manifest")
    files = inventory(root)
    del files["manifest.json"]
    if manifest.get("files") != files:
        raise ValueError("ACT package inventory/hash verification failed")
    config = read_json(root / "config.json")
    validate_config(config, source=False)
    for key, value in temporal_dimensions(config).items():
        if key in manifest or value != 100:
            if type(manifest.get(key)) is not int or manifest[key] != value:
                raise ValueError("ACT export temporal manifest differs from saved config")
    required = (
        CORE_FILES
        | validate_processors(root, config)
        | temporal_files(root, config)
        | control_files(root, config)
        | {"recipe.json", "parity.json"}
    )
    if "train_config.json" in files:
        required.add("train_config.json")
    if set(files) != required:
        raise ValueError("Unexpected ACT package files")
    return manifest
