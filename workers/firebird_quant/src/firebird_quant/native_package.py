"""Bounded, byte-bound native policy package; no ML imports in this module."""

import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path

SCHEMA = 1
FORMAT = "firebird_quant"
MODEL_DOMAIN = b"firebird-native-packed-policy-v1\0"
JSON_LIMIT = 1024 * 1024
WEIGHT_LIMIT = 512 * 1024 * 1024
TOTAL_LIMIT = 768 * 1024 * 1024
NAME = re.compile(r"[A-Za-z0-9_.-]{1,120}\Z")
SHA = re.compile(r"[a-f0-9]{64}\Z")
RUNTIME = {"lerobot": "0.6.1", "torch": "2.11.0", "torchvision": "0.26.0", "safetensors": "0.8.0"}
BASE = {"config.json", "policy_preprocessor.json", "policy_postprocessor.json"}


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def finite(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Nonfinite JSON")
        return number

    value = json.loads(
        raw.decode("utf-8"), object_pairs_hook=pairs, parse_float=finite, parse_constant=finite
    )
    if not isinstance(value, dict):
        raise ValueError("JSON record must be an object")
    return value


def checked_path(value):
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise ValueError("A bounded absolute local path is required")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Paths must be absolute without traversal")
    for parent in [path, *path.parents]:
        if parent.is_symlink() or (hasattr(parent, "is_junction") and parent.is_junction()):
            raise ValueError("Path links are unsupported")
    return path


def read(path, limit=JSON_LIMIT):
    checked_path(str(path))
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    parent = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = child
        fd = os.open(path.name, flags, dir_fd=parent)
    finally:
        os.close(parent)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit:
            raise ValueError("Expected bounded nonempty regular file")
        value = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
        if len(value) != before.st_size or (before.st_mtime_ns, before.st_size) != (
            after.st_mtime_ns,
            after.st_size,
        ):
            raise ValueError("File changed during reading")
        return value


def read_json(path):
    return decode(read(path))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def inventory(root):
    checked_path(str(root))
    if not root.is_dir():
        raise ValueError("Policy directory is missing")
    entries = list(root.iterdir())
    if not 1 <= len(entries) <= 16:
        raise ValueError("Policy requires 1..16 flat files")
    result, total = {}, 0
    for path in sorted(entries):
        if not NAME.fullmatch(path.name):
            raise ValueError("Invalid policy filename")
        raw = read(
            path, WEIGHT_LIMIT if path.name in {"model.safetensors", "model.fbq"} else JSON_LIMIT
        )
        total += len(raw)
        if total > TOTAL_LIMIT:
            raise ValueError("Policy exceeds aggregate byte bound")
        result[path.name] = {"sha256": sha(raw), "bytes": len(raw)}
    return result


def validate_inventory(value):
    if not isinstance(value, dict) or not 1 <= len(value) <= 16:
        raise ValueError("A complete bounded source inventory is required")
    total = 0
    for name, item in value.items():
        if not isinstance(name, str) or not NAME.fullmatch(name) or not isinstance(item, dict):
            raise ValueError("Invalid source inventory entry")
        if (
            set(item) != {"sha256", "bytes"}
            or not isinstance(item["sha256"], str)
            or not SHA.fullmatch(item["sha256"])
        ):
            raise ValueError("Invalid source hash")
        limit = WEIGHT_LIMIT if name == "model.safetensors" else JSON_LIMIT
        if type(item["bytes"]) is not int or not 0 < item["bytes"] <= limit:
            raise ValueError("Invalid source file size")
        total += item["bytes"]
    if total > TOTAL_LIMIT:
        raise ValueError("Source exceeds aggregate byte bound")


def admit_source(root, expected, manifest_sha):
    from firebird_act.bundle import (
        tensor_header,
        validate_config,
        validate_processors,
        verify_export,
    )

    validate_inventory(expected)
    if inventory(root) != expected:
        raise ValueError("Source inventory differs from admitted bytes")
    if "manifest.json" in expected:
        if not isinstance(manifest_sha, str) or manifest_sha != expected["manifest.json"]["sha256"]:
            raise ValueError("Source manifest hash differs")
        manifest = read_json(root / "manifest.json")
        if manifest.get("recipe") == "act-vae-removal-fp32-v1":
            verify_export(root)
        else:
            files = manifest.get("files")
            actual = {k: v for k, v in expected.items() if k != "manifest.json"}
            if (
                type(manifest.get("schema_version")) is not int
                or manifest["schema_version"] != 1
                or not isinstance(files, dict)
            ):
                raise ValueError("Unsupported source manifest")
            if files not in (actual, {k: v["sha256"] for k, v in actual.items()}):
                raise ValueError("Source manifest inventory differs")
    elif manifest_sha is not None:
        raise ValueError("Source has no direct manifest")
    config = read_json(root / "config.json")
    validate_config(config, source=False)
    required = BASE | validate_processors(root, config) | {"model.safetensors"}
    if not required <= set(expected) or not set(expected) <= required | {
        "manifest.json",
        "recipe.json",
        "parity.json",
        "train_config.json",
        "export-lineage.json",
    }:
        raise ValueError("Unsupported complete ACT policy files")
    tensor_header(read(root / "model.safetensors", WEIGHT_LIMIT))
    if inventory(root) != expected:
        raise ValueError("Source changed during admission")
    return required, config


def encoding(bits):
    if type(bits) is not int or bits not in {4, 8}:
        raise ValueError("Packed precision must be integer 4 or 8")
    return {
        "schema_version": 1,
        "format": FORMAT,
        "format_version": 1,
        "policy_family": "act",
        "weights": "model.fbq",
        "compute_dtype": "float32",
        "recipe": {
            "bits": bits,
            "group_size": 64,
            "min_elements": 128,
            "min_ndim": 2,
            "include": [],
            "exclude": [],
        },
        "runtime": dict(RUNTIME),
    }


def model_identity(files):
    digest = hashlib.sha256(MODEL_DOMAIN)
    for name, item in sorted(files.items()):
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(item["bytes"].to_bytes(8, "big"))
        digest.update(bytes.fromhex(item["sha256"]))
    return "sha256:" + digest.hexdigest()


def inspect_policy(root):
    from firebird_act.bundle import validate_config, validate_processors

    files = inventory(root)
    encoded = read_json(root / "encoding.json")
    recipe = encoded.get("recipe")
    if not isinstance(recipe, dict):
        raise ValueError("Native packed encoding requires a recipe object")
    bits = recipe.get("bits")
    if (
        type(encoded.get("schema_version")) is not int
        or type(encoded.get("format_version")) is not int
        or type(recipe.get("group_size")) is not int
        or canonical(encoded) != canonical(encoding(bits))
    ):
        raise ValueError("Unsupported native packed encoding")
    config = read_json(root / "config.json")
    validate_config(config, source=False)
    required = BASE | validate_processors(root, config) | {"model.fbq", "encoding.json"}
    if set(files) != required:
        raise ValueError(
            "Native package must contain exactly its inference files and no float master"
        )
    return {
        "model_id": model_identity(files),
        "files": files,
        "encoding": encoded,
        "quality_verified": False,
        "calibration_verified": False,
        "speedup_verified": False,
        "task_success": None,
    }


def write_new(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
