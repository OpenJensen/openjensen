"""Strict, bounded control and corpus contracts; importing this module never imports Torch."""

import hashlib
import json
import math
import os
import re
from pathlib import Path

from firebird_act.bundle import (
    CORE_FILES,
    canonical,
    temporal_dimensions,
    inventory,
    safe_file,
    validate_config,
    validate_processors,
)

SHA = re.compile(r"[0-9a-f]{64}")
MAX_SAMPLES = 256
MAX_CORPUS_BYTES = 2 * 1024**3
MAX_SAMPLE_BYTES = 8 * 1024**2
SPLITS = {"train", "validation", "final"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def number(raw):
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError("Non-finite JSON number")
    return value


def decode(raw):
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_float=number,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite JSON")),
    )
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value


def read(path):
    return decode(safe_file(path, 1024**2))


def exact_keys(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("Unexpected or missing fields")


def integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"Expected integer {low}..{high}")
    return value


def identity(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError("Expected bounded identity")
    return value


def sha(value):
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ValueError("Expected SHA256")
    return value


def path(value):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError("Expected operator-owned absolute path")
    result = Path(value)
    if result != result.resolve() or any(p.is_symlink() for p in [result, *result.parents]):
        # /tmp is a legitimate macOS alias; callers must resolve before dispatch.
        raise ValueError("Source/output paths must be resolved without symlinks")
    return result


def validate_files(files):
    if not isinstance(files, dict) or not 1 <= len(files) <= 16:
        raise ValueError("Expected complete bounded teacher inventory")
    for name, item in files.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,150}", name):
            raise ValueError("Expected flat source filenames")
        exact_keys(item, {"sha256", "bytes"})
        sha(item["sha256"])
        integer(item["bytes"], 1, 512 * 1024**2)


def request(value):
    exact_keys(
        value,
        {
            "schema_version",
            "job_id",
            "operation",
            "teacher",
            "dataset",
            "recipe",
            "output_dir",
            "timeout_seconds",
        },
    )
    integer(value["schema_version"], 1, 1)
    identity(value["job_id"])
    if value["operation"] != "policy.distill":
        raise ValueError("Only policy.distill is supported")
    teacher = value["teacher"]
    exact_keys(teacher, {"path", "files", "artifact_id", "artifact_manifest_sha256"})
    identity(teacher["artifact_id"])
    sha(teacher["artifact_manifest_sha256"])
    validate_files(teacher["files"])
    exact_keys(value["dataset"], {"path", "manifest_sha256"})
    sha(value["dataset"]["manifest_sha256"])
    recipe = value["recipe"]
    exact_keys(recipe, {"adapter", "student", "steps", "learning_rate", "seed"})
    if recipe["adapter"] != "act-act-v1" or recipe["student"] != "act-256":
        raise ValueError("Only the ACT→ACT256 adapter is implemented")
    integer(recipe["steps"], 1, 10000)
    integer(recipe["seed"], 0, 2**31 - 1)
    lr = recipe["learning_rate"]
    if type(lr) not in (int, float) or not math.isfinite(lr) or not 1e-7 <= lr <= 1e-3:
        raise ValueError("Invalid learning rate")
    integer(value["timeout_seconds"], 1, 3600)
    roots = [path(teacher["path"]), path(value["dataset"]["path"]), path(value["output_dir"])]
    for i, left in enumerate(roots):
        for right in roots[i + 1 :]:
            if left == right or left.is_relative_to(right) or right.is_relative_to(left):
                raise ValueError("Teacher, corpus and output directories must be separate")
    if os.path.lexists(roots[2] / "distilled-policy"):
        raise FileExistsError("Distilled artifact already exists")
    return roots


def policy_info(root, expected, *, source_metadata=False):
    if inventory(root) != expected:
        raise ValueError("Teacher inventory changed")
    cfg = read(root / "config.json")
    if type(cfg.get("use_vae")) is not bool:
        raise ValueError("Teacher must declare VAE mode")
    for key in ("input_features", "output_features"):
        if not isinstance(cfg.get(key), dict) or any(
            not isinstance(v, dict) for v in cfg[key].values()
        ):
            raise ValueError("Invalid teacher features")
    validate_config(cfg, source=cfg["use_vae"])
    stats = validate_processors(root, cfg)
    from .provenance import inherited_files, policy_metadata

    policy_metadata(root, cfg)
    required = CORE_FILES | inherited_files(root, cfg)
    allowed = required | ({
        "manifest.json", "recipe.json", "parity.json", "train_config.json", "export-lineage.json"
    } if source_metadata else set())
    if not required <= set(expected) <= allowed:
        raise ValueError("Unexpected files in ACT inference payload")
    camera = next(k for k in cfg["input_features"] if k.startswith("observation.images."))
    processors = {
        name: expected[name]
        for name in sorted(stats | {"policy_preprocessor.json", "policy_postprocessor.json"})
    }
    return cfg, camera, digest(canonical(processors))


def teacher_info(root, expected):
    cfg, camera, processors = policy_info(root, expected, source_metadata=True)
    if cfg["dim_model"] <= 256 or cfg["n_encoder_layers"] < 2:
        raise ValueError("Teacher must be larger than the fixed ACT256 student")
    return cfg, camera, processors


def corpus(root, expected_sha, cfg, camera, processors_sha, *, metadata=None, expected_fps=None):
    raw = safe_file(root / "manifest.json", 1024**2)
    if digest(raw) != expected_sha:
        raise ValueError("Corpus manifest identity mismatch")
    doc = decode(raw)
    from .provenance import check_corpus_metadata

    if metadata is None:
        metadata = {
            **temporal_dimensions(cfg), "temporal_contract_sha256": None,
            "control_contract": None, "control_contract_sha256": None,
        }
    extra = check_corpus_metadata(doc, cfg, metadata, expected_fps)
    exact_keys(
        doc,
        {
            "schema_version",
            "format",
            "source",
            "semantics",
            "camera",
            "image_shape",
            "chunk_size",
            "samples",
        } | (extra if "execution_horizon" in doc else set()),
    )
    integer(doc["schema_version"], 1, 1)
    if doc["format"] != "act-observation-corpus-v1" or doc["camera"] != camera:
        raise ValueError("Corpus adapter/camera mismatch")
    shape = doc["image_shape"]
    if (
        not isinstance(shape, list)
        or any(type(n) is not int for n in shape)
        or shape != cfg["input_features"][camera]["shape"]
    ):
        raise ValueError("Corpus image shape must match the teacher")
    source = doc["source"]
    exact_keys(source, {"kind", "identity", "revision", "inventory_sha256"})
    if source["kind"] not in {"lerobot", "generated_fixture"}:
        raise ValueError("Unsupported observation provenance")
    identity(source["identity"])
    identity(source["revision"])
    sha(source["inventory_sha256"])
    semantics = doc["semantics"]
    exact_keys(
        semantics,
        {"state_names", "action_names", "units", "compatibility", "teacher_processors_sha256"},
    )
    for key in ("state_names", "action_names", "units"):
        if not isinstance(semantics[key], list) or len(semantics[key]) != 6:
            raise ValueError("Six explicit coordinate names/units are required")
        for item in semantics[key]:
            identity(item)
    if semantics["state_names"] != semantics["action_names"]:
        raise ValueError("State/action ordering must match this ACT adapter")
    if len(set(semantics["state_names"])) != 6:
        raise ValueError("Coordinate names must be unique")
    expected = (
        "generated_fixture"
        if source["kind"] == "generated_fixture"
        else "operator_attested_teacher_recorded_coordinates"
    )
    if (
        semantics["compatibility"] != expected
        or semantics["teacher_processors_sha256"] != processors_sha
    ):
        raise ValueError("Teacher coordinate/processor compatibility is not established")
    samples = doc["samples"]
    if not isinstance(samples, list) or not 3 <= len(samples) <= MAX_SAMPLES:
        raise ValueError("Corpus must contain 3..256 samples")
    episodes, groups, observed, files, counts = {}, {}, set(), set(), dict.fromkeys(SPLITS, 0)
    total = 0
    for i, sample in enumerate(samples):
        exact_keys(
            sample,
            {
                "file",
                "sha256",
                "bytes",
                "episode_id",
                "lineage_group",
                "frame_index",
                "episode_length",
                "split",
            },
        )
        if sample["file"] != f"sample-{i:06d}.safetensors":
            raise ValueError("Corpus filenames must be canonical and ordered")
        sha(sample["sha256"])
        total += integer(sample["bytes"], 1, MAX_SAMPLE_BYTES)
        episode = integer(sample["episode_id"], 0, 2**31 - 1)
        frame = integer(sample["frame_index"], 0, 10**7)
        length = integer(sample["episode_length"], 1, 10**7)
        group = identity(sample["lineage_group"])
        split = sample["split"]
        if (
            not isinstance(split, str)
            or split not in SPLITS
            or frame >= length
            or (episode, frame) in observed
        ):
            raise ValueError("Invalid split/frame or repeated observation")
        if episodes.setdefault(episode, (split, group, length)) != (split, group, length):
            raise ValueError("Episode leaks across splits or changes lineage/length")
        if groups.setdefault(group, split) != split:
            raise ValueError("Lineage group leaks across splits")
        counts[split] += 1
        observed.add((episode, frame))
        files.add(sample["file"])
    if total > MAX_CORPUS_BYTES or not all(counts.values()):
        raise ValueError("Bounded train/validation/final data are required")
    if root.is_symlink() or set(p.name for p in root.iterdir()) != files | {"manifest.json"}:
        raise ValueError("Unexpected corpus files")
    return doc


def load_sample(root, sample, shape, prediction_horizon=100):
    """Decode an immutable, size/hash checked byte snapshot, never an arbitrary pickle."""
    import torch
    from safetensors.torch import load

    integer(prediction_horizon, 1, 1024)
    raw = safe_file(root / sample["file"], MAX_SAMPLE_BYTES)
    if len(raw) != sample["bytes"] or digest(raw) != sample["sha256"]:
        raise ValueError("Corpus sample changed")
    data = load(raw)
    specs = {
        "image": (torch.uint8, shape),
        "state": (torch.float32, [6]),
        "actions": (torch.float32, [prediction_horizon, 6]),
        "padding": (torch.bool, [prediction_horizon]),
    }
    if set(data) != set(specs):
        raise ValueError("Unexpected observation tensors")
    for key, (dtype, dims) in specs.items():
        if (
            data[key].dtype != dtype
            or list(data[key].shape) != dims
            or not torch.isfinite(data[key]).all()
        ):
            raise ValueError("Invalid observation tensor shape/dtype/values")
    expected = torch.arange(prediction_horizon) + sample["frame_index"] >= sample["episode_length"]
    if not torch.equal(data["padding"], expected) or bool(expected.all()):
        raise ValueError("Action padding does not match the true episode boundary")
    return data
