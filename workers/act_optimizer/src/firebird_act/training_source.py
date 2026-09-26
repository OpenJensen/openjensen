"""Point-in-time admission of a complete native ACT training bundle; no model loading."""

import hashlib
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .bundle import (
    CORE_FILES,
    JSON_LIMIT,
    decode,
    inventory,
    read_json,
    removal_keys,
    safe_file,
    tensor_header,
    validate_config,
    validate_processors,
)

TRAINING_REVISION = "e595b7902714ba51f91e47523f66f89c5181b649"
MAX_ENTRIES = 128
MAX_FILE_BYTES = 2 * 1024**3
MAX_BUNDLE_BYTES = 4 * 1024**3
RESUME_FILES = {
    "recipe.json",
    "probe-batch.pt",
    "probe-action.pt",
    "pretrained_model/config.json",
    "pretrained_model/model.safetensors",
    "pretrained_model/train_config.json",
    "training_state/optimizer_state.safetensors",
    "training_state/optimizer_param_groups.json",
    "training_state/rng_state.safetensors",
    "training_state/training_step.json",
}


@dataclass(frozen=True)
class ActTrainingSource:
    """A checked path, not a durable authorization to load it after files change."""

    source: Path
    artifact_id: str
    manifest_sha256: str
    checkpoint_manifest_sha256: str
    step: int
    dataset_id: str
    dataset_revision: str
    camera: str
    source_files: tuple[tuple[str, str, int], ...]

    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "scope": "offline ACT training-source admission only",
            "source_artifact_id": self.artifact_id,
            "source_manifest_sha256": self.manifest_sha256,
            "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
            "checkpoint_step": self.step,
            "source_subdirectory": "checkpoint/pretrained_model",
            "upstream_revision": TRAINING_REVISION,
            "dataset": {
                "repository": self.dataset_id,
                "revision": self.dataset_revision,
            },
            "camera_key": self.camera,
            "source_files": {
                name: {"sha256": digest, "bytes": size} for name, digest, size in self.source_files
            },
            "runtime_compatibility_verified": False,
            "export_verified": False,
            "task_success": None,
        }


def _sha(value: Any, length: int = 64) -> bool:
    return isinstance(value, str) and re.fullmatch(rf"[a-f0-9]{{{length}}}", value) is not None


def _name(value: str) -> bool:
    path = PurePosixPath(value)
    return (
        bool(value)
        and len(value) <= 1024
        and not path.is_absolute()
        and path.as_posix() == value
        and not any(
            part.startswith(".") or part.endswith((".tmp", ".partial")) for part in path.parts
        )
        and not any(char in value for char in ("\\", ":", "\0"))
    )


def _tree(root: Path) -> dict[str, tuple[str, int]]:
    """Hash bounded regular files without following descendant symlinks or opening FIFOs."""
    result: dict[str, tuple[str, int]] = {}
    pending = [root]
    count = total = 0
    while pending:
        directory = pending.pop()
        if directory.is_symlink() or directory.is_junction() or not directory.is_dir():
            raise ValueError("Training bundle directories must not be symlinks")
        with os.scandir(directory) as entries:
            for entry in entries:
                count += 1
                name = Path(entry.path).relative_to(root).as_posix()
                if count > MAX_ENTRIES or not _name(name):
                    raise ValueError("Training bundle has unsafe paths or too many entries")
                if entry.is_symlink() or Path(entry.path).is_junction():
                    raise ValueError("Training bundle symlinks are not supported")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                    continue
                flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
                fd = os.open(entry.path, flags)
                with os.fdopen(fd, "rb") as stream:
                    info = os.fstat(stream.fileno())
                    if not stat.S_ISREG(info.st_mode) or not 0 <= info.st_size <= MAX_FILE_BYTES:
                        raise ValueError("Training bundle requires bounded regular files")
                    total += info.st_size
                    if total > MAX_BUNDLE_BYTES:
                        raise ValueError("Training bundle exceeds its size limit")
                    digest, size = hashlib.sha256(), 0
                    while block := stream.read(min(1024**2, info.st_size + 1 - size)):
                        size += len(block)
                        if size > info.st_size:
                            raise ValueError("Training bundle changed while reading")
                        digest.update(block)
                    if size != info.st_size:
                        raise ValueError("Training bundle changed while reading")
                    result[name] = digest.hexdigest(), size
    return result


def _manifest(root: Path, files: dict[str, tuple[str, int]]) -> dict[str, Any]:
    data = safe_file(root / "manifest.json", JSON_LIMIT)
    if files.get("manifest.json") != (hashlib.sha256(data).hexdigest(), len(data)):
        raise ValueError("Training manifest changed during admission")
    document = decode(data)
    expected = document.get("files")
    if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise ValueError("Unsupported training manifest schema")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Training manifest requires a complete file inventory")
    if any(not _name(name) or not _sha(digest) for name, digest in expected.items()):
        raise ValueError("Training manifest has unsafe paths or invalid hashes")
    actual = {name: digest for name, (digest, _) in files.items() if name != "manifest.json"}
    if actual != expected:
        raise ValueError("Training manifest inventory or hash mismatch")
    return document


def admit_training_source(
    bundle: Path, *, artifact_id: str, manifest_sha256: str
) -> ActTrainingSource:
    """Select a completed full ACT bundle's policy without changing any source byte.

    The caller supplies the registered outer manifest hash. Trusted operator-owned
    parent directories are required, as with the exporter. Re-admit after changes;
    this performs no native load, extraction, export, resume or cross-version proof.
    """
    if not isinstance(artifact_id, str) or not artifact_id.strip() or len(artifact_id) > 200:
        raise ValueError("A registered source artifact identity is required")
    if not _sha(manifest_sha256):
        raise ValueError("A registered source manifest hash is required")
    bundle = Path(bundle)
    if bundle.is_symlink() or bundle.is_junction() or not bundle.is_dir():
        raise ValueError("Training bundle must be a nonsymlink directory")
    bundle = bundle.resolve()
    before = _tree(bundle)
    if "remote.json" in before or "remote-checkpoint.json" in before:
        raise ValueError("Remote training descriptors must be materialized before admission")
    if before.get("manifest.json", (None,))[0] != manifest_sha256:
        raise ValueError("Registered source manifest hash mismatch")
    outer = _manifest(bundle, before)
    metadata = outer.get("metadata")
    if not isinstance(metadata, dict) or any(
        metadata.get(key) != value
        for key, value in {
            "architecture": "act",
            "training_backend": "lerobot",
            "method": "full",
        }.items()
    ):
        raise ValueError("Source must be a native full ACT training bundle")
    inner_files = {
        name[11:]: value for name, value in before.items() if name.startswith("checkpoint/")
    }
    checkpoint = bundle / "checkpoint"
    inner = _manifest(checkpoint, inner_files)
    if not RESUME_FILES.issubset(inner_files):
        raise ValueError("Source is missing its complete native resume inventory")
    if any(
        inner.get(key) != value
        for key, value in {
            "representation": "lerobot-native-full",
            "training_backend": "lerobot",
            "policy_type": "act",
            "upstream_revision": TRAINING_REVISION,
        }.items()
    ):
        raise ValueError("Unsupported ACT training checkpoint identity or upstream pin")
    step = inner.get("step")
    if type(step) is not int or step < 1:
        raise ValueError("ACT training checkpoint requires a positive optimizer step")
    # The pinned upstream save_training_metadata writes step beside topology and
    # precision metadata. Bind its counter without coercing booleans/strings.
    saved_step = read_json(checkpoint / "training_state/training_step.json").get("step")
    if type(saved_step) is not int or saved_step < 1 or saved_step != step:
        raise ValueError("Saved native training step differs from the checkpoint manifest")
    recipe = read_json(checkpoint / "recipe.json")
    if any(
        recipe.get(key) != value
        for key, value in {
            "policy_type": "act",
            "training_backend": "lerobot",
            "method": "full",
            "upstream_revision": TRAINING_REVISION,
            "model_revision": TRAINING_REVISION,
            "model_id": "code://lerobot/act",
        }.items()
    ):
        raise ValueError("ACT training recipe identity or upstream pin differs")
    if type(recipe.get("steps")) is not int or not step <= recipe["steps"]:
        raise ValueError("Checkpoint step exceeds the saved training recipe")
    if metadata.get("base_model") != {
        "repository": recipe["model_id"],
        "revision": TRAINING_REVISION,
    }:
        raise ValueError("ACT base-model lineage differs from its saved recipe")
    verification = read_json(bundle / "verification.json")
    difference = verification.get("max_abs_action_difference")
    if (
        metadata.get("reload_verified") is not True
        or verification.get("reload_verified") is not True
        or not isinstance(difference, (int, float))
        or isinstance(difference, bool)
        or not math.isfinite(difference)
        or difference < 0
    ):
        raise ValueError("Completed training reload evidence is required")
    source = checkpoint / "pretrained_model"
    config = read_json(source / "config.json")
    validate_config(config, source=True)
    names = CORE_FILES | validate_processors(source, config) | {"train_config.json"}
    read_json(source / "train_config.json")
    selected = inventory(source)
    if set(selected) != names:
        raise ValueError("Unexpected or missing pretrained ACT files")
    header, _ = tensor_header(safe_file(source / "model.safetensors"))
    if {key for key in header if key.startswith("model.vae_encoder")} != removal_keys(config):
        raise ValueError("ACT VAE tensor inventory differs from the supported recipe")
    camera = next(key for key in config["input_features"] if key.startswith("observation.images."))
    if recipe.get("camera_keys") != [camera] or metadata.get("camera_keys") != [camera]:
        raise ValueError("Saved ACT camera lineage differs from its policy")
    if recipe.get("chunk_size", 100) != config["chunk_size"] or metadata.get("action_dim") != 6:
        raise ValueError("Saved ACT action/chunk lineage differs from its policy")
    dataset = metadata.get("dataset")
    if (
        not isinstance(dataset, dict)
        or dataset.get("source") != "huggingface"
        or dataset.get("repo_id") != recipe.get("dataset_id")
        or not isinstance(recipe.get("dataset_id"), str)
        or not recipe["dataset_id"]
        or dataset.get("revision") != recipe.get("dataset_revision")
        or not _sha(recipe.get("dataset_revision"), 40)
    ):
        raise ValueError("Saved ACT dataset lineage differs from its pinned recipe")
    for name, identity in selected.items():
        if before.get("checkpoint/pretrained_model/" + name) != (
            identity["sha256"],
            identity["bytes"],
        ):
            raise ValueError("Selected ACT source changed during admission")
    if _tree(bundle) != before:
        raise ValueError("Training bundle changed during admission")
    return ActTrainingSource(
        source,
        artifact_id,
        manifest_sha256,
        inner_files["manifest.json"][0],
        step,
        recipe["dataset_id"],
        recipe["dataset_revision"],
        camera,
        tuple((name, item["sha256"], item["bytes"]) for name, item in sorted(selected.items())),
    )
