"""Application admission and receipt binding for the isolated ACT CPU exporter."""

import hashlib
import json
from pathlib import Path

from .training_catalog import TRAINING_MODEL_BY_ID

RECIPE = "act-vae-removal-fp32-v1"


def check_source(runtime, artifact, directory: Path, *, allow_cloud=False) -> None:
    if runtime.execution != "native" or runtime.provider != "local":
        raise ValueError("ACT inference export currently requires a local CPU worker")
    if not runtime.act_export_python or not runtime.act_export_root:
        raise ValueError("Configure the separate ACT inference export environment first")
    if artifact.format != "training_checkpoint" or artifact.metadata.get("architecture") != "act":
        raise ValueError("ACT export requires a complete native ACT training checkpoint")
    if (
        artifact.metadata.get("training_backend") != "lerobot"
        or artifact.metadata.get("method") != "full"
    ):
        raise ValueError("ACT export requires a native full-training checkpoint")
    dataset = artifact.metadata.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("source") != "huggingface":
        raise ValueError(
            "ACT export requires pinned Hugging Face lineage; local snapshots are unsupported"
        )
    if allow_cloud and (directory / "remote.json").is_file():
        from .cloud_materialize import descriptor

        _, _, manifest = descriptor(directory, artifact.manifest_sha256)
        if manifest.get("metadata") != artifact.metadata:
            raise ValueError("Registered cloud checkpoint metadata changed")
        return
    if any((directory / name).exists() for name in ("remote.json", "remote-checkpoint.json")):
        raise ValueError("Materialize the complete cloud checkpoint locally before ACT export")
    if artifact.metadata.get("storage") == "gcs" or artifact.metadata.get("remote"):
        raise ValueError("Materialize the complete cloud checkpoint locally before ACT export")
    if not (directory / "checkpoint/pretrained_model/model.safetensors").is_file():
        raise ValueError("ACT export requires a complete local checkpoint bundle")


def read_json(path: Path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("ACT export receipt is missing or oversized")
    return json.loads(path.read_bytes())


def check_result(info, manifest, directory: Path, report, source, source_directory: Path) -> None:
    """Do not register an artifact with another source's proof or unsupported claims."""
    if info.get("format") != "inference_export":
        raise ValueError("ACT worker must publish a distinct inference export")
    metadata = manifest.get("metadata", {})
    checkpoint = source_directory / "checkpoint/manifest.json"
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    checkpoint_step = read_json(checkpoint).get("step")
    expected = {
        "checkpoint_manifest_sha256": checkpoint_sha,
        "checkpoint_step": checkpoint_step,
        "dataset": {
            key: source.metadata.get("dataset", {}).get(key)
            for key in ("source", "repo_id", "revision")
        },
        "camera_keys": source.metadata.get("camera_keys"),
        "architecture": "act",
        "recipe": RECIPE,
        "precision": "float32",
        "inference_only": True,
        "training_resume_supported": False,
        "policy_subdirectory": "policy",
        "source_artifact_id": source.id,
        "source_manifest_sha256": source.manifest_sha256,
        "base_model": {
            "repository": "code://lerobot/act",
            "revision": TRAINING_MODEL_BY_ID["act"].model_revision,
        },
        "synthetic_parity_verified": True,
        "fresh_reload_verified": True,
        "task": "unverified",
        "task_success": None,
        "calibration_verified": False,
    }
    if any(
        metadata.get(key) != value
        or report.get(key) != value
        or (
            type(value) is bool
            and (type(metadata.get(key)) is not bool or type(report.get(key)) is not bool)
        )
        for key, value in expected.items()
    ):
        raise ValueError("ACT inference export identity or proof differs from its source")
    if type(metadata.get("checkpoint_step")) is not int or metadata["checkpoint_step"] < 1:
        raise ValueError("ACT inference export lacks a saved checkpoint step")
    if report.get("checkpoint_step") != metadata["checkpoint_step"]:
        raise ValueError("ACT export checkpoint step differs across its evidence")
    policy_manifest = directory / "policy/manifest.json"
    digest = hashlib.sha256(policy_manifest.read_bytes()).hexdigest()
    if (
        metadata.get("policy_manifest_sha256") != digest
        or report.get("policy_manifest_sha256") != digest
    ):
        raise ValueError("ACT export policy manifest differs from the exact tested package")
    receipt = read_json(directory / "verification.json")
    reload = receipt.get("final_package_reload", {})
    if (
        reload.get("status") != "passed"
        or reload.get("exact_equal") is not True
        or reload.get("source_reads_blocked") is not True
        or reload.get("manifest_sha256") != digest
    ):
        raise ValueError("ACT export lacks complete-package CPU reload evidence")
    if report.get("gpu_memory_bytes") is not None or report.get("inference_speedup") is not None:
        raise ValueError("ACT CPU export cannot claim a measured GPU improvement")
