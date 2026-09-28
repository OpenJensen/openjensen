"""Application admission and receipt binding for the isolated ACT CPU exporter."""

import hashlib
from pathlib import Path

from .control_provenance import policy_claims, training_claims
from .control_schema import canonical
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
    dataset_lineage(artifact.metadata, directory)
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
    training_claims(directory, artifact.metadata)


def read_json(path: Path):
    from .simulation import strict_json

    return strict_json(path, 4 * 1024 * 1024)


def dataset_lineage(metadata: dict, directory: Path) -> tuple[dict, dict]:
    """Project portable lineage from saved training bytes, never caller overrides."""
    dataset = metadata.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("ACT export requires pinned dataset lineage")
    if dataset.get("source") == "huggingface":
        return {key: dataset.get(key) for key in ("source", "repo_id", "revision")}, {}
    if dataset.get("source") != "local":
        raise ValueError("ACT export requires a pinned Hub dataset or complete local snapshot")
    from vla_platform.contracts import DatasetProfile

    try:
        profile = DatasetProfile.model_validate(dataset)
    except ValueError as error:
        raise ValueError("ACT export requires a valid complete local dataset profile") from error
    if profile.inspection_scope != "complete_snapshot" or profile.snapshot is None:
        raise ValueError("ACT export requires a complete immutable local dataset snapshot")
    snapshot = profile.snapshot
    checkpoint = directory / "checkpoint"
    manifest = read_json(checkpoint / "manifest.json")
    recipe_path = checkpoint / "recipe.json"
    recipe = read_json(recipe_path)
    recipe_sha = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
    if (
        not isinstance(manifest.get("files"), dict)
        or manifest["files"].get("recipe.json") != recipe_sha
    ):
        raise ValueError("ACT local dataset recipe differs from its checkpoint manifest")
    expected = {
        "dataset_source": "local",
        "dataset_id": "firebird/local-" + snapshot.manifest_sha256[:16],
        "dataset_revision": snapshot.id,
        "dataset_manifest_sha256": snapshot.manifest_sha256,
        "dataset_lineage_validated": snapshot.lineage_validated,
    }
    if any(type(recipe.get(k)) is not type(v) or recipe.get(k) != v for k, v in expected.items()):
        raise ValueError("ACT local snapshot differs from the saved training recipe")
    claims = {
        "dataset_snapshot_id": snapshot.id,
        "dataset_manifest_sha256": snapshot.manifest_sha256,
    }
    # Earlier native local checkpoints contain the full dataset profile and recipe,
    # but not these additive outer fields. Their absence is not a different identity.
    if any(key in metadata and metadata[key] != value for key, value in claims.items()):
        raise ValueError("ACT local snapshot metadata differs from its saved training recipe")
    portable = {
        "source": "local",
        "snapshot_id": snapshot.id,
        "manifest_sha256": snapshot.manifest_sha256,
    }
    portable.update(
        {key: dataset[key] for key in ("repo_id", "revision") if dataset.get(key) is not None}
    )
    return portable, claims


def check_result(info, manifest, directory: Path, report, source, source_directory: Path) -> None:
    """Do not register an artifact with another source's proof or unsupported claims."""
    if info.get("format") != "inference_export":
        raise ValueError("ACT worker must publish a distinct inference export")
    metadata = manifest.get("metadata", {})
    checkpoint = source_directory / "checkpoint/manifest.json"
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    checkpoint_step = read_json(checkpoint).get("step")
    dataset, dataset_claims = dataset_lineage(source.metadata, source_directory)
    control = training_claims(source_directory, source.metadata)
    if policy_claims(directory / "policy", metadata) != control:
        raise ValueError("ACT export changed its simulator control contract")
    expected = {
        **control,
        **dataset_claims,
        "checkpoint_manifest_sha256": checkpoint_sha,
        "checkpoint_step": checkpoint_step,
        "dataset": dataset,
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
        canonical(metadata.get(key)) != canonical(value)
        or canonical(report.get(key)) != canonical(value)
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
    from .native_quantization import exact_json, temporal_info

    source_policy = source_directory / "checkpoint/pretrained_model"
    source_config = read_json(source_policy / "config.json")
    temporal = temporal_info(source_config)
    temporal_keys = {"prediction_horizon", "execution_horizon", "temporal_contract_sha256"}
    if (
        temporal_keys & metadata.keys()
        or (temporal["prediction_horizon"], temporal["execution_horizon"]) != (100, 100)
        or (source_directory / "checkpoint/temporal-contract.json").exists()
    ):
        exported_config = read_json(directory / "policy/config.json")
        if not exact_json(exported_config, source_config | {"use_vae": False}):
            raise ValueError("ACT export changed config beyond VAE removal")
        temporal = temporal_info(source_config)
        temporal_path = source_directory / "checkpoint/temporal-contract.json"
        if temporal_path.exists():
            original = temporal_path.read_bytes()
            if (directory / "policy/temporal-contract.json").read_bytes() != original:
                raise ValueError("ACT export changed temporal training provenance")
            temporal = temporal_info(
                source_config,
                directory / "policy",
                {"temporal-contract.json": {"sha256": hashlib.sha256(original).hexdigest()}},
            )
        if any(
            type(metadata.get(k)) is not type(v)
            or metadata.get(k) != v
            or type(report.get(k)) is not type(v)
            or report.get(k) != v
            for k, v in temporal.items()
        ):
            raise ValueError("ACT export temporal claims differ from saved checkpoint")
    if report.get("gpu_memory_bytes") is not None or report.get("inference_speedup") is not None:
        raise ValueError("ACT CPU export cannot claim a measured GPU improvement")
