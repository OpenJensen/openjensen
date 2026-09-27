"""Fixed application protocol for a registered ACT training checkpoint's CPU export."""

import argparse
import hashlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from .bundle import (
    JSON_LIMIT,
    canonical,
    inventory,
    publish_new_directory,
    read_json,
    safe_file,
    temporal_dimensions,
    verify_export,
)
from .control_schema import metadata as control_metadata
from .export import RECIPE, export_policy
from .training_source import TRAINING_REVISION, ActTrainingSource, admit_training_source

PROBE_TIMEOUT_SECONDS = 120


def _identity(value: Any, label: str, limit: int = 200) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"A bounded {label} is required")
    return value


def _paths(job: dict[str, Any]) -> tuple[Path, Path]:
    artifact = job.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError("A registered training checkpoint artifact is required")
    source = Path(_identity(artifact.get("path"), "artifact path", 4096))
    output = Path(_identity(job.get("output_dir"), "output directory", 4096))
    if not source.is_absolute() or not output.is_absolute():
        raise ValueError("Application paths must be absolute")
    if output.is_symlink() or output.is_junction():
        raise ValueError("Application output directory must not be a symlink")
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError("Export output must remain outside the training bundle")
    return source, output


def _selected_files(admission: ActTrainingSource) -> dict[str, dict[str, Any]]:
    return {name: {"sha256": sha, "bytes": size} for name, sha, size in admission.source_files}


def _checked_export(
    policy: Path, receipt: dict[str, Any], admission: ActTrainingSource
) -> tuple[dict[str, Any], str]:
    manifest = verify_export(policy)
    package_bytes = sum(item["bytes"] for item in inventory(policy).values())
    if type(receipt.get("package_bytes")) is not int or receipt["package_bytes"] != package_bytes:
        raise ValueError("ACT export receipt package size differs from the tested package")
    selected = _selected_files(admission)
    if admission.temporal_sha256 is not None:
        if (
            hashlib.sha256(safe_file(policy / "temporal-contract.json")).hexdigest()
            != admission.temporal_sha256
        ):
            raise ValueError("Exported temporal contract differs from registered source")
    control = control_metadata(policy)
    if (
        control.get("control_contract") != admission.control_contract
        or control.get("control_contract_sha256") != admission.control_contract_sha256
    ):
        raise ValueError("Exported simulator contract differs from registered source")
    parity = read_json(policy / "parity.json")
    recipe = read_json(policy / "recipe.json")
    if parity.get("source_checkpoint_files") != selected or recipe.get("source_files") != selected:
        raise ValueError("Exported source bytes differ from registered source admission")
    manifest_sha = hashlib.sha256(safe_file(policy / "manifest.json", JSON_LIMIT)).hexdigest()
    reload = receipt.get("final_package_reload")
    if (
        not isinstance(reload, dict)
        or reload.get("status") != "passed"
        or reload.get("exact_equal") is not True
        or reload.get("source_reads_blocked") is not True
        or reload.get("manifest_sha256") != manifest_sha
    ):
        raise ValueError("Complete ACT package reload proof is missing or mismatched")
    if any(receipt.get(key) != value for key, value in manifest.items()):
        raise ValueError("ACT export receipt differs from the exact tested manifest")
    if (
        parity.get("status") != "passed"
        or parity.get("exact_equal") is not True
        or parity.get("maximum_absolute_error") != 0.0
        or parity.get("calibration_verified") is not False
        or parity.get("task_success") is not None
    ):
        raise ValueError("ACT synthetic parity proof is incomplete")
    return manifest, manifest_sha


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    """Export only admitted local bytes, then atomically publish an app-owned envelope."""
    if type(job.get("schema_version")) is not int or job["schema_version"] != 1:
        raise ValueError("Unsupported application protocol")
    _identity(job.get("job_id"), "job identity")
    if job.get("operation") != "policy.export":
        raise ValueError("The ACT application worker supports only policy.export")
    source, output = _paths(job)
    artifact = job["artifact"]
    if artifact.get("format") != "training_checkpoint":
        raise ValueError("ACT export requires a registered training checkpoint")
    destination = output / "inference-export"
    if os.path.lexists(destination):
        raise FileExistsError("Refusing to replace an existing inference export")
    admission = admit_training_source(
        source,
        artifact_id=artifact.get("id"),
        manifest_sha256=artifact.get("manifest_sha256"),
    )
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".act-application-", dir=output) as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir()
        policy = bundle / "policy"
        # The exporter snapshots, rehashes and reloads the exact selected policy.
        # Bind its actual snapshot identities back to the prior full-bundle admission.
        receipt = export_policy(
            admission.source,
            policy,
            timeout=PROBE_TIMEOUT_SECONDS,
            **({"temporal_source": admission.temporal_source} if admission.temporal_source else {}),
        )
        _, manifest_sha = _checked_export(policy, receipt, admission)
        package_files = inventory(policy)
        current = admit_training_source(
            source,
            artifact_id=admission.artifact_id,
            manifest_sha256=admission.manifest_sha256,
        )
        if current != admission:
            raise ValueError("Training source changed before export publication")
        metadata = {
            "architecture": "act",
            **control_metadata(policy),
            **(
                {
                    "dataset_snapshot_id": admission.dataset_revision,
                    "dataset_manifest_sha256": admission.dataset_manifest_sha256,
                }
                if admission.dataset_manifest_sha256 is not None
                else {}
            ),
            **temporal_dimensions(read_json(policy / "config.json")),
            "temporal_contract_sha256": admission.temporal_sha256,
            "inference_only": True,
            "training_resume_supported": False,
            "precision": "float32",
            "recipe": RECIPE,
            "policy_subdirectory": "policy",
            "policy_manifest_sha256": manifest_sha,
            "source_artifact_id": admission.artifact_id,
            "source_manifest_sha256": admission.manifest_sha256,
            "checkpoint_manifest_sha256": admission.checkpoint_manifest_sha256,
            "checkpoint_step": admission.step,
            "base_model": {"repository": "code://lerobot/act", "revision": TRAINING_REVISION},
            "dataset": (
                {
                    "source": "local",
                    "snapshot_id": admission.dataset_revision,
                    "manifest_sha256": admission.dataset_manifest_sha256,
                    **{
                        k: admission.dataset_metadata[k]
                        for k in ("repo_id", "revision")
                        if isinstance(admission.dataset_metadata.get(k), str)
                    },
                }
                if admission.dataset_manifest_sha256 is not None
                else {
                    "source": "huggingface",
                    "repo_id": admission.dataset_id,
                    "revision": admission.dataset_revision,
                }
            ),
            "camera_keys": [admission.camera],
            "synthetic_parity_verified": True,
            "fresh_reload_verified": True,
            "task": "unverified",
            "task_success": None,
            "calibration_verified": False,
        }
        (bundle / "lineage.json").write_bytes(canonical(admission.receipt()))
        (bundle / "verification.json").write_bytes(canonical(receipt))
        files = {
            path.relative_to(bundle).as_posix(): hashlib.sha256(safe_file(path)).hexdigest()
            for path in sorted(bundle.rglob("*"))
            if path.is_file()
        }
        (bundle / "manifest.json").write_bytes(
            canonical({"schema_version": 1, "metadata": metadata, "files": files})
        )
        if inventory(policy) != package_files:
            raise ValueError("Tested ACT package changed while building its application envelope")
        for path in bundle.rglob("*"):
            if path.is_file():
                with path.open("r+b") as stream:
                    os.fsync(stream.fileno())
        publish_new_directory(bundle, destination)
    return {
        "artifact": {
            "path": str(destination),
            "label": "ACT inference export (FP32, VAE removed)",
            "format": "inference_export",
        },
        "report": {
            "scope": "ACT inference export and synthetic CPU parity",
            **metadata,
            "source_model_bytes": receipt["source_model_bytes"],
            "export_model_bytes": receipt["export_model_bytes"],
            "policy_package_bytes": receipt["package_bytes"],
            "source_read_protection": "Python open audit hook; not an OS filesystem sandbox",
            "gpu_memory_bytes": None,
            "inference_speedup": None,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args(argv)
    try:
        job = read_json(args.request)
        job_id = _identity(job.get("job_id"), "job identity")
        _, output = _paths(job)
        if args.result.parent.resolve() != output.resolve():
            raise ValueError("Result must be a new file in the application's output directory")
        if os.path.lexists(args.result):
            raise FileExistsError("Application result must be a new file")
        result: dict[str, Any] = {"schema_version": 1, "job_id": job_id}
        try:
            result.update(run_job(job))
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"[:2000]
        with args.result.open("xb") as stream:
            stream.write(canonical(result))
            stream.flush()
            os.fsync(stream.fileno())
        if "error" in result:
            print(result["error"], file=sys.stderr)
            return 1
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"ACT application export failed: {error}"[:2000], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
