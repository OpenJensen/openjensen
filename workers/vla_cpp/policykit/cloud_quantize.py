"""Cloud entry point for a selected checkpoint to a verified packed GGUF bundle.

SkyPilot installs this beside the training worker and downloads input weights
on the cloud machine. Floating intermediates are private temporary files and
are removed before the cloud bootstrap publishes the quantized result.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

from . import application
from .inference_smoke import verify_cpu_inference
from .worker import atomic_json, describe_runtime, sha256


def progress(phase, message):
    print(
        json.dumps({"phase": phase, "message": message, "operation": "policy.quantize"}), flush=True
    )


def quantize_checkpoint(job):
    artifact = job.get("artifact")
    if not artifact or artifact.get("format") not in {
        "training_checkpoint",
        "native_checkpoint",
        "gguf",
    }:
        raise ValueError("Quantization requires a trained checkpoint or floating GGUF")
    source = Path(artifact["path"])
    manifest = application.verify(source)
    metadata = {**artifact.get("metadata", {}), **manifest.get("metadata", {})}
    if metadata.get("architecture", "smolvla") != "smolvla":
        raise ValueError("GGUF quantization currently supports SmolVLA checkpoints")
    vendor = Path(job["runtime"].get("conversion_vendor") or job["runtime"]["vendor"])
    # Validate the converter provenance before allocating/loading model weights.
    describe_runtime(vendor)
    original_id = artifact["id"]
    original_manifest = sha256(source / "manifest.json")
    output = Path(job["output_dir"])
    with tempfile.TemporaryDirectory(prefix=".floating-", dir=output) as temporary:
        intermediate = Path(temporary)
        if artifact["format"] == "training_checkpoint":
            from firebird_vla.export import export_checkpoint

            checkpoint = source / "checkpoint"
            # Individual saved checkpoints may be represented directly; final
            # training artifacts additionally include metrics and a wrapper.
            if not checkpoint.is_dir() and (source / "recipe.json").is_file():
                checkpoint = source
            application.verify(checkpoint)
            progress("exporting", "Merging the selected checkpoint's trained weights and adapters")
            native = intermediate / "native"
            native.mkdir()
            export_checkpoint(checkpoint, native / "policy")
            native_info = application.publish(
                native, metadata, "Native floating checkpoint", "native_checkpoint"
            )
            artifact = {**artifact, **native_info}
        if artifact["format"] == "native_checkpoint":
            progress("converting", "Converting the selected checkpoint to floating GGUF")
            conversion = intermediate / "gguf"
            conversion.mkdir()
            converted = application.import_policy(
                {**job, "artifact": artifact, "source": None, "output_dir": str(conversion)}
            )
            artifact = {**artifact, **converted["artifact"]}
        progress("quantizing", "Packing language weights and preserving the action expert")
        result = application.quantize_policy({**job, "artifact": artifact})
        bundle = Path(result["artifact"]["path"])
        final_manifest = application.verify(bundle)
        progress("verifying", "Running one native CPU prediction on fixed synthetic inputs")
        cameras = metadata.get("camera_keys") or ["synthetic-front", "synthetic-wrist"]
        smoke = verify_cpu_inference(
            bundle / "model.gguf",
            Path(job["runtime"]["smoke_build"]) / "tests/vla_predict_check",
            bundle,
            cameras=len(cameras),
        )
        # Bundle the exact model processors/config used for native export. The
        # GGUF embeds normalizer statistics, while these files document lineage.
        native_policy = intermediate / "native" / "policy"
        if not native_policy.exists() and job["artifact"]["format"] == "native_checkpoint":
            native_policy = source / "policy"
        if native_policy.is_dir():
            processor_dir = bundle / "processors"
            processor_dir.mkdir()
            for path in native_policy.iterdir():
                if (
                    path.is_file()
                    and path.name != "model.safetensors"
                    and path.suffix in {".json", ".safetensors"}
                    and path.stat().st_size <= 16 * 1024**2
                ):
                    shutil.copyfile(path, processor_dir / path.name)
        atomic_json(
            bundle / "checkpoint-lineage.json",
            {
                "source_artifact_id": original_id,
                "source_manifest_sha256": original_manifest,
                "native_inference_verified": True,
                "inference_scope": smoke["scope"],
                "source_format": job["artifact"]["format"],
                "checkpoint_step": metadata.get("step"),
                "base_model": metadata.get("base_model"),
                "dataset": metadata.get("dataset"),
            },
        )
        # The added provenance/processor files are covered by the final manifest.
        result["artifact"] = application.publish(
            bundle,
            {
                **final_manifest["metadata"],
                "source_artifact_id": original_id,
                "source_manifest_sha256": original_manifest,
                "native_inference_verified": True,
                "inference_scope": smoke["scope"],
            },
            result["artifact"]["label"],
        )
        result["report"].update(
            scope="quantization_and_native_inference",
            source_artifact_id=original_id,
            source_manifest_sha256=original_manifest,
            checkpoint_step=metadata.get("step"),
            quantized_file_bytes=(bundle / "model.gguf").stat().st_size,
            inference=smoke,
        )
        progress(
            "verifying", "Verified packed tensor precision, shape, hashes and checkpoint lineage"
        )
        return result


def main():
    request_path, result_path = map(Path, sys.argv[1:])
    job = json.loads(request_path.read_text())
    response = {"schema_version": 1, "job_id": job["job_id"]}
    try:
        if job.get("schema_version") != 1 or job.get("operation") != "policy.quantize":
            raise ValueError("Unsupported cloud quantization request")
        vendor = Path.cwd() / "vendor" / "vla.cpp"
        job["runtime"] = {
            **job.get("runtime", {}),
            "vendor": str(vendor),
            "device": "cuda",
            "smoke_build": str(Path.cwd() / "native-smoke"),
        }
        response.update(quantize_checkpoint(job))
    except Exception as exc:
        response["error"] = f"{type(exc).__name__}: {exc}"
    atomic_json(result_path, response)
    if response.get("error"):
        print(response["error"], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
