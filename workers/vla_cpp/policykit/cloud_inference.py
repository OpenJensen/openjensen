"""Measured native inference and package execution for GCS-backed SmolVLA artifacts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from . import application
from .worker import atomic_json, sha256


def progress(operation, phase, message):
    print(json.dumps({"operation": operation, "phase": phase, "message": message}), flush=True)


def inference_job(job):
    operation = job.get("operation")
    if operation not in {"policy.evaluate", "policy.run"}:
        raise ValueError("Unsupported cloud inference operation")
    artifact = job.get("artifact") or {}
    if artifact.get("format") not in {"gguf", "deployment_package"}:
        raise ValueError("Cloud inference requires a quantized GGUF or saved package")
    evaluation = job["parameters"]["evaluation"]
    if (
        evaluation.get("mode") != "engine"
        or evaluation.get("suite", "libero_object") != "libero_object"
    ):
        raise ValueError(
            "Cloud inference measures the native engine; simulator tasks are not configured"
        )
    source = Path(artifact["path"])
    manifest = application.verify(source)
    if manifest.get("metadata", {}).get("architecture", "smolvla") != "smolvla":
        raise ValueError("Cloud inference currently supports SmolVLA")
    original_sha = sha256(source / "manifest.json")
    progress(
        operation, "evaluating", "Loading the selected policy and measuring native CUDA inference"
    )
    if operation == "policy.evaluate":
        result = application.evaluate_policy(job)
    elif artifact["format"] == "deployment_package":
        progress(
            operation, "running", "Running this exact saved package in a fresh offline process"
        )
        result = {"report": application.evaluate_package(job, source)}
    else:
        progress(
            operation, "running", "Creating a package and running it in a fresh offline process"
        )
        result = application.run_policy(job)
    report = result["report"]
    if (
        report.get("fresh_reload_verified") is not True
        or report.get("runtime", {}).get("device") != "cuda"
        or report.get("memory_coverage", {}).get("complete") is not True
        or not report.get("samples_ms")
        or not report.get("finite_action_values")
    ):
        raise ValueError(
            "Native inference did not produce complete CUDA, reload, timing and memory evidence"
        )
    if application.verify(source) != manifest or sha256(source / "manifest.json") != original_sha:
        raise ValueError("Source artifact changed during cloud inference")
    report.update(
        scope="engine_diagnostics",
        task_success=None,
        success_rate=None,
        complete_episodes=0,
        source_artifact_id=artifact["id"],
        source_manifest_sha256=original_sha,
        execution_operation=operation,
        package_reexecuted=artifact["format"] == "deployment_package",
        deployment_verified=False,
        scope_note=(
            "Measured synthetic-input inference. No robot or simulator task-success evaluation."
        ),
    )
    atomic_json(Path(job["output_dir"]) / "inference-report.json", report)
    progress(
        operation,
        "verifying",
        "Verified finite actions, measured latency/memory and artifact identity",
    )
    return result


def main():
    request_path, result_path = map(Path, sys.argv[1:])
    job = json.loads(request_path.read_text())
    response = {"schema_version": 1, "job_id": job["job_id"]}
    try:
        if job.get("schema_version") != 1:
            raise ValueError("Unsupported cloud inference protocol")
        workdir = Path.cwd().resolve()
        job["runtime"] = {
            **job.get("runtime", {}),
            "vendor": str(workdir / "vendor/vla.cpp"),
            "build": str(workdir / "native-inference"),
            "worker_root": str(workdir / "quantization-worker"),
            "device": "cuda",
        }
        # Setup and run are separate SkyPilot shells. Keep shared CUDA libraries
        # discoverable for both this worker and its fresh inference subprocesses.
        cuda_lib = "/usr/local/cuda-12.8/lib64"
        os.environ["LD_LIBRARY_PATH"] = cuda_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        response.update(inference_job(job))
    except Exception as exc:
        response["error"] = f"{type(exc).__name__}: {exc}"
    atomic_json(result_path, response)
    if response.get("error"):
        print(response["error"], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
