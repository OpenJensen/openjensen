"""Application protocol fixture. It contains no real weights or ML inference."""

import hashlib
import json
import sys
import time
from pathlib import Path

RUNTIME = {"lerobot": "0.6.1", "torch": "2.11.0", "torchvision": "0.26.0", "safetensors": "0.8.0"}
SCOPE = "generated observations; no calibration or task-quality acceptance"
UNITS = "saved processor output coordinates; physical units unverified"


def write(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def inventory(path):
    return {
        p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
        for p in path.iterdir()
    }


def package(job):
    source = Path(job["source"]["path"])
    config = json.loads((source / "config.json").read_text())
    fault = config.get("fixture_fault")
    if fault == "hang":
        (Path(job["output_dir"]) / "pid").write_text(str(__import__("os").getpid()))
        time.sleep(300)
    if fault == "stderr":
        print("specific offline failure: " + "x" * 20000, file=sys.stderr)
        raise SystemExit(2)
    if fault == "detached-child":
        import os
        import signal
        import subprocess

        child = subprocess.Popen(
            [sys.executable, "-c", "import time;time.sleep(300)"], start_new_session=True
        )
        interrupted = False

        def stop(*_):
            nonlocal interrupted
            interrupted = True

        signal.signal(signal.SIGTERM, stop)
        Path(job["output_dir"], "child-pid").write_text(str(child.pid))
        Path(job["output_dir"], "pid").write_text(str(os.getpid()))
        while not interrupted:
            time.sleep(0.01)
        Path(job["output_dir"], "cleaning").write_text("yes")
        time.sleep(5.2)  # Deliberately exceeds the legacy 5s wrapper grace.
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=1)
        Path(job["output_dir"], "child-reaped").write_text("yes")
        raise SystemExit(2)
    root = Path(job["output_dir"]) / "native-quantized"
    policy = root / "policy"
    policy.mkdir(parents=True)
    for name in (
        "config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "stats.safetensors",
    ):
        (policy / name).write_bytes((source / name).read_bytes())
    for name in ("control-contract.json", "temporal-contract.json"):
        if (source / name).is_file():
            (policy / name).write_bytes((source / name).read_bytes())
    (policy / "model.fbq").write_bytes(b"fixture packed payload, never executable")
    bits = job["native_quantization"]["bits"]
    write(
        policy / "encoding.json",
        {
            "schema_version": 1,
            "format": "firebird_quant",
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
            "runtime": RUNTIME,
        },
    )
    if fault in {"encoding-bool", "encoding-float"}:
        value = json.loads((policy / "encoding.json").read_text())
        if fault == "encoding-bool":
            value["schema_version"] = True
        else:
            value["recipe"]["group_size"] = 64.0
        write(policy / "encoding.json", value)
    files = inventory(policy)
    digest = hashlib.sha256(b"firebird-native-packed-policy-v1\0")
    for name, item in sorted(files.items()):
        encoded = name.encode()
        digest.update(
            len(encoded).to_bytes(8, "big")
            + encoded
            + item["bytes"].to_bytes(8, "big")
            + bytes.fromhex(item["sha256"])
        )
    model_id = "sha256:" + digest.hexdigest()
    metadata = {
        "architecture": "act",
        "format": "firebird_quant",
        "format_version": 1,
        "model_id": model_id,
        "precision": f"int{bits}",
        "policy_subdirectory": "policy",
        "inference_only": True,
        "training_resume_supported": False,
        "fresh_reload_verified": True,
        "cpu_reload_verified": True,
        "runtime_verified": False,
        "isaac_runtime_verified": False,
        "quality_verified": False,
        "calibration_verified": False,
        "speedup_verified": False,
        "task_success": None,
        "source_artifact_id": job["source"]["artifact_id"],
        "source_artifact_manifest_sha256": job["source"]["artifact_manifest_sha256"],
    }
    prediction, execution = config["chunk_size"], config["n_action_steps"]
    claims = {}
    if (source / "control-contract.json").is_file():
        raw = (source / "control-contract.json").read_bytes()
        claims.update(
            control_contract=json.loads(raw),
            control_contract_sha256=hashlib.sha256(raw).hexdigest(),
        )
    if (
        claims
        or (prediction, execution) != (100, 100)
        or (source / "temporal-contract.json").is_file()
    ):
        metadata.update(
            prediction_horizon=prediction,
            execution_horizon=execution,
            temporal_contract_sha256=(
                hashlib.sha256((source / "temporal-contract.json").read_bytes()).hexdigest()
                if (source / "temporal-contract.json").is_file()
                else None
            ),
        )
    metadata.update(claims)
    chunks = [
        {
            "seed": seed,
            "input_sha256": str(index) * 64,
            "image_shape": [3, 32, 32],
            "raw": [[0.0] * 6 for _ in range(prediction)],
            "postprocessed": [[0.0] * 6 for _ in range(prediction)],
            "queue_and_reset_exact": True,
        }
        for index, seed in enumerate((171, 902))
    ]
    drift = [
        {
            "seed": row["seed"],
            "input_sha256": row["input_sha256"],
            **{
                key: {
                    "rmse": 0.0,
                    "maximum_absolute_difference": 0.0,
                    "coordinates": prediction * 6,
                }
                for key in ("raw", "postprocessed")
            },
        }
        for row in chunks
    ]
    proof = {
        "schema_version": 1,
        "model_id": model_id,
        "scope": SCOPE,
        "units": UNITS,
        "versions": RUNTIME,
        "floating": chunks,
        "packed": chunks,
        "drift_from_fp32": drift,
        "fresh_packed_reload_exact": True,
        "full_chunk_queue_reset_verified": True,
        "floating_master_reads_blocked": True,
        "network_disabled": True,
        "source_read_protection": "Python open audit hook; not an OS filesystem sandbox",
        "quality_verified": False,
        "calibration_verified": False,
        "speedup_verified": False,
        "task_success": None,
        "gpu_memory_bytes": None,
        "inference_speedup": None,
    }
    proof.update(claims)
    if "prediction_horizon" in metadata:
        proof.update(prediction_horizon=prediction, execution_horizon=execution)
    lineage = {
        "schema_version": 1,
        "source_artifact_id": job["source"]["artifact_id"],
        "source_artifact_manifest_sha256": job["source"]["artifact_manifest_sha256"],
        "source_files": job["source"]["files"],
        "source_manifest_sha256": job["source"]["manifest_sha256"],
        "source_registry_binding": "caller-owned; worker verifies supplied exact bytes",
    }
    if fault == "lineage-bool":
        lineage["schema_version"] = True
    if fault == "lineage":
        lineage["source_artifact_id"] = "another"
    if fault == "incomplete":
        proof["packed"][0]["raw"].pop()
    if fault == "quality":
        metadata["quality_verified"] = True
    if fault == "reload":
        proof["fresh_packed_reload_exact"] = False
    if fault == "source":
        (source / "model.safetensors").write_bytes(b"modified")
    write(root / "verification.json", proof)
    write(root / "lineage.json", lineage)
    if fault == "float-master":
        (policy / "model.safetensors").write_bytes(b"unexpected")
    write(
        root / "manifest.json",
        {
            "schema_version": 1,
            "metadata": metadata,
            "files": {
                p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob("*")
                if p.is_file()
            },
        },
    )
    response = {
        "schema_version": 1,
        "job_id": job["job_id"],
        "operation": "policy.quantize",
        "artifact": {"format": "native_quantized", "path": str(root), "label": "Protocol fixture"},
        "report": {
            **metadata,
            "drift_from_fp32": drift,
            "fixture_scope": SCOPE,
            "units": UNITS,
            "source_weight_bytes": job["source"]["files"]["model.safetensors"]["bytes"],
            "packed_weight_bytes": files["model.fbq"]["bytes"],
            "policy_package_bytes": sum(row["bytes"] for row in files.values()),
            "gpu_memory_bytes": None,
            "inference_speedup": None,
        },
    }
    if fault == "identity":
        response["job_id"] = "other-job"
    if fault == "size":
        response["report"]["packed_weight_bytes"] += 1
    if fault == "drift":
        response["report"]["drift_from_fp32"][0]["raw"]["rmse"] = 1.0
    if fault == "model":
        response["report"]["model_id"] = "sha256:" + "a" * 64
    return response


if __name__ == "__main__":
    request, result = map(Path, sys.argv[1:])
    write(result, package(json.loads(request.read_text())))
