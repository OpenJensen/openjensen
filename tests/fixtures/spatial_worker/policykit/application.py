"""Synthetic subprocess fixture for orchestration, never hardware/model evidence."""

import hashlib
import json
import sys
from pathlib import Path

request, destination = map(Path, sys.argv[1:])
job = json.loads(request.read_text())
out = Path(job["output_dir"])
op = job["operation"]
result = {"schema_version": 1, "job_id": job["job_id"], "report": {"scope": "spatial_test_fixture"}}


def sha(data):
    return hashlib.sha256(data).hexdigest()


if op in {"policy.import", "policy.quantize", "policy.run"}:
    bundle = out / "bundle"
    bundle.mkdir()
    if op == "policy.import":
        data = b"synthetic float" * 10
        metadata = {
            "precision": "float",
            "task": "libero_spatial",
            "fixture_only": True,
            "native_model_sha256": "a" * 64,
            "floating_model_sha256": sha(data),
        }
    else:
        source = Path(job["artifact"]["path"])
        metadata = json.loads((source / "manifest.json").read_text())["metadata"]
        data = (source / "model.gguf").read_bytes()
        if op == "policy.quantize":
            precision = job["parameters"]["precision"]
            metadata["precision"] = precision
            data = precision["language"].encode() * (10 if precision["language"] == "Q8_0" else 5)
    metadata.update(model_sha256=sha(data), weight_bytes=len(data))
    (bundle / "model.gguf").write_bytes(data)
    files = {"model.gguf": sha(data)}
    if op == "policy.run":
        receipt = {
            "manifest_sha256": sha((source / "manifest.json").read_bytes()),
            "files": json.loads((source / "manifest.json").read_text())["files"],
        }
        receipt_path = bundle / "tested-payload.json"
        receipt_path.write_text(json.dumps(receipt))
        files["tested-payload.json"] = sha(receipt_path.read_bytes())
        metadata["deployment_verified"] = True
    (bundle / "manifest.json").write_text(json.dumps({"files": files, "metadata": metadata}))
    result["artifact"] = {
        "path": str(bundle),
        "label": "Synthetic fixture",
        "format": "deployment_package" if op == "policy.run" else "gguf",
    }
if op in {"policy.evaluate", "policy.run"}:
    source = Path(job["artifact"]["path"])
    metadata = json.loads((source / "manifest.json").read_text())["metadata"]
    evaluation = job["parameters"]["evaluation"]
    states = sorted(evaluation["final_states"] if job["final"] else evaluation["initial_states"])
    tasks = sorted(evaluation["task_ids"])
    backend = job.get("evaluation_backend", "cpp")
    precision = metadata["precision"]
    configuration = (
        "native-bf16"
        if backend == "native-bf16"
        else (
            "cpp-bf16"
            if precision == "float"
            else "cpp-" + precision["language"] + ("-vision" if precision.get("vision") else "")
        )
    )
    protocol = {
        "suite": "libero_spatial",
        "task_ids": tasks,
        "state_ids": states,
        "seed": evaluation["seed"],
        "steps": evaluation["steps"],
        "action_steps": 50,
        "parity_limits": evaluation["parity_limits"],
        "fixture_sha256s": ["f" * 64],
        "warmups": evaluation["warmups"],
        "repetitions": evaluation["repetitions"],
        "inference_assets": {"fixture_only": True},
    }
    episodes = [
        {
            "task_id": task,
            "init_state_id": state,
            "seed": evaluation["seed"] + state,
            "noise_seed": evaluation["seed"] + 1000 * state,
            "status": "episode_complete",
            "task_success": True,
            "steps": 280,
            "benchmark_horizon": 280,
        }
        for task in tasks
        for state in states
    ]
    result["report"] = report = {
        "scope": "spatial_test_fixture",
        "backend": backend,
        "backend_configuration": configuration,
        "runtime": {"family": backend, "fixture_only": True},
        "target_identity": {
            "gpu_uuid": "fixture",
            "name": "synthetic GPU",
            "driver_version": "fixture",
        },
        "protocol": protocol,
        "protocol_sha256": sha(
            json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
        ),
        "model_sha256": sha((source / "model.gguf").read_bytes()),
        "artifact_manifest_sha256": sha((source / "manifest.json").read_bytes()),
        "native_model_sha256": metadata["native_model_sha256"],
        "source_native_model_sha256": metadata["native_model_sha256"],
        "p95_ms": 10,
        "peak_device_mib": 10,
        "memory_coverage": {"complete": True},
        "complete_episodes": len(episodes),
        "requested_episodes": len(episodes),
        "episodes": episodes,
        "success_rate": 1,
        "fixture_actions": [
            {"fixture_sha256": "f" * 64, "actions": [[0.0] * 7 for _ in range(50)]}
        ],
        "fixture_actions_deterministic": True,
        "fresh_reload_verified": True,
    }
    label = job["runtime"]["label"]
    scenario = json.loads(label.removeprefix("scenario:")) if label.startswith("scenario:") else {}
    fault = scenario.get(out.name, {})
    if "failures" in fault:
        for episode in episodes[: fault["failures"]]:
            episode["task_success"] = False
        report["success_rate"] = sum(ep["task_success"] for ep in episodes) / len(episodes)
    if fault.get("parity"):
        report["fixture_actions"][0]["actions"][0][0] = 0.1
    if fault.get("duplicate_episode"):
        episodes[-1] = episodes[0].copy()
    if fault.get("package_model"):
        changed = b"different package payload"
        metadata["model_sha256"] = sha(changed)
        (bundle / "model.gguf").write_bytes(changed)
        (bundle / "manifest.json").write_text(
            json.dumps({"files": {"model.gguf": sha(changed)}, "metadata": metadata})
        )
    for key, value in fault.get("fields", {}).items():
        report[key] = value
print(json.dumps({"step": 1}), flush=True)
destination.write_text(json.dumps(result, allow_nan=False))
