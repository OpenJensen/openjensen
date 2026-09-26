"""Deterministic subprocess protocol fixture, explicitly not hardware evidence."""

import hashlib
import json
import sys
import time
from pathlib import Path

request, destination = map(Path, sys.argv[1:])
job = json.loads(request.read_text())
out = Path(job["output_dir"])
op = job["operation"]
if job["runtime"]["label"] == "slow fixture" or (
    job["runtime"]["label"] == "slow evaluation fixture" and op == "policy.evaluate"
):
    (out / "started").write_text("fixture is running")
    time.sleep(60)
result = {"schema_version": 1, "job_id": job["job_id"], "report": {"scope": "test_fixture"}}
if op in {"policy.import", "policy.quantize", "policy.run"}:
    directory = out / "bundle"
    directory.mkdir()
    precision = "float" if op == "policy.import" else job["parameters"]["precision"]
    size = 100 if precision == "float" else 50 if precision["language"] == "Q8_0" else 25
    if (
        job["runtime"]["label"] == "failed q4 fixture"
        and precision != "float"
        and precision["language"] == "Q4_0"
    ):
        result["error"] = "Intentional candidate failure"
    else:
        (directory / "model").write_bytes(b"x" * size)
        metadata = {"precision": precision, "weight_bytes": size, "fixture_only": True}
        (directory / "manifest.json").write_text(
            json.dumps(
                {"files": {"model": hashlib.sha256(b"x" * size).hexdigest()}, "metadata": metadata}
            )
        )
        result["artifact"] = {
            "path": str(directory),
            "label": str(precision),
            "format": "deployment_package" if op == "policy.run" else "gguf",
        }
if op in {"policy.evaluate", "policy.run"}:
    evaluation = job["parameters"]["evaluation"]
    states = evaluation["final_states"] if job["final"] else evaluation["initial_states"]
    result["report"] = {
        "scope": "test_fixture",
        "p95_ms": 5,
        "peak_device_mib": 10,
        "runtime": {"fixture": True},
        "complete_episodes": len(states),
        "success_rate": 1,
    }
    if (
        job["runtime"]["label"] == "failed holdout fixture"
        and job["final"]
        and out.name != "final-reference"
    ):
        result["report"]["success_rate"] = 0
elif op not in {"policy.import", "policy.quantize"}:
    result["error"] = "Unsupported fixture operation"
print(json.dumps({"step": 1}), flush=True)
destination.write_text(json.dumps(result))
raise SystemExit(1 if result.get("error") else 0)
