"""Protocol fixture only: never performs model inference or claims actual parity."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

request, result = map(Path, sys.argv[1:])
job = json.loads(request.read_text())
assert job["operation"] == "policy.export"
assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
if os.environ.get("FIREBIRD_TEST_EXPORT_HANG"):
    Path(os.environ["FIREBIRD_TEST_EXPORT_HANG"]).write_text(str(os.getpid()))
    time.sleep(300)
source = job["artifact"]
source_path = Path(source["path"])
root = Path(job["output_dir"]) / "inference-export"
policy = root / "policy"
policy.mkdir(parents=True)
config = json.loads((source_path / "checkpoint/pretrained_model/config.json").read_bytes())
(policy / "config.json").write_text(json.dumps(config | {"use_vae": False}))
(policy / "manifest.json").write_text('{"fixture_only":true}')
sha = hashlib.sha256((policy / "manifest.json").read_bytes()).hexdigest()
metadata = {
    "architecture": "act",
    "recipe": "act-vae-removal-fp32-v1",
    "precision": "float32",
    "inference_only": True,
    "training_resume_supported": False,
    "policy_subdirectory": "policy",
    "source_artifact_id": source["id"],
    "source_manifest_sha256": source["manifest_sha256"],
    "base_model": {
        "repository": "code://lerobot/act",
        "revision": "e595b7902714ba51f91e47523f66f89c5181b649",
    },
    "synthetic_parity_verified": True,
    "fresh_reload_verified": True,
    "task": "unverified",
    "task_success": None,
    "calibration_verified": False,
    "checkpoint_step": 1,
    "policy_manifest_sha256": sha,
    "fixture_only": True,
    "checkpoint_manifest_sha256": hashlib.sha256(
        (source_path / "checkpoint/manifest.json").read_bytes()
    ).hexdigest(),
    "dataset": {
        key: source["metadata"]["dataset"][key] for key in ("source", "repo_id", "revision")
    },
    "camera_keys": source["metadata"]["camera_keys"],
}
fault = os.getenv("FIXTURE_FAULT")
if fault == "source":
    metadata["source_artifact_id"] = "another-source"
if fault == "step":
    metadata["checkpoint_step"] = True
reload = {
    "status": "passed",
    "exact_equal": True,
    "source_reads_blocked": True,
    "manifest_sha256": sha,
}
if fault == "reload":
    reload["exact_equal"] = False
(root / "verification.json").write_text(json.dumps({"final_package_reload": reload}))
(root / "manifest.json").write_text(
    json.dumps(
        {
            "schema_version": 1,
            "metadata": metadata,
            "files": {
                p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob("*")
                if p.is_file()
            },
        }
    )
)
result.write_text(
    json.dumps(
        {
            "schema_version": 1,
            "job_id": job["job_id"],
            "artifact": None
            if fault == "no-artifact"
            else {
                "format": "native_checkpoint" if fault == "format" else "inference_export",
                "label": "ACT protocol fixture",
                "path": str(root),
            },
            "report": {
                **metadata,
                "gpu_memory_bytes": 1 if fault == "gpu" else None,
                "inference_speedup": None,
            },
        }
    )
)
