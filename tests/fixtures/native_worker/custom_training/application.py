"""Prove registered custom modules receive pinned training jobs in a subprocess."""

import hashlib
import json
import sys
from pathlib import Path

request, destination = map(Path, sys.argv[1:])
job = json.loads(request.read_text())
assert job["operation"] == "policy.finetune"
model = job["parameters"]["training"]
assert model["model_id"] == "openvla/openvla-7b-finetuned-libero-spatial"
assert model["model_revision"] == "962318cec55ac10993ff0f5f43eda9a270b4c873"
bundle = Path(job["output_dir"]) / "bundle"
bundle.mkdir()
payload = b"protocol fixture; not model weights"
(bundle / "fixture").write_bytes(payload)
(bundle / "manifest.json").write_text(
    json.dumps(
        {
            "files": {"fixture": hashlib.sha256(payload).hexdigest()},
            "metadata": {
                "fixture_only": True,
                "method": job["parameters"]["training_method"],
                "base_model": {
                    "repository": model["model_id"],
                    "revision": model["model_revision"],
                },
            },
        }
    )
)
destination.write_text(
    json.dumps(
        {
            "schema_version": 1,
            "job_id": job["job_id"],
            "artifact": {
                "path": str(bundle),
                "label": "Protocol fixture",
                "format": "training_checkpoint",
            },
            "report": {"scope": "protocol_fixture_only"},
        }
    )
)
