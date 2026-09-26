"""Expose deterministic worker startup/failure while reusing the protocol fixture."""

import json
import runpy
import sys
import time
from pathlib import Path

request_path, result_path = map(Path, sys.argv[1:])
request = json.loads(request_path.read_text())
if request["runtime"]["id"] == "browser-failure":
    result_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "job_id": request["job_id"],
                "error": "Intentional synthetic browser worker failure",
            }
        )
    )
    raise SystemExit(1)
if (
    request["runtime"]["id"] == "browser-delayed-success"
    and request["operation"] == "policy.import"
):
    # Reproduce a valid Windows startup exceeding Playwright's default 5s assertion.
    print(json.dumps({"step": 0}), flush=True)
    time.sleep(7)
if request["runtime"]["id"] == "browser-slow":
    # The core records this real subprocess output before the browser cancels.
    print(json.dumps({"step": 0}), flush=True)
runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "native_worker/policykit/application.py"),
    run_name="__main__",
)
