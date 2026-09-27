"""Read-only JSON CLI. Explicit opt-in and one supervised subprocess per request."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from .contracts import (
    LICENSE,
    MAX_REQUEST_BYTES,
    DecisionError,
    canonical,
    parse_json,
    validate_request,
    validate_response,
)
from .integrity import read_file
from .supervisor import run_owned


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Optional Muose CPU text scorer (advisory, noncommercial license)."
    )
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument(
        "--request", required=True, type=Path, help="Bounded JSON request file; never a command."
    )
    parser.add_argument("--accept-license", choices=[LICENSE], required=True)
    parser.add_argument("--timeout-seconds", type=float, default=30)
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if not 0 < args.timeout_seconds <= 120:
            raise DecisionError("Timeout must be greater than zero and at most 120 seconds.")
        request = validate_request(parse_json(read_file(args.request, MAX_REQUEST_BYTES)))
        if args._child:
            from .scorer import DecisionScorer, offline_guard

            offline_guard()
            response = DecisionScorer(args.model_dir, args.accept_license).score(request)
        else:
            # Bind parsed input across process startup and subsequent source edits.
            with tempfile.TemporaryDirectory(prefix="firebird-decision-") as temporary:
                path = Path(temporary).resolve() / "request.json"
                path.write_bytes(canonical(request))
                command = [
                    sys.executable,
                    "-m",
                    "firebird_decision",
                    "--_child",
                    "--model-dir",
                    str(args.model_dir.absolute()),
                    "--request",
                    str(path),
                    "--accept-license",
                    LICENSE,
                ]
                raw = run_owned(command, timeout=args.timeout_seconds, env=os.environ.copy())
            response = parse_json(raw)
            validate_response(response, request)
        print(canonical(response).decode("utf-8"))
        return 0
    except DecisionError as exc:
        print(
            json.dumps({"schema_version": 1, "code": exc.code, "error": str(exc)}), file=sys.stderr
        )
        return 2
    except Exception:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "code": "worker_failed",
                    "error": "Decision worker failed; no result was accepted.",
                }
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
