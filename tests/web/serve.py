"""Serve the production web export against a disposable browser-test workspace."""

import hashlib
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import uvicorn
from vla_platform.api import create_app
from vla_platform.settings import Settings

static_dir = Path(__file__).resolve().parents[2] / "apps" / "web" / "out"
if not (static_dir / "docs" / "index.html").is_file():
    raise SystemExit("Build the web app before browser tests: pnpm build:web")

with TemporaryDirectory(prefix="firebird-browser-tests-") as data_dir:
    workspace = Path(data_dir)
    source = workspace / "synthetic-source"
    source.write_bytes(b"synthetic browser fixture: not policy weights")
    runtime_config = workspace / "runtimes.json"
    runtime_config.write_text(
        json.dumps(
            {
                "runtimes": [
                    {
                        "id": ident,
                        "label": label,
                        "python": sys.executable,
                        "worker_root": str(
                            Path(__file__).resolve().parents[1] / "fixtures/browser_worker"
                        ),
                        "vendor": str(workspace),
                        "build": str(workspace),
                    }
                    for ident, label in [
                        ("browser-success", "Synthetic CPU fixture — not model evidence"),
                        (
                            "browser-delayed-success",
                            "Delayed synthetic fixture — not model evidence",
                        ),
                        ("browser-slow", "slow fixture"),
                        ("browser-failure", "Synthetic failure fixture"),
                    ]
                ],
                "sources": [
                    {
                        "id": "synthetic-source",
                        "label": "Synthetic source — not policy weights",
                        "path": str(source),
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    if os.getenv("FIREBIRD_BROWSER_NATIVE_FIXTURE") == "1":
        # Diagnostics use real protocol subprocesses, with no model or GPU.
        worker = Path(__file__).resolve().parents[1] / "fixtures/native_worker"
        runtime_config.write_text(
            json.dumps(
                {
                    "runtimes": [
                        {
                            "id": ident,
                            "label": label,
                            "python": sys.executable,
                            "worker_root": str(worker),
                            "vendor": data_dir,
                            "build": data_dir,
                            "simulator_lane": data_dir if simulator else None,
                        }
                        for ident, label, simulator in (
                            ("fixture", "Browser protocol fixture", True),
                            ("cpu-only", "CPU fixture without simulator", False),
                            ("slow", "slow evaluation fixture", True),
                        )
                    ],
                    "sources": [
                        {
                            "id": "fixture-source",
                            "label": "Synthetic browser fixture",
                            "path": str(source),
                            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
    elif os.getenv("FIREBIRD_BROWSER_EMPTY_RUNTIME") == "1":
        runtime_config.write_text('{"runtimes": [], "sources": []}', encoding="utf-8")
    uvicorn.run(
        create_app(
            Settings(
                data_dir=workspace,
                static_dir=static_dir,
                runtime_config=runtime_config,
                local_root=Path(__file__).resolve().parents[1] / "fixtures",
            )
        ),
        host="127.0.0.1",
        port=int(os.getenv("FIREBIRD_BROWSER_PORT", "8765")),
        log_level="warning",
    )
