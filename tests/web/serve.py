"""Serve the production web export against a disposable browser-test workspace."""

import hashlib
import json
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
        port=8765,
        log_level="warning",
    )
