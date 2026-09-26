"""Serve the production web export against a disposable browser-test workspace."""

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
    runtime_config = None
    if os.getenv("FIREBIRD_BROWSER_NATIVE_FIXTURE") == "1":
        # Test-only protocol subprocesses: no models, GPU, cloud or quality evidence.
        runtime_config = Path(data_dir) / "test-runtimes.json"
        worker = Path(__file__).resolve().parents[1] / "fixtures/native_worker"
        runtimes = []
        for name, label, simulator in (
            ("fixture", "Browser protocol fixture", True),
            ("cpu-only", "CPU fixture without simulator", False),
            ("slow", "slow evaluation fixture", True),
        ):
            runtimes.append(
                {
                    "id": name,
                    "label": label,
                    "python": sys.executable,
                    "worker_root": str(worker),
                    "vendor": data_dir,
                    "build": data_dir,
                    "simulator_lane": data_dir if simulator else None,
                }
            )
        runtime_config.write_text(
            json.dumps(
                {
                    "runtimes": runtimes,
                    "sources": [
                        {
                            "id": "fixture-source",
                            "label": "Synthetic browser fixture",
                            "path": str(Path(data_dir) / "source"),
                            "sha256": "0" * 64,
                        }
                    ],
                }
            )
        )
    uvicorn.run(
        create_app(
            Settings(data_dir=Path(data_dir), static_dir=static_dir, runtime_config=runtime_config)
        ),
        host="127.0.0.1",
        port=int(os.getenv("FIREBIRD_BROWSER_PORT", "8765")),
        log_level="warning",
    )
