"""Serve the production web export against a disposable browser-test workspace."""

from pathlib import Path
from tempfile import TemporaryDirectory

import uvicorn
from vla_platform.api import create_app
from vla_platform.settings import Settings

static_dir = Path(__file__).resolve().parents[2] / "apps" / "web" / "out"
if not (static_dir / "docs" / "index.html").is_file():
    raise SystemExit("Build the web app before browser tests: pnpm build:web")

with TemporaryDirectory(prefix="firebird-browser-tests-") as data_dir:
    uvicorn.run(
        create_app(Settings(data_dir=Path(data_dir), static_dir=static_dir)),
        host="127.0.0.1",
        port=8765,
        log_level="warning",
    )
