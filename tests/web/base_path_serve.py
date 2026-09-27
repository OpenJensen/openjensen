"""Serve a built prefix-aware export against a disposable API for smoke checks."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import uvicorn
from starlette.applications import Starlette
from starlette.routing import Mount
from vla_platform.api import create_app
from vla_platform.settings import Settings

prefix = os.environ.get("NEXT_PUBLIC_BASE_PATH", "/firebird").rstrip("/")
static_dir = Path(
    os.environ.get(
        "FIREBIRD_TEST_WEB_DIR", str(Path(__file__).resolve().parents[2] / "apps" / "web" / "out")
    )
)
if not prefix or not (static_dir / "index.html").is_file():
    raise SystemExit(
        "Build with NEXT_PUBLIC_BASE_PATH=/firebird before running the prefix smoke test"
    )

with TemporaryDirectory(prefix="firebird-prefix-smoke-") as data_dir:
    application = create_app(Settings(data_dir=Path(data_dir), static_dir=static_dir))

    @asynccontextmanager
    async def lifespan(_):
        async with application.router.lifespan_context(application):
            yield

    uvicorn.run(
        Starlette(routes=[Mount(prefix, app=application)], lifespan=lifespan),
        host="127.0.0.1",
        port=int(os.environ.get("FIREBIRD_PREFIX_SMOKE_PORT", "18766")),
        log_level="warning",
    )
