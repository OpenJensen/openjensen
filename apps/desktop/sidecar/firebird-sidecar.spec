# Build specification only; not executed or accepted as a packaged artifact yet.
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

sidecar = Path(SPECPATH).resolve()
repository = sidecar.parents[2]
sys.path.insert(0, str(sidecar))
from sidecar_resources import load_resources

resources = load_resources(Path(os.environ["FIREBIRD_DESKTOP_RESOURCES"]))
a = Analysis(
    [str(sidecar / "entrypoint.py")],
    pathex=[str(sidecar), str(repository / "packages/core/src")],
    binaries=[],
    # Alembic and fixed isolated helpers use real paths relative to __file__.
    datas=[(str(resources.root / "resources.json"), "sidecar-resources"),
           (str(resources.web), "sidecar-resources/web")]
    + collect_data_files("vla_platform", include_py_files=True),
    hiddenimports=collect_submodules("vla_platform") + [
        "uvicorn.logging", "uvicorn.loops.asyncio", "uvicorn.protocols.http.h11_impl",
        "uvicorn.lifespan.on", "sqlalchemy.dialects.sqlite.aiosqlite", "aiosqlite",
    ],
    excludes=["torch", "torchvision", "textual"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name="firebird-sidecar", debug=False,
    bootloader_ignore_signals=False, strip=False, upx=False, console=True,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="firebird-sidecar")
