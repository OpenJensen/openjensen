"""One-time operator setup; browser requests cannot choose executables or source URLs."""

import json
import os
import shutil
from pathlib import Path
from typing import Annotated

import typer

from vla_platform.update_runner import load_config


def configure_updates(
    deployment_root: Annotated[Path, typer.Option()],
    repository: Annotated[Path, typer.Option()],
    service: str = "firebird",
    port: int = 8096,
    node_bin: Annotated[Path | None, typer.Option()] = None,
    uv: Annotated[Path | None, typer.Option()] = None,
    pnpm: Annotated[Path | None, typer.Option()] = None,
):
    """Enable confirmed browser updates for a Supervisor release-directory installation."""
    root = deployment_root.expanduser().resolve()

    def executable(value, *names):
        selected = (
            str(value.expanduser().absolute())
            if value
            else next((found for name in names if (found := shutil.which(name))), None)
        )
        if not selected:
            raise typer.BadParameter("Required update tool is missing: " + "/".join(names))
        return selected

    node = executable(node_bin / "node" if node_bin else None, "node")
    config = dict(
        root=str(root),
        repository=str(repository.expanduser().resolve()),
        supervisorctl=str(root / ".supervisor/bin/supervisorctl"),
        supervisor_config=str(root / "supervisord.conf"),
        service=service,
        port=port,
        uv=executable(uv, "uv"),
        pnpm=executable(pnpm, "pnpm", "corepack"),
        node_bin=str(Path(node).parent),
    )
    private = root / "private"
    private.mkdir(mode=0o700, exist_ok=True)
    path = private / "application-updates.json"
    if path.exists() or path.is_symlink():
        raise typer.BadParameter(
            "Update configuration already exists; edit it privately on the server."
        )
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(config, handle, indent=2)
            handle.write("\n")
        load_config(path)
    except Exception:
        path.unlink()
        raise
    typer.echo("Saved " + str(path))
    typer.echo(
        "Add JENSEN_UPDATE_CONFIG=" + str(path) + " to the service environment and restart it."
    )
