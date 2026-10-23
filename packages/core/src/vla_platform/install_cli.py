"""Local-only profile installer: no API jobs, model downloads or hardware actions."""

import json
from pathlib import Path
from typing import Annotated

import typer

from vla_platform.install_profiles import (
    check_profile,
    environment_path,
    execute_install,
    install_plan,
    profile,
    public_profiles,
    source_root,
)


def install(
    name: Annotated[
        str, typer.Argument(help="list, core, robot, datasets, training or simulation")
    ] = "list",
    check: Annotated[
        bool, typer.Option(help="Read installed package metadata; change nothing.")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option(help="Print the exact commands and requirements; change nothing.")
    ] = False,
    source: Annotated[Path | None, typer.Option(help="OpenJensen source checkout.")] = None,
    environment: Annotated[
        Path | None, typer.Option(help="A separate optional-profile environment.")
    ] = None,
) -> None:
    """Install only the selected feature in its compatible Python environment."""
    try:
        if check and dry_run:
            raise ValueError("Choose either --check or --dry-run.")
        if name == "list":
            if check or dry_run or environment:
                raise ValueError(
                    "Choose a named profile when using --check, --dry-run or --environment."
                )
            result = {"default": "core", "profiles": public_profiles()}
        else:
            value, root = profile(name), source_root(source)
            if check:
                result = check_profile(value, root, environment)
            elif dry_run:
                result = {
                    "profile": value.id,
                    "name": value.name,
                    "python": value.python,
                    "platforms": value.platforms,
                    "environment": str(environment_path(value, root, environment)),
                    "external_tools": value.external,
                    "manual": value.manual,
                    "notes": value.notes,
                    "commands": install_plan(value, root, environment),
                    "minimum_free_gib": 10 + value.estimated_install_gib,
                }
            else:
                result = execute_install(value, root, environment)
        typer.echo(json.dumps(result, indent=2))
        if check and not result.get("ready"):
            raise typer.Exit(1)
        if result.get("installed") is False:
            raise typer.Exit(1)
    except (OSError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None
