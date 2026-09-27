"""CLI client. Only `serve` owns the application; all other commands use its API."""

import asyncio
import json
import os
import stat
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import typer

from vla_platform import cli_client
from vla_platform.tui_client import ApiError, segment

app = typer.Typer(
    no_args_is_help=True, help="Local robotics projects, lifecycle jobs and artifacts."
)
projects = typer.Typer(no_args_is_help=True)
jobs = typer.Typer(no_args_is_help=True)
policy = typer.Typer(no_args_is_help=True)
augmentation = typer.Typer(no_args_is_help=True)
app.add_typer(augmentation, name="augmentation")
app.add_typer(policy, name="policy")
app.add_typer(projects, name="projects")
app.add_typer(jobs, name="jobs")


def execute(operation):
    try:
        return asyncio.run(operation)
    except cli_client.WaitDeadline as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(124) from None
    except (ApiError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None
    except OSError, TimeoutError:
        typer.echo(
            "Local file or download operation failed; no completed download is claimed.", err=True
        )
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        typer.echo(
            "Client interrupted. No job cancellation was requested; inspect recorded jobs.",
            err=True,
        )
        raise typer.Exit(130) from None


def call(method: str, path: str, payload: dict | None = None) -> None:
    value = execute(cli_client.request_json(method, path, payload))
    typer.echo(json.dumps(value, indent=2, ensure_ascii=False))


@app.command()
def serve(port: Annotated[int, typer.Option(min=1, max=65535)] = 8000) -> None:
    """Start one loopback application owner. Remote hosting is not enabled yet."""
    import uvicorn

    uvicorn.run("vla_platform.api:create_app", factory=True, host="127.0.0.1", port=port, workers=1)


@app.command()
def capabilities() -> None:
    call("GET", "/capabilities")


@projects.command("list")
def list_projects() -> None:
    call("GET", "/projects")


@projects.command("create")
def create_project(name: str) -> None:
    call("POST", "/projects", {"name": name})


@app.command()
def inspect(
    project_id: str,
    repo_id: Annotated[str | None, typer.Option()] = None,
    revision: str = "main",
    path: Annotated[str | None, typer.Option()] = None,
    snapshot_for_training: Annotated[bool, typer.Option()] = False,
) -> None:
    """Inspect LeRobot metadata; optionally validate and freeze local training data/media."""
    if bool(repo_id) == bool(path):
        raise typer.BadParameter("Specify exactly one of --repo-id or --path")
    if snapshot_for_training and not path:
        raise typer.BadParameter("--snapshot-for-training requires --path")
    payload = (
        {"source": "huggingface", "repo_id": repo_id, "revision": revision}
        if repo_id
        else {
            "source": "local",
            "path": path,
            "snapshot_for_training": snapshot_for_training,
        }
    )
    call("POST", f"/projects/{segment(project_id)}/intakes", payload)


@jobs.command("list")
def list_jobs(project_id: str) -> None:
    call("GET", f"/projects/{segment(project_id)}/jobs")


@jobs.command("show")
def show_job(job_id: str) -> None:
    call("GET", f"/jobs/{segment(job_id)}")


@jobs.command("cancel")
def cancel_job(job_id: str) -> None:
    call("POST", f"/jobs/{segment(job_id)}/cancel")


@policy.command("options")
def policy_options() -> None:
    """Show configured execution targets, policy sources and fine-tuning methods."""
    call("GET", "/policy-options")


@augmentation.command("options")
def augmentation_options() -> None:
    """Show Gemini Omni setup and appearance presets."""
    call("GET", "/augmentation-options")


@augmentation.command("submit")
def submit_augmentation(project_id: str, recipe: Path) -> None:
    """Augment selected dataset clips through the shared application API."""
    call("POST", f"/projects/{segment(project_id)}/augmentations", read_recipe(recipe))


@policy.command("submit")
def submit_policy(project_id: str, recipe: Path) -> None:
    """Submit a policy operation/workflow through the shared application API."""
    call("POST", f"/projects/{segment(project_id)}/policy-jobs", read_recipe(recipe))


@policy.command("artifacts")
def policy_artifacts(project_id: str) -> None:
    call("GET", f"/projects/{segment(project_id)}/artifacts")


@jobs.command("events")
def job_events(job_id: str) -> None:
    call("GET", f"/jobs/{segment(job_id)}/events")


@jobs.command("wait")
def wait_for_job(
    job_id: str,
    timeout: Annotated[int, typer.Option(min=1, max=86400)] = 600,
    interval: Annotated[float, typer.Option(min=0.1, max=30)] = 2,
) -> None:
    """Observe an existing job until terminal; never submit, retry or cancel it."""
    job = execute(cli_client.wait_job(job_id, timeout, interval))
    typer.echo(json.dumps(job, indent=2, ensure_ascii=False))
    if job["status"] != "succeeded":
        raise typer.Exit(1)


@policy.command("download")
def download_policy(
    project_id: str,
    artifact_id: str,
    output: Annotated[
        Path, typer.Option(help="New destination file; existing files are never replaced.")
    ],
    max_bytes: Annotated[
        int, typer.Option(min=1, max=cli_client.MAX_DOWNLOAD_LIMIT)
    ] = cli_client.DEFAULT_DOWNLOAD_LIMIT,
    timeout: Annotated[int, typer.Option(min=1, max=3600)] = 600,
    expected_sha256: Annotated[
        str | None, typer.Option(help="Optional independently known archive SHA256.")
    ] = None,
) -> None:
    """Download registered artifact bytes with bounds and an explicit checksum receipt."""
    result = execute(
        cli_client.download_artifact(
            project_id, artifact_id, output, max_bytes, timeout, expected_sha256
        )
    )
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False))


def read_recipe(path: Path) -> dict:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(descriptor, "rb") as source:
            metadata = os.fstat(source.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 1024 * 1024:
                raise ValueError("Recipe must be a regular JSON file of at most 1 MiB")
            raw = source.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise ValueError("Recipe exceeds 1 MiB")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("Recipe must contain a JSON object")
        return value
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(f"Cannot read recipe: {exc}") from exc


def show_version(value: bool):
    if value:
        typer.echo(version("vla-platform"))
        raise typer.Exit()


@app.callback()
def main(
    version_flag: Annotated[
        bool, typer.Option("--version", callback=show_version, is_eager=True)
    ] = False,
):
    """One application API shared by web, terminal commands and the optional TUI."""


@app.command()
def tui(
    api_url: Annotated[
        str | None, typer.Option(help="Application URL; defaults to FIREBIRD_API_URL.")
    ] = None,
):
    """Open the keyboard workbench. Install with: uv sync --all-packages --extra tui."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        typer.echo(
            "The TUI needs an interactive terminal. Use projects/jobs/inspect for JSON output.",
            err=True,
        )
        raise typer.Exit(2)
    try:
        from vla_platform.tui import FirebirdApp
    except ModuleNotFoundError as exc:
        if exc.name != "textual":
            raise
        typer.echo("Install the optional client: uv sync --all-packages --extra tui", err=True)
        raise typer.Exit(2) from exc
    try:
        terminal = FirebirdApp(api_url or os.getenv("FIREBIRD_API_URL", "http://127.0.0.1:8000"))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    terminal.run()


if __name__ == "__main__":
    app()
