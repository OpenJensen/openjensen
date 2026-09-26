"""CLI client. Only `serve` owns the application; all other commands use its API."""

import json
import os
from typing import Annotated

import httpx
import typer
import uvicorn

app = typer.Typer(no_args_is_help=True, help="Local VLA projects and robotics dataset intake.")
projects = typer.Typer(no_args_is_help=True)
jobs = typer.Typer(no_args_is_help=True)
app.add_typer(projects, name="projects")
app.add_typer(jobs, name="jobs")


def call(method: str, path: str, payload: dict | None = None) -> None:
    base = os.getenv("FIREBIRD_API_URL", "http://127.0.0.1:8000").rstrip("/")
    try:
        response = httpx.request(method, base + "/api/v1" + path, json=payload, timeout=15)
        if response.is_error:
            typer.echo(f"API error ({response.status_code}): {response.text}", err=True)
            raise typer.Exit(1)
        typer.echo(json.dumps(response.json(), indent=2, ensure_ascii=False))
    except httpx.HTTPError as exc:
        typer.echo(f"Cannot reach the application. Start 'firebird serve'. {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command()
def serve(port: Annotated[int, typer.Option(min=1, max=65535)] = 8000) -> None:
    """Start one loopback application owner. Remote hosting is not enabled yet."""
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
) -> None:
    """Inspect public HF or allowed local LeRobot metadata, without downloading media."""
    if bool(repo_id) == bool(path):
        raise typer.BadParameter("Specify exactly one of --repo-id or --path")
    payload = (
        {"source": "huggingface", "repo_id": repo_id, "revision": revision}
        if repo_id
        else {
            "source": "local",
            "path": path,
        }
    )
    call("POST", f"/projects/{project_id}/intakes", payload)


@jobs.command("list")
def list_jobs(project_id: str) -> None:
    call("GET", f"/projects/{project_id}/jobs")


@jobs.command("show")
def show_job(job_id: str) -> None:
    call("GET", f"/jobs/{job_id}")


@jobs.command("cancel")
def cancel_job(job_id: str) -> None:
    call("POST", f"/jobs/{job_id}/cancel")


if __name__ == "__main__":
    app()
