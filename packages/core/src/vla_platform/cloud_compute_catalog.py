"""Reviewed single-GPU GCP targets for the bundled SmolVLA SkyPilot recipe.

GPU names and machine families follow Google's GPU documentation and SkyPilot's
GCP catalog. Regional offerings are checked with the installed SkyPilot catalog;
they are not a capacity, quota, or price guarantee.
"""

import asyncio
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from vla_platform.lifecycle.contracts import StrictRecord


@dataclass(frozen=True)
class GcpGpu:
    id: str
    label: str
    memory_mib: int
    instance_type: str | None
    unsupported_reason: str | None = None


GCP_GPUS = (
    GcpGpu("L4", "NVIDIA L4 · 24 GB", 24 * 1024, "g2-standard-4"),
    GcpGpu("T4", "NVIDIA T4 · 16 GB", 16 * 1024, "n1-highmem-4"),
    GcpGpu("A100", "NVIDIA A100 · 40 GB", 40 * 1024, "a2-highgpu-1g"),
    GcpGpu("A100-80GB", "NVIDIA A100 · 80 GB", 80 * 1024, "a2-ultragpu-1g"),
)
GCP_GPU_BY_ID = {gpu.id: gpu for gpu in GCP_GPUS}
RUNTIME_PREFIX = "skypilot-gcp-"
REQUIRED_GCP_SERVICES = {
    "compute.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "iam.googleapis.com",
}


class CloudGpuOption(StrictRecord):
    id: str
    label: str
    accelerator: str
    gpu_memory_mib: int
    gpu_count: int = 1
    instance_type: str | None
    supported: bool
    available: bool
    launchable: bool = False
    needs_preparation: bool = False
    unavailable_reason: str | None


class SetupCheckError(ValueError):
    """A setup failure whose message is safe to return to the browser."""


def sky_executable() -> str | None:
    executable = shutil.which("sky")
    fallback = Path.home() / ".local" / "bin" / "sky"
    return executable or (
        str(fallback) if fallback.is_file() and os.access(fallback, os.X_OK) else None
    )


def sky_python(executable: str) -> str | None:
    """Find the CLI environment for a read-only import check, without a shell."""
    script = Path(executable).resolve()
    try:
        with script.open("rb") as stream:
            line = stream.readline(4096).decode("utf-8")
        if line.startswith("#!"):
            command = shlex.split(line[2:].strip())
            if command and Path(command[0]).name == "env":
                command = command[1:]
            if len(command) == 1 and Path(command[0]).name.startswith("python"):
                return shutil.which(command[0])
    except OSError, UnicodeDecodeError, ValueError:
        pass
    # Windows console-script launchers keep Python next to their Scripts folder.
    candidate = script.parent.parent / "python.exe"
    return str(candidate) if candidate.is_file() else None


async def run_setup_command(
    executable: str,
    args: list[str],
    *,
    timeout: int = 30,
    env_overrides: dict[str, str] | None = None,
) -> bytes:
    """Execute a fixed setup command with bounded, sanitized process handling.

    Callers distinguish read-only checks from explicit user-requested preparation.
    Provider/CLI diagnostics may contain secrets and must not reach the browser.
    """
    try:
        process = await asyncio.create_subprocess_exec(
            executable,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env={
                **os.environ,
                "CLOUDSDK_CORE_DISABLE_PROMPTS": "1",
                "CLOUDSDK_COMPONENT_MANAGER_DISABLE_UPDATE_CHECK": "1",
                "SKYPILOT_DISABLE_USAGE_COLLECTION": "1",
                **(env_overrides or {}),
            },
            start_new_session=os.name == "posix",
        )
    except OSError:
        raise SetupCheckError("Could not start a required setup command.") from None
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout)
    except (TimeoutError, asyncio.CancelledError) as exc:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            elif process.returncode is None:
                process.kill()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(process.communicate(), timeout=3)
        except TimeoutError:
            pass
        if isinstance(exc, asyncio.CancelledError):
            raise
        raise SetupCheckError("The setup check timed out. Try again.") from None
    if process.returncode or len(stdout) > 4 * 1024 * 1024:
        raise SetupCheckError("A required setup check failed.")
    return stdout


async def run_readonly(
    executable: str,
    args: list[str],
    *,
    timeout: int = 30,
    env_overrides: dict[str, str] | None = None,
) -> bytes:
    return await run_setup_command(executable, args, timeout=timeout, env_overrides=env_overrides)


def safe_sky_endpoint(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 2048
        or any(not char.isprintable() or char.isspace() for char in value)
    ):
        raise SetupCheckError("SkyPilot returned an invalid API server address.")
    try:
        url = urlsplit(value)
        valid = (
            url.scheme in {"http", "https"}
            and bool(url.hostname)
            and url.username is None
            and url.password is None
            and not url.query
            and not url.fragment
        )
        _ = url.port
    except ValueError:
        valid = False
    if not valid:
        raise SetupCheckError("SkyPilot returned an invalid API server address.")
    return value.rstrip("/")


def sky_workspace_name(project_id: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", project_id):
        raise SetupCheckError("Choose a valid Google Cloud project in Settings.")
    return "firebird-gcp-" + hashlib.sha256(project_id.encode()).hexdigest()[:12]


# This script always reads through the SkyPilot server so its workspace config,
# not client environment overrides, determines the cloud project. Creation is a
# part of an explicitly requested training run; read-only checks never create it.
_WORKSPACE_SCRIPT = """
import json
import sys
import sky
from sky.client import sdk
from sky.server import common

name, project, mode = sys.argv[1:]
marker = 'FIREBIRD_SKY_WORKSPACE='

@common.check_server_healthy_or_start
def run():
    workspaces = sky.get(sdk.workspaces())
    if not isinstance(workspaces, dict):
        return 'error'
    if name not in workspaces:
        if mode != 'prepare':
            return 'missing'
        response = common.make_authenticated_request(
            'POST', '/workspaces/create',
            json={'workspace_name': name, 'config': {'gcp': {'project_id': project}}},
        )
        sky.get(common.get_request_id(response))
        workspaces = sky.get(sdk.workspaces())
    config = workspaces.get(name)
    if not isinstance(config, dict) or not isinstance(config.get('gcp'), dict):
        return 'mismatch'
    if config['gcp'].get('project_id') != project:
        return 'mismatch'
    if mode == 'prepare':
        checked = sky.get(sdk.check(infra_list=('gcp',), verbose=False, workspace=name))
        capabilities = checked.get(name, {}).get('GCP', []) if isinstance(checked, dict) else []
        if 'compute' not in capabilities:
            return 'compute_access'
    return 'ready'

reason, error_type = None, None
try:
    status = run()
except Exception as error:
    status = 'error'
    error_type = type(error).__name__
    text = str(error).lower()
    reason = ('permission' if any(word in text for word in ('permission', 'forbidden', '403'))
              else 'credentials' if any(word in text for word in ('credential', '401'))
              else 'server_unreachable' if any(word in text for word in ('connect', 'timed out'))
              else 'dependency' if isinstance(error, ImportError) else 'server_error')
record = {'status': status, 'workspace': name, 'endpoint': common.get_server_url(),
          'reason': reason, 'error_type': error_type}
print(marker + json.dumps(record))
"""


async def _sky_workspace(
    executable: str, project_id: str, *, prepare: bool, endpoint: str | None = None
) -> dict[str, str]:
    name = sky_workspace_name(project_id)
    python = sky_python(executable)
    if not python:
        raise SetupCheckError("Could not locate SkyPilot's Python environment. Reinstall SkyPilot.")
    runner = run_setup_command if prepare else run_readonly
    endpoint = safe_sky_endpoint(endpoint) if endpoint is not None else None
    try:
        output = await runner(
            python,
            ["-c", _WORKSPACE_SCRIPT, name, project_id, "prepare" if prepare else "verify"],
            timeout=120 if prepare else 30,
            env_overrides={"SKYPILOT_API_SERVER_ENDPOINT": endpoint} if endpoint else None,
        )
    except SetupCheckError:
        raise SetupCheckError(
            "Could not prepare the SkyPilot workspace. Check the SkyPilot server and retry."
            if prepare
            else "Could not verify the SkyPilot workspace. Check the SkyPilot server and retry."
        ) from None
    marker = b"FIREBIRD_SKY_WORKSPACE="
    records = [line[len(marker) :] for line in output.splitlines() if line.startswith(marker)]
    try:
        if len(records) != 1:
            raise ValueError("Ambiguous workspace response")
        result = json.loads(records[0])
        if not isinstance(result, dict) or result.get("workspace") != name:
            raise ValueError("Invalid workspace response")
    except ValueError, UnicodeDecodeError:
        raise SetupCheckError("SkyPilot returned an invalid workspace check response.") from None
    if result.get("status") == "missing":
        raise SetupCheckError(
            "Prepare a SkyPilot workspace in Settings to pin this Google Cloud project."
        )
    if result.get("status") == "mismatch":
        raise SetupCheckError(
            "The OPEN JENSEN SkyPilot workspace points to a different project. "
            "Resolve it in SkyPilot before continuing; OPEN JENSEN will not overwrite it."
        )
    if result.get("status") == "compute_access":
        raise SetupCheckError(
            "SkyPilot could not enable Google Cloud compute for this project. "
            "Check the connected account's compute and service-account permissions."
        )
    if result.get("status") != "ready":
        reason = result.get("reason")
        message = {
            "permission": "SkyPilot was denied permission to configure this workspace or project.",
            "credentials": "SkyPilot's server credentials expired. Reconnect Google Cloud.",
            "server_unreachable": "The configured SkyPilot server could not be reached.",
            "dependency": "SkyPilot's server is missing a required Google Cloud dependency.",
        }.get(reason)
        if not message:
            error_type = result.get("error_type")
            suffix = (
                f" ({error_type})"
                if isinstance(error_type, str)
                and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", error_type)
                else ""
            )
            message = (
                f"SkyPilot's workspace API failed{suffix}. "
                "Retry training or check the SkyPilot server."
            )
        raise SetupCheckError(message)
    actual_endpoint = safe_sky_endpoint(result.get("endpoint"))
    if endpoint is not None and actual_endpoint != endpoint:
        raise SetupCheckError("SkyPilot's API server differs from this run's saved server.")
    return {"workspace": name, "sky_api_endpoint": actual_endpoint}


async def verify_sky_target(
    executable: str, project_id: str, endpoint: str | None = None
) -> dict[str, str]:
    """Read and pin both workspace and server, rejecting a changed expected server."""
    return await _sky_workspace(executable, project_id, prepare=False, endpoint=endpoint)


async def verify_sky_workspace(executable: str, project_id: str) -> str:
    """Read the server's exact OPEN JENSEN workspace; never create or modify it."""
    return (await verify_sky_target(executable, project_id))["workspace"]


async def prepare_sky_workspace(executable: str, project_id: str) -> str:
    """Explicit Settings action: create only a missing project-pinned workspace.

    SkyPilot's supported workspace API preserves unrelated configuration and
    performs its own setup check, which can enable Google Cloud APIs. No machines
    are provisioned. Never invoke this as a check or implicit launch prerequisite.
    """
    return (await prepare_sky_target(executable, project_id))["workspace"]


async def prepare_sky_target(
    executable: str, project_id: str, endpoint: str | None = None
) -> dict[str, str]:
    return await _sky_workspace(executable, project_id, prepare=True, endpoint=endpoint)


async def configured_sky_endpoint(executable: str) -> str:
    """Read client configuration only; no server startup, auth, or cloud calls."""
    python = sky_python(executable)
    if not python:
        raise SetupCheckError("Could not locate SkyPilot's Python environment. Reinstall SkyPilot.")
    script = (
        "import json; from sky.server import common; "
        "print('FIREBIRD_SKY_ENDPOINT=' + json.dumps(common.get_server_url()))"
    )
    # Importing SkyPilot can exceed five seconds on a cold WSL filesystem.
    # This still reads configuration only; provider setup remains in the job.
    output = await run_readonly(python, ["-c", script], timeout=20)
    marker = b"FIREBIRD_SKY_ENDPOINT="
    records = [line[len(marker) :] for line in output.splitlines() if line.startswith(marker)]
    try:
        if len(records) != 1:
            raise ValueError("Ambiguous server address")
        return safe_sky_endpoint(json.loads(records[0]))
    except ValueError, UnicodeDecodeError:
        raise SetupCheckError("Could not read SkyPilot's configured API server address.") from None


def regional_gpu_offerings(payload: bytes, region: str) -> set[str]:
    try:
        catalog = json.loads(payload)
    except ValueError, UnicodeDecodeError:
        raise SetupCheckError("SkyPilot returned an invalid GPU catalog.") from None
    if not isinstance(catalog, dict):
        raise SetupCheckError("SkyPilot returned an invalid GPU catalog.")
    offerings = set()
    for gpu in GCP_GPUS:
        if not gpu.instance_type:
            continue
        entries = catalog.get(gpu.id, [])
        if not isinstance(entries, list):
            continue
        if any(
            isinstance(entry, dict)
            and str(entry.get("cloud", "")).lower() == "gcp"
            and entry.get("region") == region
            and entry.get("accelerator_name") == gpu.id
            and entry.get("accelerator_count") == 1
            and entry.get("instance_type") == gpu.instance_type
            for entry in entries
        ):
            offerings.add(gpu.id)
    return offerings
