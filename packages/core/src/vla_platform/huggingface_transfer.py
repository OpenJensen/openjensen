"""Publish a folder to Hugging Face, or fetch one back, from a pasted link.

This is how a dataset or model moves between machines that both run OpenJensen: publish
on one, paste the link on the other. The official client does the transfer in its own
process (``huggingface_transfer_worker.py``); this module only decides what to run.
"""

import asyncio
import json
import os
import re
import shutil
import signal
from pathlib import Path
from typing import Literal

from vla_platform.environment import getenv

# The one tested client version. ``uv`` fetches it into its own cache on first use.
CLIENT = "huggingface_hub==1.23.0"
WORKER = Path(__file__).with_name("huggingface_transfer_worker.py")
NAME = r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}"
REPOSITORY = re.compile(rf"{NAME}/{NAME}\Z")
REVISION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,199}\Z")
RepoType = Literal["dataset", "model"]


class TransferError(ValueError):
    def __init__(self, message: str, status: int = 422):
        self.status = status
        super().__init__(message)


def repository_url(repo_type: RepoType, repo_id: str) -> str:
    return "https://huggingface.co/" + ("datasets/" if repo_type == "dataset" else "") + repo_id


def slug(name: str) -> str:
    """A repository name people can read: ``Pick up the pen!`` -> ``pick-up-the-pen``."""
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip().lower()).strip("-._")
    return (value or "untitled")[:96]


def parse_reference(text: str, repo_type: RepoType) -> tuple[str, str | None]:
    """Accept what people paste: a page link, ``datasets/owner/name`` or ``owner/name``.

    Returns the repository and, when the link names one, its revision.
    """
    value = text.strip()
    value = re.sub(r"^(https?://)?(www\.)?(huggingface\.co|hf\.co)/", "", value, flags=re.I)
    value = value.split("?")[0].split("#")[0].strip("/")
    if value.startswith("spaces/") or (repo_type == "model" and value.startswith("datasets/")):
        raise TransferError(f"That link is not a Hugging Face {repo_type}.")
    value = value.removeprefix("datasets/" if repo_type == "dataset" else "models/")
    parts = value.split("/")
    repo, rest = "/".join(parts[:2]), parts[2:]
    revision = None
    if rest[:1] in (["tree"], ["blob"], ["resolve"], ["commit"]) and len(rest) > 1:
        revision = rest[1]
    elif rest:
        raise TransferError("Paste the repository page link or its owner/name.")
    if not REPOSITORY.fullmatch(repo) or any(part in (".", "..") for part in repo.split("/")):
        raise TransferError("Paste a Hugging Face link or an owner/name such as you/pick-pen.")
    if revision is not None and not REVISION.fullmatch(revision):
        raise TransferError("That link names an invalid revision.")
    return repo, revision


def command() -> list[str]:
    """The interpreter that has the Hub client. Never the application's own."""
    if configured := getenv("JENSEN_HUGGINGFACE_PYTHON"):
        return [configured]
    uv = shutil.which("uv")
    if uv is None:
        raise TransferError(
            "Hugging Face transfers need uv on the app host to fetch the official client.", 503
        )
    return [uv, "run", "--no-project", "--quiet", "--with", CLIENT, "python"]


async def call(request: dict, token: str | None, *, timeout: float = 3600, progress=None) -> dict:
    """Run the client once and return its answer. The token travels in its environment only.

    ``token=None`` leaves the client to use the login saved on this computer by ``hf``.
    ``progress`` receives each progress report the client prints before its answer.
    """
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_OFFLINE"}
    }
    if token:
        environment["HF_TOKEN"] = token
    environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    environment["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    try:
        process = await asyncio.create_subprocess_exec(
            *command(),
            str(WORKER),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=environment,
            limit=4 * 1024**2,
            start_new_session=os.name == "posix",
        )
    except OSError as exc:
        raise TransferError("The Hugging Face client could not be started.", 503) from exc
    result = None
    try:
        async with asyncio.timeout(timeout):
            try:
                process.stdin.write(json.dumps(request).encode())
                await process.stdin.drain()
            except BrokenPipeError, ConnectionResetError:
                pass  # The client stopped early; its answer, if any, says why.
            process.stdin.close()
            while line := await process.stdout.readline():
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                if isinstance(value, dict) and set(value) == {"progress"}:
                    if progress is not None and isinstance(value["progress"], dict):
                        progress(value["progress"])
                else:
                    result = value
            await process.wait()
    except TimeoutError as exc:
        raise TransferError("The Hugging Face transfer timed out.", 504) from exc
    except ValueError:
        result = None  # One line longer than any honest answer.
    finally:
        if process.returncode is None:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            await process.wait()
    if not isinstance(result, dict):
        raise TransferError(
            "The Hugging Face client returned no result. "
            "The first transfer downloads the client and needs a connection.",
            502,
        )
    if process.returncode or "error" in result:
        message = str(result.get("error") or "The Hugging Face transfer failed.")
        # Belt and braces: a client message must never carry the credential.
        if token and token in message:
            message = "The Hugging Face transfer failed."
        status = result.get("status")
        raise TransferError(message[:500], status if status in (401, 403, 404, 409) else 502)
    return result


async def run(request: dict, token: str | None, *, timeout: float = 3600, progress=None) -> dict:
    """Run one transfer and return where the folder now is."""
    result = await call(request, token, timeout=timeout, progress=progress)
    revision = result.get("revision")
    if result.get("repo_id") != request["repo_id"] or not re.fullmatch(
        r"[0-9a-f]{40}", str(revision)
    ):
        raise TransferError("The Hugging Face client returned an invalid result.", 502)
    return {
        "repo_id": request["repo_id"],
        "revision": revision,
        "private": bool(result.get("private")),
        "url": repository_url(request["repo_type"], request["repo_id"]),
    }


async def whoami(token: str | None) -> str:
    """The account a publish would go to: the given token's, or this computer's login."""
    result = await call({"operation": "whoami"}, token, timeout=60)
    name = result.get("username")
    if not isinstance(name, str) or not re.fullmatch(NAME, name):
        raise TransferError("Hugging Face returned an invalid account.", 502)
    return name


async def push(
    repo_type: RepoType,
    folder: Path,
    repo_id: str,
    token: str | None,
    *,
    private: bool = True,
    message: str = "Publish from OpenJensen",
    files: dict[str, str] | None = None,
    begin: dict[str, str] | None = None,
    marker: str | None = None,
    expect: list[str] | None = None,
    tag: str | None = None,
    progress=None,
) -> dict:
    """Publish ``folder``. ``token=None`` uses the login saved on this computer.

    ``begin`` files are written before the folder and ``files`` after it, so a
    repository whose publish was interrupted can be told from a finished one. A
    repository that already holds files but not ``marker`` belongs to something else
    and is refused. ``expect`` names the files that must be listed once it is done.
    """
    if not REPOSITORY.fullmatch(repo_id):
        raise TransferError("Use a repository name such as you/pick-pen.")
    if token is not None and not token:
        raise TransferError("Connect a Hugging Face account in Settings first.", 409)
    return await run(
        {
            "operation": "push",
            "repo_type": repo_type,
            "repo_id": repo_id,
            "folder": str(folder),
            "private": private,
            "message": message,
            "begin": begin or {},
            "files": files or {},
            "marker": marker,
            "expect": expect or [],
            "tag": tag,
        },
        token,
        progress=progress,
    )


async def pull(
    repo_type: RepoType,
    repo_id: str,
    folder: Path,
    token: str | None,
    *,
    revision: str | None = None,
    max_files: int = 4096,
    max_bytes: int = 2 * 1024**3,
    progress=None,
) -> dict:
    if not REPOSITORY.fullmatch(repo_id):
        raise TransferError("Use a repository name such as you/pick-pen.")
    return await run(
        {
            "operation": "pull",
            "repo_type": repo_type,
            "repo_id": repo_id,
            "folder": str(folder),
            "revision": revision,
            "max_files": max_files,
            "max_bytes": max_bytes,
        },
        token,
        progress=progress,
    )
