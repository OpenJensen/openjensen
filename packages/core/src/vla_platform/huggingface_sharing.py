"""Share a saved dataset or a trained model through Hugging Face.

Publishing sends the files exactly as they are saved here, beside one small file,
``openjensen-share.json``, that lists them with their hashes. Adding from a link fetches
the repository, checks every file against that list and saves the result: the same
dataset, or the same model, as on the machine that published it.

Each publish or fetch is a job with its own saved record, shown with the other jobs.
The access token never enters a record. It comes from Settings, or from the login
``hf auth login`` saved on this computer, and is handed to the client process only.
"""

import asyncio
import hashlib
import json
import logging
import os
import queue
import re
import shutil
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from vla_platform import build
from vla_platform import huggingface_transfer as transfer
from vla_platform.contracts import Timestamp, now
from vla_platform.datasets.library import MAX_BYTES as DATASET_BYTES
from vla_platform.datasets.library import MAX_FILES as DATASET_FILES
from vla_platform.datasets.library import LibraryError
from vla_platform.huggingface_transfer import TransferError
from vla_platform.job_observability import JobObservability, redact
from vla_platform.lifecycle.contracts import PolicyArtifact

LOGGER = logging.getLogger(__name__)
MARKER = "openjensen-share.json"
IDENTITY = r"^[a-f0-9]{32}$"
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}
DISK_HEADROOM = 1024**3
MODEL_BYTES = 2 * 1024**3
# A transfer is a job that reports where it is and can be cancelled, so it may take as
# long as a slow connection needs: 2 GiB up at 0.1 MB/s is close to six hours.
TRANSFER_SECONDS = 6 * 3600
HEARTBEAT_SECONDS = 30
MODEL_FILES = 4096
MAX_RECORD = 1024**2
MAX_MARKER = 8 * 1024**2
Kind = Literal["dataset", "model"]
NOT_CONNECTED = (
    "Connect a Hugging Face account first: save a token in Settings, "
    "or run hf auth login on this computer."
)


def megabytes(size: int) -> str:
    return f"{size / 1024**3:.2f} GB" if size >= 1024**3 else f"{size / 1024**2:.1f} MB"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(root: Path):
    """Every file under ``root`` with its size and hash, and one hash over that list.

    The list has the order and shape of the dataset reader's own inventory, so the
    hash over it is the identity a saved dataset records for the same files.
    """
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("A dataset that contains links cannot be published")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("Only regular dataset files can be published")
        files.append([path.relative_to(root).as_posix(), path.stat().st_size, digest(path)])
    identity = hashlib.sha256(json.dumps(files, separators=(",", ":")).encode()).hexdigest()
    return identity, files


def machine_token_path() -> Path:
    """Where ``hf auth login`` keeps this computer's login. Only its presence is checked."""
    if configured := os.environ.get("HF_TOKEN_PATH"):
        return Path(configured)
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "token"
    cache = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(cache) / "huggingface" / "token"


class ShareEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timestamp: Timestamp
    stage: str
    message: str


class ShareJob(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal[1] = 1
    id: str = Field(pattern=IDENTITY)
    project_id: str = Field(min_length=1, max_length=128)
    operation: Literal["dataset.publish", "dataset.fetch", "policy.publish", "policy.fetch"]
    title: str = Field(min_length=1, max_length=200)
    status: Literal["running", "succeeded", "failed", "cancelled", "interrupted"]
    stage: str
    message: str | None = None
    error: str | None = None
    inputs: dict = Field(default_factory=dict)
    result: dict | None = None
    events: list[ShareEvent] = Field(default_factory=list)
    created_at: Timestamp
    updated_at: Timestamp


class ShareJobs:
    """One small file per publish or fetch, written whole each time it changes."""

    def __init__(self, data_dir):
        self.directory = Path(data_dir) / "huggingface-jobs"
        self.observability = JobObservability(Path(data_dir))

    def path(self, ident):
        if not isinstance(ident, str) or not re.fullmatch(IDENTITY, ident):
            raise ValueError("Sharing job identity is invalid")
        path = self.directory / f"{ident}.json"
        if self.directory.is_symlink() or path.is_symlink():
            raise ValueError("Sharing job storage needs administrator attention")
        return path

    def get(self, ident):
        path = self.path(ident)
        if not path.is_file() or path.stat().st_size > MAX_RECORD:
            raise ValueError("Sharing job was not found")
        record = ShareJob.model_validate_json(path.read_bytes())
        if record.id != ident:
            raise ValueError("Sharing job identity differs from its saved record")
        return record.model_dump(mode="json")

    def records(self, project_id=None):
        if not self.directory.is_dir() or self.directory.is_symlink():
            return []
        rows = []
        for path in self.directory.glob("*.json"):
            try:
                value = self.get(path.stem)
            except OSError, ValueError:
                continue  # One damaged record does not hide the others.
            if project_id is None or value["project_id"] == project_id:
                rows.append(value)
        return sorted(rows, key=lambda row: (row["created_at"], row["id"]), reverse=True)

    def save(self, value):
        value["updated_at"] = now()
        record = ShareJob.model_validate(value)
        path = self.path(record.id)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        encoded = record.model_dump_json().encode() + b"\n"
        if len(encoded) > MAX_RECORD:
            raise ValueError("Sharing job record exceeds its supported size")
        descriptor, temporary = tempfile.mkstemp(prefix=".share-", dir=self.directory)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return record.model_dump(mode="json")

    def create(self, project_id, operation, title, inputs):
        value = {
            "id": uuid4().hex,
            "project_id": project_id,
            "operation": operation,
            "title": title[:200],
            "status": "running",
            "stage": "starting",
            "inputs": inputs,
            "created_at": now(),
        }
        saved = self.save(value)
        self.observability.try_record(
            "share:" + saved["id"],
            "Job accepted",
            component="huggingface",
            stage="starting",
            data={"operation": operation},
        )
        return saved

    def progress(self, ident, stage, message, *, event=True):
        value = self.get(ident)
        if value["status"] in TERMINAL:
            return value
        value.update(stage=stage, message=message)
        if event:
            value["events"] = [
                *value["events"][-199:],
                {"timestamp": now(), "stage": stage, "message": message},
            ]
            self.observability.try_record(
                "share:" + ident, message, component="huggingface", stage=stage
            )
        return self.save(value)

    def finish(self, ident, status, *, result=None, error=None):
        value = self.get(ident)
        if value["status"] in TERMINAL:
            return value
        message = redact(error)[:1000] if error else "Done"
        value.update(
            status=status,
            stage=status,
            error=message if error else None,
            message=None,
            result=result,
        )
        value["events"] = [
            *value["events"][-199:],
            {"timestamp": now(), "stage": status, "message": message},
        ]
        saved = self.save(value)
        self.observability.try_record(
            "share:" + ident,
            message,
            component="huggingface",
            stage=status,
            severity="error" if status == "failed" else "info",
            data={"status": status},
        )
        if status == "failed":
            self.observability.try_failure(
                "share:" + ident,
                stage=value["events"][-2]["stage"] if len(value["events"]) > 1 else "starting",
                message=message,
                code="HUGGINGFACE_TRANSFER_FAILED",
            )
        return saved


class _QueueReader:
    """A file for ``tarfile`` to read, fed one chunk at a time from the event loop."""

    def __init__(self, chunks):
        self.chunks, self.buffer, self.ended = chunks, b"", False

    def read(self, size=-1):
        while not self.ended and (size < 0 or len(self.buffer) < size):
            chunk = self.chunks.get()
            if chunk is None:
                self.ended = True
            else:
                self.buffer += chunk
        if size < 0:
            size = len(self.buffer)
        value, self.buffer = self.buffer[:size], self.buffer[size:]
        return value


def _extract(reader, destination: Path, limit: int):
    total = 0
    with tarfile.open(fileobj=reader, mode="r|") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if member.isdir():
                continue
            if (
                not member.isfile()
                or len(name.parts) < 2
                or name.parts[0] != "policy"
                or ".." in name.parts
                or name.is_absolute()
                or "\\" in member.name
            ):
                raise ValueError("The stored model archive has an unexpected layout")
            total += member.size
            if total > limit:
                raise ValueError("The stored model is larger than its registered size")
            target = destination.joinpath(*name.parts[1:])
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as stream, target.open("xb") as output:
                shutil.copyfileobj(stream, output, 1024**2)
    # Drain what follows the end marker so the sender is never left blocked.
    while reader.read(1024**2):
        pass


async def unpack(source, destination: Path, limit: int, received=None):
    """Write a ``policy/`` archive stream out as a folder, without keeping the archive.

    ``received`` is told how many bytes have arrived so far, after every piece.
    """
    chunks: queue.Queue = queue.Queue(maxsize=8)
    destination.mkdir(parents=True)
    worker = asyncio.create_task(
        asyncio.to_thread(_extract, _QueueReader(chunks), destination, limit)
    )

    async def send(chunk):
        while not worker.done():
            try:
                chunks.put_nowait(chunk)
                return True
            except queue.Full:
                await asyncio.sleep(0.01)
        return False

    failed, arrived = True, 0
    try:
        async for chunk in source:
            if not await send(chunk):
                break
            arrived += len(chunk)
            if received is not None:
                received(arrived)
        failed = False
    finally:
        closing = getattr(source, "aclose", None)
        if closing is not None:
            await closing()
        await send(None)  # The reader always sees the end, so its thread always finishes.
        if failed:
            await asyncio.gather(worker, return_exceptions=True)
    await worker


def _remove_empty_directories(root: Path):
    for path in sorted((item for item in root.rglob("*") if item.is_dir()), reverse=True):
        if not any(path.iterdir()):
            path.rmdir()


def read_marker(root: Path, kind: Kind):
    """What the publishing machine said this repository holds, or nothing."""
    path = root / MARKER
    if path.is_symlink() or not path.is_file():
        return None
    if path.stat().st_size > MAX_MARKER:
        raise ValueError("This repository's OpenJensen record is too large to trust")
    try:
        marker = json.loads(path.read_bytes())
    except ValueError:
        marker = None
    if (
        not isinstance(marker, dict)
        or marker.get("schema_version") != 1
        or marker.get("kind") not in ("dataset", "model")
    ):
        raise ValueError("This repository's OpenJensen record cannot be read")
    if marker["kind"] != kind:
        raise ValueError(
            f"That link is a {marker['kind']} published from OpenJensen, not a {kind}. "
            f"Add it on the {'Datasets' if marker['kind'] == 'dataset' else 'Models'} page."
        )
    if marker.get("state") != "published":
        raise ValueError(
            "Publishing this repository did not finish. "
            "Publish it again on the machine that has the original."
        )
    return marker


def settle_dataset(root: Path):
    """Reduce a fetched dataset repository to exactly the files that were published."""
    shutil.rmtree(root / ".cache", ignore_errors=True)
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("The fetched repository contains links, which are not accepted")
    marker = read_marker(root, "dataset")
    if marker is None:
        (root / ".gitattributes").unlink(missing_ok=True)
        return None
    listed = marker.get("files")
    if (
        not isinstance(listed, dict)
        or not listed
        or not isinstance(marker.get("sha256"), str)
        or any(
            not isinstance(item, dict) or type(item.get("size")) is not int
            for item in listed.values()
        )
    ):
        raise ValueError("This repository's OpenJensen record cannot be read")
    for path in list(root.rglob("*")):
        if path.is_file() and path.relative_to(root).as_posix() not in listed:
            path.unlink()
    _remove_empty_directories(root)
    for name, expected in listed.items():
        path = root / name
        if (
            PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts
            or not path.is_file()
            or path.stat().st_size != expected["size"]
            or digest(path) != expected.get("sha256")
        ):
            raise ValueError(
                f"The fetched dataset differs from what was published ({name}). "
                "Publish it again, then add it from the link."
            )
    return marker


def settle_model(root: Path):
    """Reduce a fetched model repository to its bundle and check every recorded hash."""
    from vla_platform.lifecycle.cloud_storage import checked_files
    from vla_platform.lifecycle.service import validate_bundle

    shutil.rmtree(root / ".cache", ignore_errors=True)
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("The fetched repository contains links, which are not accepted")
    marker = read_marker(root, "model")
    manifest_path = root / "manifest.json"
    if marker is None or not manifest_path.is_file():
        raise ValueError(
            "This repository was not published from OpenJensen, so it cannot be added "
            "as a model here. Use Import a model for other policies."
        )
    if manifest_path.stat().st_size > MAX_MARKER:
        raise ValueError("The fetched model's manifest is too large to trust")
    try:
        listed = checked_files(json.loads(manifest_path.read_bytes()))
    except ValueError:
        raise ValueError("The fetched model's manifest cannot be read") from None
    for path in list(root.rglob("*")):
        name = path.relative_to(root).as_posix()
        if path.is_file() and name != "manifest.json" and name not in listed:
            path.unlink()
    _remove_empty_directories(root)
    try:
        manifest, sha, total = validate_bundle(root, root.parent)
    except ValueError as exc:
        raise ValueError(
            f"The fetched model differs from what was published ({exc}). "
            "Publish it again, then add it from the link."
        ) from None
    details = marker.get("model")
    if (
        sha != marker.get("manifest_sha256")
        or not isinstance(details, dict)
        or not isinstance(manifest.get("metadata", {}), dict)
    ):
        raise ValueError(
            "The fetched model's manifest differs from what was published. "
            "Publish it again, then add it from the link."
        )
    return marker, manifest, sha, total


def dataset_card(name, detection):
    facts = ""
    if detection.get("episodes") and detection.get("frames") and detection.get("fps"):
        facts = (
            f" It holds {detection['episodes']} episodes and {detection['frames']} frames "
            f"at {detection['fps']} frames per second."
        )
    return (
        "---\ntask_categories:\n- robotics\ntags:\n- LeRobot\nconfigs:\n- config_name: default\n"
        "  data_files: data/*/*.parquet\n---\n\n"
        f"# {name}\n\nA LeRobot dataset published from OpenJensen.{facts}\n\n"
        "To use it in another OpenJensen workspace, open Datasets and choose "
        "**Add from a Hugging Face link**.\n"
    )


def model_card(name, artifact):
    family = artifact.metadata.get("architecture")
    described = f"A {family} policy" if isinstance(family, str) and family else "A policy"
    return (
        "---\ntags:\n- robotics\n---\n\n"
        f"# {name}\n\n{described} published from OpenJensen, with the records of how it "
        "was made. The files are an OpenJensen model bundle.\n\n"
        "To use it in another OpenJensen workspace, open Models and choose "
        "**Add from a Hugging Face link**.\n"
    )


def fetched_models(data_dir: Path, project_id: str) -> list[PolicyArtifact]:
    """Models added from a link. Each is a complete local bundle with its own record."""
    root = Path(data_dir) / "huggingface-models"
    if not root.is_dir() or root.is_symlink():
        return []
    rows = []
    for path in sorted(root.glob("*.json")):
        try:
            if path.is_symlink() or path.stat().st_size > MAX_RECORD:
                continue
            record = json.loads(path.read_bytes())
            artifact = PolicyArtifact.model_validate(record["artifact"])
        except OSError, ValueError, KeyError, TypeError:
            continue
        if (
            artifact.project_id == project_id
            and artifact.job_id == path.stem
            and artifact.path == f"jobs/{path.stem}/huggingface/bundle"
        ):
            rows.append(artifact)
    return rows


def fetched_record(data_dir: Path, job_id: str):
    """The saved record of one model added from a link, or nothing."""
    if not isinstance(job_id, str) or not re.fullmatch(IDENTITY, job_id):
        return None
    path = Path(data_dir) / "huggingface-models" / f"{job_id}.json"
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_RECORD:
            return None
        record = json.loads(path.read_bytes())
    except OSError, ValueError:
        return None
    return record if isinstance(record, dict) else None


class HuggingFaceSharing:
    def __init__(self, app, data_dir):
        self.app, self.data_dir = app, Path(data_dir)
        self.jobs = ShareJobs(data_dir)
        self.work = self.data_dir / "huggingface-work"
        self.tasks: dict[str, asyncio.Task] = {}
        self._cancelled: set[str] = set()
        self._machine = None
        self._register = asyncio.Lock()
        for value in self.jobs.records():
            if value["status"] not in TERMINAL:
                try:
                    self.jobs.finish(
                        value["id"],
                        "interrupted",
                        error="Interrupted when the application stopped. Start it again.",
                    )
                except OSError, ValueError:
                    continue
        shutil.rmtree(self.work, ignore_errors=True)

    # --- Whose account -----------------------------------------------------------------
    def token(self) -> str | None:
        """The token saved in Settings, or ``None`` to use this computer's own login."""
        return self.app.state.huggingface_connection.token()

    async def account(self) -> dict:
        status = self.app.state.huggingface_connection.status()
        if status.configured and status.username:
            return {"connected": True, "username": status.username, "source": "settings"}
        path = machine_token_path()
        try:
            stat = path.stat()
            signature = (str(path), stat.st_mtime_ns, stat.st_size)
        except OSError:
            signature = None
        if signature is None or not signature[2]:
            return {
                "connected": False,
                "username": None,
                "source": None,
                "message": NOT_CONNECTED,
            }
        if self._machine is None or self._machine[0] != signature:
            try:
                self._machine = (signature, await transfer.whoami(None))
            except TransferError as exc:
                return {"connected": False, "username": None, "source": None, "message": str(exc)}
        return {"connected": True, "username": self._machine[1], "source": "machine"}

    # --- What would be sent, and where ---------------------------------------------------
    async def _source(self, project_id, kind, ident):
        if kind == "dataset":
            library = self.app.state.dataset_library
            value = library.get(ident, project_id)
            if value["status"] != "ready" or not value.get("normalized"):
                raise TransferError("Only a saved dataset that is ready can be published.", 409)
            return value["name"], value
        try:
            artifact = await self.app.state.execution.lifecycle.artifact(project_id, ident)
        except ValueError as exc:
            raise TransferError("That model does not exist in this project.", 404) from exc
        return artifact.run_name or artifact.label, artifact

    def _latest(self, project_id, operation, source_id):
        active, published = None, None
        for value in self.jobs.records(project_id):
            if value["operation"] != operation or value["inputs"].get("source_id") != source_id:
                continue
            if value["status"] not in TERMINAL and active is None:
                active = "share:" + value["id"]
            if value["status"] == "succeeded" and published is None:
                published = {**value["result"], "published_at": value["updated_at"]}
        return active, published

    async def destination(self, project_id, kind: Kind, ident) -> dict:
        name, _ = await self._source(project_id, kind, ident)
        account = await self.account()
        operation = "dataset.publish" if kind == "dataset" else "policy.publish"
        active, published = self._latest(project_id, operation, ident)
        repo_id = (
            published["repo_id"]
            if published
            else f"{account['username']}/{transfer.slug(name)}"
            if account["connected"]
            else None
        )
        return {
            "kind": kind,
            "id": ident,
            "name": name,
            "account": account,
            "repo_id": repo_id,
            "url": transfer.repository_url(kind, repo_id) if repo_id else None,
            "private": True,
            "active_job": active,
            "published": published,
        }

    # --- Starting and stopping work --------------------------------------------------------
    def _start(self, record, work):
        ident = record["id"]

        async def run():
            outcome = {"status": "failed", "error": "Stopped unexpectedly"}
            try:
                outcome = {"status": "succeeded", "result": await work(record)}
            except asyncio.CancelledError:
                cancelled = ident in self._cancelled
                outcome = {
                    "status": "cancelled" if cancelled else "interrupted",
                    "error": "Cancelled. Nothing was saved here."
                    if cancelled
                    else "Interrupted when the application stopped. Start it again.",
                }
                raise
            except (TransferError, LibraryError, ValueError, OSError) as exc:
                outcome = {"status": "failed", "error": str(exc) or type(exc).__name__}
            except Exception as exc:
                LOGGER.exception("Hugging Face sharing job %s stopped unexpectedly", ident)
                outcome = {"status": "failed", "error": f"Unexpected {type(exc).__name__}"}
            finally:
                # Working files go before the job reports its end, so "done" means done.
                self._cancelled.discard(ident)
                shutil.rmtree(self.work / ident, ignore_errors=True)
                try:
                    self.work.rmdir()
                except OSError:
                    pass
                self.jobs.finish(ident, outcome.pop("status"), **outcome)

        task = asyncio.create_task(run())
        self.tasks[ident] = task
        task.add_done_callback(
            lambda finished: (
                self.tasks.pop(ident, None) if self.tasks.get(ident) is finished else None
            )
        )
        return record

    async def publish(self, project_id, kind: Kind, ident, repo_id=None, private=True):
        name, _ = await self._source(project_id, kind, ident)
        account = await self.account()
        if not account["connected"]:
            raise TransferError(account.get("message") or NOT_CONNECTED, 409)
        repo_id = (repo_id or "").strip() or f"{account['username']}/{transfer.slug(name)}"
        if not transfer.REPOSITORY.fullmatch(repo_id):
            raise TransferError("Use a repository name such as you/pick-pen.")
        operation = "dataset.publish" if kind == "dataset" else "policy.publish"
        for value in self.jobs.records(project_id):
            if (
                value["operation"] == operation
                and value["inputs"].get("source_id") == ident
                and value["status"] not in TERMINAL
            ):
                return value  # A second click joins the publish already under way.
        record = self.jobs.create(
            project_id,
            operation,
            name,
            {
                "kind": kind,
                "source_id": ident,
                "repo_id": repo_id,
                "private": bool(private),
                "url": transfer.repository_url(kind, repo_id),
                "account": account["source"],
            },
        )
        return self._start(
            record, self._publish_dataset if kind == "dataset" else self._publish_model
        )

    async def fetch(self, project_id, kind: Kind, link):
        repo_id, revision = transfer.parse_reference(link, kind)
        operation = "dataset.fetch" if kind == "dataset" else "policy.fetch"
        for value in self.jobs.records(project_id):
            if (
                value["operation"] == operation
                and value["inputs"].get("repo_id") == repo_id
                and value["inputs"].get("revision") == revision
                and value["status"] not in TERMINAL
            ):
                return value
        record = self.jobs.create(
            project_id,
            operation,
            repo_id,
            {
                "kind": kind,
                "repo_id": repo_id,
                "revision": revision,
                "url": transfer.repository_url(kind, repo_id),
            },
        )
        return self._start(record, self._fetch_dataset if kind == "dataset" else self._fetch_model)

    async def cancel(self, ident):
        task = self.tasks.get(ident)
        if task is None or task.done():
            raise TransferError("This job cannot be cancelled in its current state", 409)
        self._cancelled.add(ident)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return self.jobs.get(ident)

    async def close(self):
        owned = list(self.tasks.values())
        for task in owned:
            task.cancel()
        await asyncio.gather(*owned, return_exceptions=True)

    # --- Publishing -----------------------------------------------------------------------
    def _marker(self, kind, name, **fields):
        return {
            "schema_version": 1,
            "kind": kind,
            "name": name,
            "application": build.current().label,
            **fields,
        }

    async def _push(self, record, kind, folder, marker, expect, card, tag=None):
        ident, inputs = record["id"], record["inputs"]
        size = sum((folder / name).stat().st_size for name in expect)

        uploading = f"Uploading {len(expect)} files ({megabytes(size)}) to {inputs['repo_id']}"
        state = {"task": None}

        async def elapsed():
            # The client reports no byte counts for an upload, so say how long it has run.
            started = time.monotonic()
            while True:
                await asyncio.sleep(HEARTBEAT_SECONDS)
                minutes = max(1, round((time.monotonic() - started) / 60))
                self.jobs.progress(
                    ident, "uploading", f"{uploading} · {minutes} min so far", event=False
                )

        def progress(report):
            if report.get("stage") == "uploading" and state["task"] is None:
                self.jobs.progress(ident, "uploading", uploading)
                state["task"] = asyncio.create_task(elapsed())

        published = {**marker, "state": "published", "published_at": now()}
        files = {MARKER: json.dumps(published, indent=1) + "\n"}
        if "README.md" not in expect:
            files["README.md"] = card
        try:
            result = await transfer.push(
                kind,
                folder,
                inputs["repo_id"],
                self.token(),
                private=inputs["private"],
                message=f"Publish {record['title']} from OpenJensen",
                begin={MARKER: json.dumps({**marker, "state": "publishing"}, indent=1) + "\n"},
                files=files,
                marker=MARKER,
                expect=expect,
                tag=tag,
                progress=progress,
                timeout=TRANSFER_SECONDS,
            )
        finally:
            if state["task"] is not None:
                state["task"].cancel()
                await asyncio.gather(state["task"], return_exceptions=True)
        return {**result, "files": len(expect), "bytes": size}

    async def _publish_dataset(self, record):
        ident, inputs = record["id"], record["inputs"]
        library = self.app.state.dataset_library
        value = library.get(inputs["source_id"], record["project_id"])
        root = library.resolve(inputs["source_id"], record["project_id"])
        self.jobs.progress(ident, "preparing", "Listing the dataset's files and their hashes")
        identity, files = await asyncio.to_thread(inventory, root)
        if any(row[0] == MARKER for row in files):
            raise ValueError(f"A dataset that already contains {MARKER} cannot be published")
        tag = None
        try:
            version = json.loads((root / "meta/info.json").read_bytes()).get("codebase_version")
            if isinstance(version, str) and re.fullmatch(r"v\d+\.\d+(\.\d+)?", version):
                tag = version
        except OSError, ValueError, AttributeError:
            pass
        detection = value.get("detection") or {}
        marker = self._marker(
            "dataset",
            value["name"],
            sha256=identity,
            files={name: {"size": size, "sha256": sha} for name, size, sha in files},
            dataset={
                "format": (value.get("converted") or detection).get("format")
                or detection.get("format"),
                "annotations": value.get("annotations"),
                "conversion": value.get("conversion"),
            },
        )
        result = await self._push(
            record,
            "dataset",
            root,
            marker,
            [row[0] for row in files],
            dataset_card(value["name"], value.get("converted") or detection),
            tag,
        )
        return {**result, "sha256": identity}

    async def _publish_model(self, record):
        from vla_platform import robot_policy_install
        from vla_platform.lifecycle.service import validate_bundle

        ident, inputs, project_id = record["id"], record["inputs"], record["project_id"]
        lifecycle = self.app.state.execution.lifecycle
        artifact = await lifecycle.artifact(project_id, inputs["source_id"])
        directory = self.data_dir / artifact.path
        if (directory / "remote.json").is_file():
            # The files live in cloud storage: bring them here once, straight to a folder.
            if shutil.disk_usage(self.data_dir).free < artifact.file_bytes + DISK_HEADROOM:
                raise ValueError("Not enough free storage to bring this model here for publishing")
            bringing = f"Bringing the model from cloud storage ({megabytes(artifact.file_bytes)})"
            self.jobs.progress(ident, "collecting", bringing)
            directory = self.work / ident / "bundle"
            directory.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            reported = [time.monotonic()]

            def received(arrived):
                if time.monotonic() - reported[0] >= 2:
                    reported[0] = time.monotonic()
                    share = min(99, arrived * 100 // max(1, artifact.file_bytes))
                    self.jobs.progress(ident, "collecting", f"{bringing} · {share}%", event=False)

            # The same allowance as installing a model, which fetches the same archive.
            allowed = robot_policy_install.DOWNLOAD_SECONDS
            try:
                async with asyncio.timeout(allowed) as limit:
                    source = await lifecycle.download(project_id, artifact.id)
                    await unpack(source, directory, artifact.file_bytes + MAX_MARKER, received)
            except TimeoutError:
                if not limit.expired():
                    raise
                raise ValueError(
                    "The model could not be fetched from storage within "
                    f"{max(1, round(allowed / 60))} minutes. Check the connection and publish "
                    "it again."
                ) from None
        self.jobs.progress(ident, "preparing", "Checking every file against the model's manifest")
        manifest, sha, total = await asyncio.to_thread(validate_bundle, directory, directory.parent)
        if sha != artifact.manifest_sha256:
            raise ValueError("The model's files differ from its registered manifest")
        if MARKER in manifest["files"]:
            raise ValueError(f"A model that already contains {MARKER} cannot be published")
        name = artifact.run_name or artifact.label
        marker = self._marker(
            "model",
            name,
            manifest_sha256=sha,
            file_bytes=total,
            model={
                "label": artifact.label,
                "format": artifact.format,
                "run_name": artifact.run_name,
                # What the training recording proves about units and joint order travels
                # with the model, so it installs on another machine as it does here.
                "recorded_control": await robot_policy_install.recorded_control(
                    self.app, project_id, artifact
                ),
            },
        )
        result = await self._push(
            record,
            "model",
            directory,
            marker,
            ["manifest.json", *sorted(manifest["files"])],
            model_card(name, artifact),
        )
        return {**result, "manifest_sha256": sha}

    # --- Adding from a link -----------------------------------------------------------------
    def _download_progress(self, ident, folder, repo_id):
        """Report how much of a download has arrived by watching its folder grow."""
        state = {"task": None}

        async def watch(files, total):
            def arrived():
                return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())

            while True:
                await asyncio.sleep(2)
                try:
                    done = await asyncio.to_thread(arrived)
                except OSError:
                    continue
                share = min(99, int(done * 100 / total)) if total else 0
                self.jobs.progress(
                    ident,
                    "downloading",
                    f"Downloading {files} files ({megabytes(total)}) from {repo_id} · {share}%",
                    event=False,
                )

        def progress(report):
            if report.get("stage") == "downloading" and state["task"] is None:
                files, total = int(report.get("files") or 0), int(report.get("bytes") or 0)
                self.jobs.progress(
                    ident,
                    "downloading",
                    f"Downloading {files} files ({megabytes(total)}) from {repo_id}",
                )
                state["task"] = asyncio.create_task(watch(files, total))

        async def stop():
            if state["task"] is not None:
                state["task"].cancel()
                await asyncio.gather(state["task"], return_exceptions=True)

        return progress, stop

    async def _pull(self, record, kind, folder, max_files, max_bytes):
        ident, inputs = record["id"], record["inputs"]
        if shutil.disk_usage(self.data_dir).free < DISK_HEADROOM:
            raise ValueError("Not enough free storage to add this from Hugging Face")
        folder.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.jobs.progress(ident, "connecting", f"Looking up {inputs['repo_id']} on Hugging Face")
        progress, stop = self._download_progress(ident, folder, inputs["repo_id"])
        try:
            return await transfer.pull(
                kind,
                inputs["repo_id"],
                folder,
                self.token(),
                revision=inputs.get("revision"),
                max_files=max_files,
                max_bytes=max_bytes,
                progress=progress,
                timeout=TRANSFER_SECONDS,
            )
        finally:
            await stop()

    async def _fetch_dataset(self, record):
        ident, inputs, project_id = record["id"], record["inputs"], record["project_id"]
        library = self.app.state.dataset_library
        folder = self.work / ident / "dataset"
        pulled = await self._pull(
            record, "dataset", folder, DATASET_FILES + 3, DATASET_BYTES + MAX_MARKER
        )
        self.jobs.progress(ident, "verifying", "Checking the files that arrived")
        marker = await asyncio.to_thread(settle_dataset, folder)
        files = {
            path.relative_to(folder).as_posix(): path.stat().st_size
            for path in sorted(folder.rglob("*"))
            if path.is_file()
        }
        if not files or len(files) > DATASET_FILES or sum(files.values()) > DATASET_BYTES:
            raise ValueError(
                "This repository is empty or larger than a saved dataset can be "
                f"({DATASET_BYTES // 1024**3} GiB, {DATASET_FILES} files)."
            )
        name = (marker or {}).get("name")
        if not isinstance(name, str) or not name.strip() or not name.isprintable():
            name = inputs["repo_id"].split("/")[1]
        name = name.strip()[:100]
        created = library.begin(project_id, name)
        saved = created["id"]
        operation = library.get(saved).get("_import_job_id")
        try:
            self.jobs.progress(ident, "saving", "Saving the dataset in this project")
            source = library.directory(saved) / "source"
            source.rmdir()
            folder.rename(source)
            async with library.lock:
                value = library.get(saved)
                value.update(
                    files=files,
                    total_bytes=sum(files.values()),
                    huggingface={
                        "repo_id": inputs["repo_id"],
                        "revision": pulled["revision"],
                        "url": pulled["url"],
                    },
                )
                library.save(value)
            # Finalizing reads the dataset back. The same files added twice to one
            # project give the dataset that is already saved there.
            dataset = await library.finalize(saved)
            detection = dataset.get("detection") or {}
            if marker is not None and (
                dataset["status"] != "ready" or detection.get("sha256") != marker["sha256"]
            ):
                raise ValueError("The fetched dataset could not be read back as it was published")
            if dataset["status"] != "ready" and not detection.get("convertible"):
                raise ValueError(
                    "This repository is not a dataset OpenJensen can read. "
                    "It needs a LeRobot dataset, or recordings it can convert."
                )
        except BaseException as exc:
            # Leave no half-made dataset and no job that looks as if it is still uploading.
            shutil.rmtree(library.directory(saved), ignore_errors=True)
            if operation:
                library.operations.update(
                    operation,
                    status="failed",
                    stage="fetching",
                    error=str(exc) or "Adding this dataset was interrupted",
                )
            raise
        labels = ((marker or {}).get("dataset") or {}).get("annotations")
        if (
            dataset["id"] == saved
            and dataset["status"] == "ready"
            and isinstance(labels, dict)
            and (labels.get("views") or labels.get("frames"))
        ):
            try:
                dataset = await asyncio.to_thread(
                    library.annotate,
                    saved,
                    {
                        "revision": dataset["annotation_revision"],
                        "views": labels.get("views") or {},
                        "frames": labels.get("frames") or {},
                    },
                )
            except LibraryError, KeyError, TypeError, AttributeError:
                pass  # The files are what matters; labels that no longer fit are left out.
        return {
            **pulled,
            "dataset_id": dataset["id"],
            "name": dataset["name"],
            "status": dataset["status"],
            "sha256": detection.get("sha256"),
            "already_saved": dataset["id"] != saved,
            "verified": marker is not None,
        }

    async def _fetch_model(self, record):
        ident, inputs, project_id = record["id"], record["inputs"], record["project_id"]
        folder = self.work / ident / "bundle"
        pulled = await self._pull(record, "model", folder, MODEL_FILES, MODEL_BYTES + MAX_MARKER)
        self.jobs.progress(ident, "verifying", "Checking every file against the model's manifest")
        marker, manifest, sha, total = await asyncio.to_thread(settle_model, folder)
        details = marker["model"]
        async with self._register:
            for existing in fetched_models(self.data_dir, project_id):
                if existing.manifest_sha256 == sha:
                    return {
                        **pulled,
                        "artifact_id": existing.id,
                        "name": existing.label,
                        "manifest_sha256": sha,
                        "already_saved": True,
                    }
            self.jobs.progress(ident, "saving", "Saving the model in this project")
            destination = self.data_dir / "jobs" / ident / "huggingface" / "bundle"
            try:
                artifact = PolicyArtifact(
                    id=f"{ident}:huggingface",
                    project_id=project_id,
                    job_id=ident,
                    label=str(details.get("label") or marker.get("name") or inputs["repo_id"])[
                        :200
                    ],
                    format=details.get("format"),
                    path=destination.relative_to(self.data_dir).as_posix(),
                    manifest_sha256=sha,
                    file_bytes=total,
                    metadata=manifest.get("metadata", {}),
                )
            except ValueError:
                raise ValueError("This repository's OpenJensen record cannot be read") from None
            destination.parent.mkdir(parents=True, mode=0o700)
            folder.rename(destination)
            registry = self.data_dir / "huggingface-models"
            registry.mkdir(mode=0o700, exist_ok=True)
            document = {
                "schema_version": 1,
                "artifact": artifact.model_dump(mode="json", exclude={"run_name"}),
                "huggingface": {
                    "repo_id": inputs["repo_id"],
                    "revision": pulled["revision"],
                    "url": pulled["url"],
                },
                "recorded_control": details.get("recorded_control"),
                "added_at": now(),
            }
            descriptor, temporary = tempfile.mkstemp(prefix=".model-", dir=registry)
            try:
                with os.fdopen(descriptor, "w") as stream:
                    json.dump(document, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, registry / f"{ident}.json")
            except BaseException:
                shutil.rmtree(destination.parent.parent, ignore_errors=True)
                raise
            finally:
                Path(temporary).unlink(missing_ok=True)
            name = details.get("run_name")
            if isinstance(name, str) and name.strip():
                try:
                    self.app.state.execution.lifecycle.model_names.rename(
                        project_id, ident, name.strip()[:80]
                    )
                except OSError, ValueError:
                    pass  # The model keeps its automatic name.
        return {
            **pulled,
            "artifact_id": artifact.id,
            "name": artifact.label,
            "manifest_sha256": sha,
            "already_saved": False,
        }
