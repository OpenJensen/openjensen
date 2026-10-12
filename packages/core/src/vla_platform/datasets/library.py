"""Workspace-owned dataset uploads, deterministic adapters and editable annotations."""

import asyncio
import errno
import json
import math
import os
import re
import shutil
import signal
import tempfile
import unicodedata
import zipfile
import zlib
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from filelock import FileLock

from vla_platform.contracts import DatasetProfile, IntakeRequest, now
from vla_platform.datasets.local_preview import reader_python
from vla_platform.environment import getenv

MAX_BYTES = 2 * 1024**3
MAX_FILES = 4096
MAX_RECORD_BYTES = 32 * 1024**2
DISK_HEADROOM = 1024**3


def canonical_name(name):
    if (
        not isinstance(name, str)
        or not name
        or len(name) > 1024
        or any(character in name for character in "\\\0:")
        or any(not character.isprintable() for character in name)
        or any(part in ("", ".", "..") for part in name.split("/"))
    ):
        raise LibraryError("Dataset file must use a canonical relative path")
    return name


def name_key(name):
    # Keep uploads portable across case-insensitive and Unicode-normalizing filesystems.
    return unicodedata.normalize("NFC", canonical_name(name)).casefold()


def record_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate dataset record keys")
        value[key] = item
    return value


def invalid_number(_):
    raise ValueError("Non-finite dataset record number")


def finite_float(raw):
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError("Non-finite dataset record number")
    return value


def sync_directory(directory):
    if os.name == "posix":
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


async def owned_thread(function, *args):
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    result = task.result()
    if cancelled:
        raise asyncio.CancelledError
    return result


class LibraryError(ValueError):
    def __init__(self, message, status=422):
        self.status = status
        super().__init__(message)


def contained(root, name):
    canonical_name(name)
    path = Path(name)
    if path.is_absolute() or any(p in (".", "..") for p in path.parts):
        raise LibraryError("Dataset file must be relative to its folder")
    target = root / path
    if not target.resolve().is_relative_to(root.resolve()):
        raise LibraryError("Dataset file escapes its folder")
    return target


class DatasetLibrary:
    def __init__(self, data_dir, *, reconcile=True):
        self.root = (data_dir / "dataset-library").resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.tasks = {}
        self.lock = asyncio.Lock()
        self.worker_limit = asyncio.Semaphore(1)
        for value in self.records():
            if reconcile and value["status"] == "converting":
                value.update(
                    status="failed",
                    error="Conversion was interrupted by app restart. Retry conversion.",
                )
                try:
                    self.save(value)
                except OSError:
                    # A read-only or full workspace must still serve its saved diagnostics.
                    continue

    def directory(self, ident):
        if not re.fullmatch(r"[a-f0-9]{32}", ident):
            raise LibraryError("Dataset not found", 404)
        directory = self.root / ident
        if directory.is_symlink():
            raise LibraryError("Dataset directory is unavailable", 422)
        return directory

    def validate_record(self, value, ident):
        if not isinstance(value, dict) or value.get("id") != ident:
            raise ValueError("Invalid record identity")
        if (
            not isinstance(value.get("project_id"), str)
            or not 1 <= len(value["project_id"]) <= 128
            or not isinstance(value.get("name"), str)
            or not 1 <= len(value["name"]) <= 100
            or value.get("source") != "local"
            or value.get("status") not in {"uploading", "detected", "converting", "ready", "failed"}
            or not isinstance(value.get("created_at"), str)
        ):
            raise ValueError("Invalid record fields")
        datetime.fromisoformat(value["created_at"])
        files = value.get("files")
        if not isinstance(files, dict) or len(files) > MAX_FILES:
            raise ValueError("Invalid file inventory")
        keys = set()
        for name, size in files.items():
            key = name_key(name)
            if key in keys or type(size) is not int or size < 0:
                raise ValueError("Invalid file inventory")
            keys.add(key)
        if (
            type(value.get("total_bytes")) is not int
            or value["total_bytes"] != sum(files.values())
            or not 0 <= value["total_bytes"] <= MAX_BYTES
            or type(value.get("annotation_revision")) is not int
            or value["annotation_revision"] < 0
        ):
            raise ValueError("Invalid record inventory or revision")
        annotations = value.get("annotations")
        if not isinstance(annotations, dict):
            raise ValueError("Invalid annotations")
        for key, maximum, length in (("views", 8, 100), ("frames", 12000, 1000)):
            labels = annotations.get(key)
            if (
                not isinstance(labels, dict)
                or len(labels) > maximum
                or any(
                    not isinstance(name, str) or not isinstance(label, str) or len(label) > length
                    for name, label in labels.items()
                )
            ):
                raise ValueError("Invalid annotations")
        detection = value.get("detection")
        if detection is not None or value["status"] != "uploading":
            if (
                not isinstance(detection, dict)
                or not isinstance(detection.get("sha256"), str)
                or not re.fullmatch(r"[a-f0-9]{64}", detection["sha256"])
                or not isinstance(detection.get("format"), str)
                or not detection["format"]
                or type(detection.get("convertible")) is not bool
                or not isinstance(detection.get("cameras"), list)
                or any(not isinstance(camera, str) for camera in detection["cameras"])
            ):
                raise ValueError("Invalid detection")
        if "converted" in value and (
            not isinstance(value["converted"], dict)
            or not isinstance(value["converted"].get("cameras"), list)
            or any(not isinstance(camera, str) for camera in value["converted"]["cameras"])
        ):
            raise ValueError("Invalid converted dataset")
        if value.get("error") is not None and not isinstance(value["error"], str):
            raise ValueError("Invalid record diagnostic")
        for key in ("normalized", "source_directory"):
            if value.get(key) is not None:
                canonical_name(value[key])
        if value["status"] == "ready" and (
            not value.get("normalized") or not isinstance(value.get("detection"), dict)
        ):
            raise ValueError("Incomplete accepted dataset")
        return value

    def unavailable(self, ident):
        return {
            "id": ident,
            "project_id": None,
            "name": "Unavailable saved dataset",
            "source": "local",
            "status": "unavailable",
            "created_at": "",
            "total_bytes": None,
            "uploaded_files": None,
            "can_discard": False,
            "error": (
                "Saved dataset metadata is missing, invalid or unreadable. "
                "Original files are preserved; review the workspace record before repairing it."
            ),
        }

    def records(self):
        for directory in self.root.iterdir():
            if re.fullmatch(r"[a-f0-9]{32}", directory.name) and directory.is_dir():
                yield self.get(directory.name)

    def save(self, value):
        self.validate_record(value, value["id"])
        path = self.directory(value["id"]) / "record.json"
        payload = json.dumps(value, allow_nan=False).encode()
        if len(payload) > MAX_RECORD_BYTES:
            raise LibraryError("Saved dataset record exceeds its size limit")
        descriptor, name = tempfile.mkstemp(prefix=".record-", dir=path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
            sync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def get(self, ident, project_id=None):
        if not re.fullmatch(r"[a-f0-9]{32}", ident):
            raise LibraryError("Dataset not found", 404)
        try:
            directory = self.directory(ident)
            if not directory.exists():
                raise LibraryError("Dataset not found", 404)
            path = directory / "record.json"
            if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_RECORD_BYTES:
                raise ValueError("Invalid record file")
            with path.open("rb") as handle:
                raw = handle.read(MAX_RECORD_BYTES + 1)
            if len(raw) > MAX_RECORD_BYTES:
                raise ValueError("Oversized record")
            value = self.validate_record(
                json.loads(
                    raw,
                    object_pairs_hook=record_pairs,
                    parse_constant=invalid_number,
                    parse_float=finite_float,
                ),
                ident,
            )
        except LibraryError as exc:
            if exc.status == 404:
                raise
            value = self.unavailable(ident)
        except OSError, ValueError, TypeError, KeyError, RecursionError:
            value = self.unavailable(ident)
        if project_id and value["project_id"] != project_id:
            raise LibraryError("Dataset does not belong to this project", 404)
        return value

    def public(self, value):
        result = {k: v for k, v in value.items() if k not in ("normalized", "files")}
        result["uploaded_files"] = len(value["files"]) if "files" in value else None
        result["can_discard"] = value["status"] == "uploading" and not (
            value.get("detection") or value.get("normalized") or value.get("converted")
        )
        if value["status"] == "uploading" and not result.get("error"):
            result["error"] = (
                "Upload is unfinished. Finalize the saved files or discard this upload."
            )
        return result

    def begin(self, project_id, name):
        value = {
            "id": uuid4().hex,
            "project_id": project_id,
            "name": name,
            "source": "local",
            "status": "uploading",
            "created_at": now(),
            "files": {},
            "total_bytes": 0,
            "detection": None,
            "normalized": None,
            "error": None,
            "annotations": {"views": {}, "frames": {}},
            "annotation_revision": 0,
        }
        (self.directory(value["id"]) / "source").mkdir(parents=True)
        self.save(value)
        return self.public(value)

    async def upload(self, ident, name, stream):
        # A complete file is published atomically. Failed or duplicate writes never overwrite it.
        async with self.lock:
            value = self.get(ident)
            if value["status"] != "uploading":
                raise LibraryError("This upload is already finalized", 409)
            key = name_key(name)
            if len(value["files"]) >= MAX_FILES or key in {name_key(n) for n in value["files"]}:
                raise LibraryError("File already uploaded or file count limit exceeded", 409)
            destination = contained(self.directory(ident) / "source", name)
            if destination.exists() or destination.is_symlink():
                raise LibraryError("File already exists; its original bytes were preserved", 409)
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=".upload-", dir=destination.parent)
            size = 0
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    async for chunk in stream:
                        size += len(chunk)
                        if size + value["total_bytes"] > MAX_BYTES:
                            raise LibraryError("Local uploads support at most 2 GiB", 413)
                        if shutil.disk_usage(self.root).free < DISK_HEADROOM + len(chunk):
                            raise LibraryError("Not enough free storage for this upload", 507)
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
                # A hard link publishes without replacing a file created by an alias or race.
                try:
                    os.link(temporary, destination)
                except FileExistsError as exc:
                    raise LibraryError(
                        "File already exists; its original bytes were preserved", 409
                    ) from exc
                sync_directory(destination.parent)
                value["files"][name] = size
                value["total_bytes"] += size
                value["error"] = None
                self.save(value)
            finally:
                Path(temporary).unlink(missing_ok=True)
            return {"uploaded_files": len(value["files"]), "total_bytes": value["total_bytes"]}

    def extract(self, ident):
        value = self.get(ident)
        source = self.directory(ident) / "source"
        names = list(value["files"])
        if len(names) != 1 or not names[0].lower().endswith(".zip"):
            return
        directory = self.directory(ident)
        destination = directory / "extracted"
        staging = None
        try:
            with zipfile.ZipFile(source / names[0]) as archive:
                entries, keys, total = [], set(), 0
                for info in archive.infolist():
                    name = info.filename.rstrip("/") if info.is_dir() else info.filename
                    canonical_name(name)
                    if info.is_dir():
                        continue
                    if (info.external_attr >> 16) & 0o170000 == 0o120000:
                        raise LibraryError("ZIP symlinks are unsupported")
                    key = name_key(info.filename)
                    if key in keys:
                        raise LibraryError("ZIP has duplicate file paths")
                    keys.add(key)
                    total += info.file_size
                    entries.append(info)
                    if total > MAX_BYTES or len(entries) > MAX_FILES:
                        raise LibraryError("Expanded ZIP exceeds 2 GiB / 4096 files", 413)
                if not entries:
                    raise LibraryError("ZIP contains no dataset files")
                saved = value.get("source_directory")
                if saved and saved.split("/")[0].startswith("extracted"):
                    destination = directory / saved.split("/")[0]
                if self.complete_extraction(destination, entries):
                    pass  # Also recovers publication before its record was saved.
                else:
                    if destination.exists() or destination.is_symlink():
                        # Preserve older partial extractions; publish into a fresh owned folder.
                        destination = directory / ("extracted-" + uuid4().hex)
                    if shutil.disk_usage(self.root).free < total + DISK_HEADROOM:
                        raise LibraryError("Not enough free storage to extract this ZIP", 507)
                    staging = Path(tempfile.mkdtemp(prefix=".extract-", dir=directory))
                    expanded = 0
                    for info in entries:
                        target = contained(staging, info.filename)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(info) as original, target.open("xb") as handle:
                            size = 0
                            while chunk := original.read(1024**2):
                                size += len(chunk)
                                expanded += len(chunk)
                                if size > info.file_size or expanded > MAX_BYTES:
                                    raise LibraryError(
                                        "Expanded ZIP exceeds its declared size", 413
                                    )
                                if shutil.disk_usage(self.root).free < DISK_HEADROOM + len(chunk):
                                    raise LibraryError(
                                        "Not enough free storage to extract this ZIP", 507
                                    )
                                handle.write(chunk)
                            if size != info.file_size:
                                raise LibraryError("ZIP file is incomplete")
                            handle.flush()
                            os.fsync(handle.fileno())
                        sync_directory(target.parent)
                    sync_directory(staging)
                    staging.rename(destination)
                    staging = None
                    sync_directory(directory)
            children = list(destination.iterdir())
            if len(children) == 1 and children[0].is_dir():
                value["source_directory"] = destination.name + "/" + children[0].name
            else:
                value["source_directory"] = destination.name
            self.save(value)
        except (zipfile.BadZipFile, RuntimeError) as exc:
            raise LibraryError("ZIP is invalid, incomplete or encrypted") from exc
        except OSError as exc:
            raise LibraryError(
                "ZIP extraction could not be saved; uploaded source bytes were preserved",
                507 if exc.errno == errno.ENOSPC else 422,
            ) from exc
        finally:
            if staging is not None:
                shutil.rmtree(staging)

    def complete_extraction(self, directory, entries):
        if directory.is_symlink() or not directory.is_dir():
            return False
        expected = {info.filename: info for info in entries}
        actual = {}
        try:
            for path in directory.rglob("*"):
                if path.is_symlink():
                    return False
                if path.is_file():
                    actual[path.relative_to(directory).as_posix()] = path
            if set(actual) != set(expected):
                return False
            for name, path in actual.items():
                info = expected[name]
                if path.stat().st_size != info.file_size:
                    return False
                crc = 0
                with path.open("rb") as handle:
                    while chunk := handle.read(1024**2):
                        crc = zlib.crc32(chunk, crc)
                if crc != info.CRC:
                    return False
        except OSError:
            return False
        return True

    def source(self, value):
        return contained(self.directory(value["id"]), value.get("source_directory", "source"))

    async def worker(self, payload, timeout=300):
        executable = getenv("JENSEN_CPU_READER_PYTHON") or str(reader_python())
        async with self.worker_limit:
            try:
                process = await asyncio.create_subprocess_exec(
                    executable,
                    "-I",
                    "-B",
                    str(Path(__file__).with_name("import_worker.py")),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                    start_new_session=os.name == "posix",
                    env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
                )
            except OSError as exc:
                raise LibraryError(
                    "Dataset reader unavailable. Configure the CPU reader on the app host.",
                    503,
                ) from exc
            try:
                async with asyncio.timeout(timeout):
                    raw, _ = await process.communicate(json.dumps(payload).encode())
                if len(raw) > 1024**2:
                    raise LibraryError("Dataset reader exceeded its response limit")
                try:
                    result = json.loads(raw)
                except (ValueError, UnicodeError) as exc:
                    raise LibraryError("Dataset reader returned no valid result") from exc
                if process.returncode or "error" in result:
                    raise LibraryError(result.get("error", "Dataset reader failed"))
                return result
            except TimeoutError as exc:
                raise LibraryError("Dataset reader timed out; use a smaller import") from exc
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

    async def finalize(self, ident):
        async with self.lock:
            value = self.get(ident)
            if value["status"] != "uploading":
                return self.public(value)
            if not value["files"]:
                raise LibraryError("Select dataset files first")
            try:
                await owned_thread(self.extract, ident)
                value = self.get(ident)
                detection = await self.worker(
                    {"operation": "detect", "root": str(self.source(value))}
                )
            except (LibraryError, asyncio.CancelledError) as exc:
                value = self.get(ident)
                if value["status"] == "uploading":
                    value["error"] = (
                        "Finalization was interrupted. Retry finalizing the saved files."
                        if isinstance(exc, asyncio.CancelledError)
                        else str(exc)
                    )
                    self.save(value)
                raise
            # Identical uploads in one project reuse the saved source and labels.
            for existing in self.records():
                if (
                    existing["id"] != ident
                    and existing["project_id"] == value["project_id"]
                    and (existing.get("detection") or {}).get("sha256") == detection["sha256"]
                ):
                    shutil.rmtree(self.directory(ident))
                    return self.public(existing)
            value.update(detection=detection, status="detected", error=None)
            if detection["format"] == "lerobot_v3":
                value.update(status="ready", normalized=value.get("source_directory", "source"))
                try:
                    await self.worker(
                        {
                            "operation": "preview",
                            "root": str(self.source(value)),
                            "output": str(self.directory(ident) / "previews"),
                        },
                        60,
                    )
                except LibraryError as exc:
                    value["preview_note"] = str(exc)
            self.save(value)
            return self.public(value)

    async def example(self, project_id):
        for value in self.records():
            if value["project_id"] == project_id and value.get("example"):
                return self.public(value)
        value = self.begin(project_id, "Robot labeling playground")
        try:
            detection = await self.worker(
                {"operation": "demo", "root": str(self.directory(value["id"]) / "source")}
            )
            value = self.get(value["id"])
            value.update(status="detected", detection=detection, example=True)
            self.save(value)
            return await self.start_conversion(
                value["id"],
                {"fps": 6, "task": "Move the red block into the target (synthetic example)"},
            )
        except BaseException:
            shutil.rmtree(self.directory(value["id"]))
            raise

    async def start_conversion(self, ident, settings):
        async with self.lock:
            value = self.get(ident)
            if value["status"] in ("converting", "ready"):
                return self.public(value)
            if not value.get("detection") or not value["detection"]["convertible"]:
                raise LibraryError("This dataset format does not have a conversion adapter")
            value.update(status="converting", error=None, conversion=settings)
            self.save(value)
            task = asyncio.create_task(self.convert(ident, settings))
            self.tasks[ident] = task
            task.add_done_callback(lambda _: self.tasks.pop(ident, None))
            return self.public(value)

    async def convert(self, ident, settings):
        value = self.get(ident)
        output = self.directory(ident) / "converted.pending"
        if output.exists():
            shutil.rmtree(output)
        try:
            result = await self.worker(
                {
                    "operation": "convert",
                    "root": str(self.source(value)),
                    "output": str(output),
                    "sha256": value["detection"]["sha256"],
                    "settings": settings,
                }
            )
            output.rename(self.directory(ident) / "converted")
            value.update(status="ready", normalized="converted", converted=result, error=None)
        except (LibraryError, OSError, ValueError) as exc:
            value.update(status="failed", error=str(exc))
            if output.exists():
                shutil.rmtree(output)
        finally:
            self.save(value)

    def resolve(self, ident, project_id):
        value = self.get(ident, project_id)
        if value["status"] != "ready" or not value["normalized"]:
            raise LibraryError("Convert the dataset before inspecting or training")
        return contained(self.directory(ident), value["normalized"])

    def samples(self, ident):
        path = self.directory(ident) / "samples.json"
        return json.loads(path.read_text()) if path.is_file() else []

    def annotate(self, ident, payload):
        with FileLock(self.directory(ident) / "annotations.lock"):
            value = self.get(ident)
            if value["status"] != "ready":
                raise LibraryError("Wait for conversion before labeling")
            if payload["revision"] != value["annotation_revision"]:
                raise LibraryError("Labels changed in another window. Reload before saving.", 409)
            cameras = set(value.get("converted", value["detection"])["cameras"])
            if not set(payload["views"]).issubset(cameras):
                raise LibraryError("Unknown camera view")
            samples = {
                f"{r['episode_index']}:{r['frame_index']}:{r['camera']}"
                for r in self.samples(ident)
            }
            if not set(payload["frames"]).issubset(samples):
                raise LibraryError("Frame label must reference a displayed sample")
            value["annotations"] = {"views": payload["views"], "frames": payload["frames"]}
            value["annotation_revision"] += 1
            self.save(value)
            return self.public(value)

    def export(self, ident):
        value = self.get(ident)
        root = self.resolve(ident, value["project_id"])
        descriptor, name = tempfile.mkstemp(suffix=".zip", prefix="dataset-export-", dir=self.root)
        os.close(descriptor)
        with zipfile.ZipFile(name, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    archive.write(path, path.relative_to(root).as_posix())
            archive.writestr("openjensen/annotations.json", json.dumps(value["annotations"]))
            archive.writestr(
                "openjensen/source.json",
                json.dumps(
                    {
                        "sha256": value["detection"]["sha256"],
                        "format": value["detection"]["format"],
                        "conversion": value.get("conversion"),
                        "example": value.get("example", False),
                    }
                ),
            )
        return Path(name)

    async def list(self, projects, execution, project_id=None):
        rows = [self.public(value) for value in self.records()]
        rows = [
            r
            for r in rows
            if not project_id
            or r["project_id"] == project_id
            or (r["status"] == "unavailable" and r["project_id"] is None)
        ]
        for project in await projects.list():
            if project_id and project.id != project_id:
                continue
            seen = set()
            for job in sorted(
                await execution.list(project.id), key=lambda j: j.created_at, reverse=True
            ):
                if not isinstance(job.result, DatasetProfile):
                    continue
                if isinstance(job.request, IntakeRequest) and job.request.library_id:
                    for row in rows:
                        if row["id"] == job.request.library_id:
                            row["can_discard"] = False
                            if not row.get("job_id") or job.result.snapshot:
                                row["job_id"] = job.id
                            row["training_copy"] = row.get("training_copy", False) or bool(
                                job.result.snapshot
                            )
                    continue
                key = (job.result.repo_id, job.result.revision, job.result.metadata_sha256)
                if key in seen:
                    continue
                seen.add(key)
                profile = job.result
                rows.append(
                    {
                        "id": "inspection:" + job.id,
                        "job_id": job.id,
                        "project_id": project.id,
                        "name": profile.repo_id or Path(job.request.path or "Local dataset").name,
                        "source": profile.source,
                        "created_at": job.created_at,
                        "status": "ready",
                        "profile": profile.model_dump(),
                        "training_copy": bool(profile.snapshot),
                        "annotations": None,
                    }
                )
        return sorted(rows, key=lambda row: row["created_at"], reverse=True)

    async def discard(self, ident, project_id, execution):
        async with self.lock:
            value = self.get(ident, project_id)
            if not self.public(value)["can_discard"] or ident in self.tasks:
                raise LibraryError("Only an unfinished, unaccepted upload can be discarded", 409)
            for job in await execution.list(project_id):
                if isinstance(job.request, IntakeRequest) and job.request.library_id == ident:
                    raise LibraryError(
                        "This upload is referenced by a saved job and is preserved", 409
                    )
            directory = self.directory(ident)
            await owned_thread(shutil.rmtree, directory)
            sync_directory(self.root)
            return {"discarded": True, "id": ident}

    async def close(self):
        for task in list(self.tasks.values()):
            task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
