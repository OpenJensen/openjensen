"""Workspace-owned dataset uploads, deterministic adapters and editable annotations."""

import asyncio
import json
import os
import re
import shutil
import signal
import tempfile
import zipfile
from pathlib import Path
from uuid import uuid4

from filelock import FileLock

from vla_platform.contracts import DatasetProfile, IntakeRequest, now
from vla_platform.datasets.local_preview import reader_python

MAX_BYTES = 2 * 1024**3
MAX_FILES = 4096


class LibraryError(ValueError):
    def __init__(self, message, status=422):
        self.status = status
        super().__init__(message)


def contained(root, name):
    if not name or len(name) > 1024 or "\\" in name or "\0" in name:
        raise LibraryError("Invalid dataset file name")
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
        for record in self.root.glob("*/record.json"):
            value = json.loads(record.read_text())
            if reconcile and value["status"] == "converting":
                value.update(
                    status="failed",
                    error="Conversion was interrupted by app restart. Retry conversion.",
                )
                self.save(value)

    def directory(self, ident):
        if not re.fullmatch(r"[a-f0-9]{32}", ident):
            raise LibraryError("Dataset not found", 404)
        return self.root / ident

    def save(self, value):
        path = self.directory(value["id"]) / "record.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, allow_nan=False))
        temporary.replace(path)

    def get(self, ident, project_id=None):
        try:
            value = json.loads((self.directory(ident) / "record.json").read_text())
        except FileNotFoundError as exc:
            raise LibraryError("Dataset not found", 404) from exc
        if project_id and value["project_id"] != project_id:
            raise LibraryError("Dataset does not belong to this project", 404)
        return value

    def public(self, value):
        return {k: v for k, v in value.items() if k not in ("normalized", "files")}

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
            if len(value["files"]) >= MAX_FILES or name in value["files"]:
                raise LibraryError("File already uploaded or file count limit exceeded", 409)
            destination = contained(self.directory(ident) / "source", name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=".upload-", dir=destination.parent)
            size = 0
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    async for chunk in stream:
                        size += len(chunk)
                        if size + value["total_bytes"] > MAX_BYTES:
                            raise LibraryError("Local uploads support at most 2 GiB", 413)
                        if shutil.disk_usage(self.root).free < 1024**3 + len(chunk):
                            raise LibraryError("Not enough free storage for this upload", 507)
                        handle.write(chunk)
                Path(temporary).replace(destination)
                value["files"][name] = size
                value["total_bytes"] += size
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
        destination = self.directory(ident) / "extracted"
        destination.mkdir()
        total, count = 0, 0
        try:
            with zipfile.ZipFile(source / names[0]) as archive:
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    if (info.external_attr >> 16) & 0o170000 == 0o120000:
                        raise LibraryError("ZIP symlinks are unsupported")
                    target = contained(destination, info.filename)
                    total += info.file_size
                    count += 1
                    if total > MAX_BYTES or count > MAX_FILES:
                        raise LibraryError("Expanded ZIP exceeds 2 GiB / 4096 files", 413)
                    if target.exists():
                        raise LibraryError("ZIP has duplicate file paths")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as original, target.open("xb") as handle:
                        shutil.copyfileobj(original, handle, 1024**2)
            children = list(destination.iterdir())
            if len(children) == 1 and children[0].is_dir():
                value["source_directory"] = "extracted/" + children[0].name
            else:
                value["source_directory"] = "extracted"
            self.save(value)
        except BaseException:
            shutil.rmtree(destination)
            raise

    def source(self, value):
        return contained(self.directory(value["id"]), value.get("source_directory", "source"))

    async def worker(self, payload, timeout=300):
        executable = os.environ.get("FIREBIRD_CPU_READER_PYTHON") or str(reader_python())
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
            await asyncio.to_thread(self.extract, ident)
            value = self.get(ident)
            detection = await self.worker({"operation": "detect", "root": str(self.source(value))})
            # Identical uploads in one project reuse the saved source and labels.
            for record in self.root.glob("*/record.json"):
                existing = json.loads(record.read_text())
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
        for record in self.root.glob("*/record.json"):
            value = json.loads(record.read_text())
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
        rows = [self.public(json.loads(p.read_text())) for p in self.root.glob("*/record.json")]
        rows = [
            r
            for r in rows
            if r["status"] != "uploading" and (not project_id or r["project_id"] == project_id)
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

    async def close(self):
        for task in list(self.tasks.values()):
            task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
