"""Offline, opt-in desktop preflight/backup foundation; never starts an application.

No entrypoint or native IPC uses this module yet. Original files are opened read-only;
flock coordinates with the existing Unix FileLock. A separate sibling maintenance lock
serializes this helper. It neither migrates data nor changes a workspace marker.
Limits bound cooperative work/SQLite queries, not a stalled kernel/filesystem call.
"""

import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import stat
import time
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath


class MaintenanceError(ValueError):
    """Original workspace must remain preserved; no activation is implied."""


@dataclass(frozen=True)
class Limits:
    entries: int = 20_000
    bytes: int = 2 * 1024**3
    records: int = 10_000
    json_bytes: int = 4 * 1024**2
    seconds: int = 120
    spare_bytes: int = 64 * 1024**2

    def __post_init__(self):
        maxima = (20_000, 2 * 1024**3, 10_000, 4 * 1024**2, 600, 1024**3)
        for value, maximum in zip(asdict(self).values(), maxima, strict=True):
            if type(value) is not int or not 0 < value <= maximum:
                raise MaintenanceError("Invalid maintenance budget")


class Budget:
    def __init__(self, limits):
        self.limits = limits
        self.end = time.monotonic() + limits.seconds

    def check(self):
        if time.monotonic() >= self.end:
            raise MaintenanceError("Offline maintenance deadline exceeded")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def parse_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise MaintenanceError("Duplicate JSON field")
            result[key] = value
        return result

    def finite(value):
        number = float(value)
        if not math.isfinite(number):
            raise MaintenanceError("Nonfinite JSON number")
        return number

    def constant(_):
        raise MaintenanceError("Nonfinite JSON constant")

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_float=finite,
            parse_constant=constant,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise MaintenanceError("Invalid stored UTF-8 JSON") from exc


def directory(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise MaintenanceError("An absolute trusted directory is required")
    for parent in (path, *path.parents):
        info = parent.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise MaintenanceError("Directory links and special entries are unsupported")
    return path


def identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def opened(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    stream = os.fdopen(descriptor, "rb")
    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
        stream.close()
        raise MaintenanceError("Expected a regular file")
    return stream


def read_json(path, limit):
    with opened(path) as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise MaintenanceError("Stored JSON exceeds its bound")
    return parse_json(raw)


@contextmanager
def locks(workspace):
    # Import only on explicit invocation; unsupported platforms fail before any writes.
    try:
        import fcntl
    except ImportError as exc:
        raise MaintenanceError(
            "This offline foundation supports Unix workspace locks only"
        ) from exc
    handles = []
    try:
        # Check the original lock before creating the separate helper lock.
        owner = opened(workspace / "owner.lock")
        handles.append(owner)
        path = workspace.parent / ".desktop-maintenance.lock"
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        helper = os.fdopen(descriptor, "r+b")
        handles.append(helper)
        if not stat.S_ISREG(os.fstat(helper.fileno()).st_mode):
            raise MaintenanceError("Invalid maintenance lock")
        for handle in (helper, owner):
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise MaintenanceError("Workspace or maintenance is already owned") from exc
        yield
    finally:
        for handle in reversed(handles):
            handle.close()


def inventory(root, budget):
    result = {}
    total = 0

    def visit(folder):
        nonlocal total
        budget.check()
        with os.scandir(folder) as entries:
            for entry in entries:
                budget.check()
                path = Path(entry.path)
                name = path.relative_to(root).as_posix()
                if len(name.encode()) > 4096 or len(Path(name).parts) > 64:
                    raise MaintenanceError("Workspace path exceeds its bound")
                before = entry.stat(follow_symlinks=False)
                mode = stat.S_IMODE(before.st_mode)
                if mode & 0o7000:
                    raise MaintenanceError("Special permission bits are unsupported")
                if stat.S_ISDIR(before.st_mode):
                    result[name] = {"kind": "directory", "mode": mode}
                elif stat.S_ISREG(before.st_mode):
                    total += before.st_size
                    if total > budget.limits.bytes:
                        raise MaintenanceError("Workspace exceeds backup byte budget")
                    digest = hashlib.sha256()
                    with opened(path) as stream:
                        if identity(os.fstat(stream.fileno())) != identity(before):
                            raise MaintenanceError("Workspace changed during inventory")
                        seen = 0
                        while chunk := stream.read(65536):
                            budget.check()
                            seen += len(chunk)
                            if seen > before.st_size:
                                raise MaintenanceError("Workspace file grew during inventory")
                            digest.update(chunk)
                        if seen != before.st_size or identity(
                            os.fstat(stream.fileno())
                        ) != identity(before):
                            raise MaintenanceError("Workspace changed during inventory")
                    if identity(path.lstat()) != identity(before):
                        raise MaintenanceError("Workspace entry changed during inventory")
                    result[name] = {
                        "kind": "file",
                        "mode": mode,
                        "bytes": seen,
                        "sha256": digest.hexdigest(),
                    }
                else:
                    raise MaintenanceError("Workspace links and special entries are unsupported")
                if len(result) > budget.limits.entries:
                    raise MaintenanceError("Workspace exceeds backup entry budget")
                if stat.S_ISDIR(before.st_mode):
                    visit(path)

    visit(root)
    return dict(sorted(result.items()))


def marker(workspace, budget):
    value = read_json(workspace / "desktop-owner.json", min(4096, budget.limits.json_bytes))
    expected = {
        "schema_version",
        "owner",
        "build_id",
        "resources_sha256",
        "payload_identity_sha256",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 2
        or value["owner"] != "openjensen-desktop"
        or any(
            not isinstance(value[key], str) or not re.fullmatch(pattern, value[key])
            for key, pattern in (
                ("build_id", r"[a-f0-9]{40}"),
                ("resources_sha256", r"[a-f0-9]{64}"),
                ("payload_identity_sha256", r"[a-f0-9]{64}"),
            )
        )
    ):
        raise MaintenanceError("Unsupported desktop ownership marker")
    return value


def relative_path(value):
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or PurePosixPath(value).is_absolute()
        or value != PurePosixPath(value).as_posix()
        or any(part in (".", "..") for part in PurePosixPath(value).parts)
    ):
        raise MaintenanceError("Invalid workspace-relative artifact path")
    return value


def local_artifact(workspace, artifact, files, budget):
    prefix = relative_path(artifact.path)
    manifest_name = prefix + "/manifest.json"
    manifest_row = files.get(manifest_name, {})
    if manifest_row.get("sha256") != artifact.manifest_sha256:
        raise MaintenanceError("Registered artifact manifest is missing or changed")
    manifest = read_json(workspace / manifest_name, budget.limits.json_bytes)
    if (
        not isinstance(manifest, dict)
        or not isinstance(manifest.get("files"), dict)
        or not manifest["files"]
    ):
        raise MaintenanceError("Invalid local artifact inventory")
    expected = {}
    for name, digest in manifest["files"].items():
        name = relative_path(name)
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise MaintenanceError("Invalid artifact digest")
        expected[prefix + "/" + name] = digest
    actual = {
        name: row
        for name, row in files.items()
        if name.startswith(prefix + "/") and row["kind"] == "file" and name != manifest_name
    }
    if set(actual) != set(expected) or any(
        row["sha256"] != expected[name] for name, row in actual.items()
    ):
        raise MaintenanceError("Registered artifact payload changed")
    if sum(row["bytes"] for row in actual.values()) != artifact.file_bytes:
        raise MaintenanceError("Registered artifact byte count differs")


def validate_database(workspace, files, budget):
    # Schema-only imports: never import API/Execution/Storage or a runtime registry.
    from vla_platform.contracts import TERMINAL, Job, Project
    from vla_platform.lifecycle.contracts import LifecycleResult

    if files.get("workspace.sqlite3", {}).get("kind") != "file":
        raise MaintenanceError("Workspace database is missing")
    for name in files:
        if name in {"workspace.sqlite3-wal", "workspace.sqlite3-shm", "workspace.sqlite3-journal"}:
            raise MaintenanceError("Unsettled SQLite journal requires separate recovery")
        if Path(name).name in {"sky-state.json", "remote.json"}:
            raise MaintenanceError(
                "Remote dispatch or descriptor requires separate compatibility review"
            )
    uri = (workspace / "workspace.sqlite3").as_uri() + "?mode=ro&immutable=1"
    try:
        with closing(sqlite3.connect(uri, uri=True, timeout=0)) as database:
            database.set_progress_handler(lambda: int(time.monotonic() >= budget.end), 1000)
            database.execute("PRAGMA query_only=ON")
            database.execute("PRAGMA trusted_schema=OFF")
            if database.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise MaintenanceError("SQLite integrity check failed")
            schema = database.execute(
                "SELECT name,type FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
            ).fetchall()
            if set(schema) != {
                ("projects", "table"),
                ("jobs", "table"),
                ("alembic_version", "table"),
                ("ix_jobs_project_id", "index"),
            }:
                raise MaintenanceError("Unsupported SQLite schema objects")
            if database.execute("SELECT version_num FROM alembic_version").fetchall() != [
                ("0001",)
            ]:
                raise MaintenanceError("Unsupported SQLite migration revision")
            for table, columns in (
                ("projects", ["id", "name", "created_at"]),
                ("jobs", ["id", "project_id", "status", "record"]),
                ("alembic_version", ["version_num"]),
            ):
                schema_columns = database.execute(f"PRAGMA table_info({table})").fetchall()
                if [item[1] for item in schema_columns] != columns or schema_columns[0][5] != 1:
                    raise MaintenanceError("Unsupported SQLite table layout")
            if [item[2] for item in database.execute("PRAGMA index_info(ix_jobs_project_id)")] != [
                "project_id"
            ]:
                raise MaintenanceError("Unsupported job index")
            for table, columns in (
                ("projects", {"id": 4096, "name": 400, "created_at": 100}),
                (
                    "jobs",
                    {
                        "id": 4096,
                        "project_id": 4096,
                        "status": 32,
                        "record": budget.limits.json_bytes,
                    },
                ),
            ):
                if (
                    database.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                    > budget.limits.records
                ):
                    raise MaintenanceError("Excessive stored records")
                for column, maximum in columns.items():
                    length = database.execute(
                        f"SELECT max(length(CAST({column} AS BLOB))) FROM {table}"
                    ).fetchone()[0]
                    if length is not None and length > maximum:
                        raise MaintenanceError("Stored field exceeds its byte bound")
            projects = {}
            for row in database.execute("SELECT id,name,created_at FROM projects"):
                budget.check()
                project = Project.model_validate(
                    dict(zip(("id", "name", "created_at"), row, strict=True)), strict=True
                )
                if (
                    project.id != row[0]
                    or project.id in projects
                    or len(projects) >= budget.limits.records
                ):
                    raise MaintenanceError("Invalid or excessive project records")
                projects[project.id] = project
            count = 0
            for row in database.execute(
                "SELECT id,project_id,status,length(CAST(record AS BLOB)) FROM jobs"
            ):
                budget.check()
                count += 1
                job_id, project_id, status, size = row
                if (
                    count > budget.limits.records
                    or type(size) is not int
                    or size > budget.limits.json_bytes
                ):
                    raise MaintenanceError("Job records exceed the maintenance bound")
                value = database.execute(
                    "SELECT record FROM jobs WHERE id=?", (job_id,)
                ).fetchone()[0]
                if not isinstance(value, str):
                    raise MaintenanceError("Stored job is not JSON text")
                job = Job.model_validate(parse_json(value.encode()), strict=True)
                if (job.id, job.project_id, job.status) != (
                    job_id,
                    project_id,
                    status,
                ) or project_id not in projects:
                    raise MaintenanceError("Indexed and stored job identity differ")
                if status not in TERMINAL:
                    raise MaintenanceError("Active jobs prevent offline maintenance")
                if job.compute_target is not None or job.simulation_target is not None:
                    raise MaintenanceError("Remote target requires separate compatibility review")
                if isinstance(job.result, LifecycleResult):
                    for artifact in job.result.artifacts:
                        if artifact.project_id != project_id or artifact.job_id != job_id:
                            raise MaintenanceError("Registered artifact ownership differs")
                        local_artifact(workspace, artifact, files, budget)
                snapshot = getattr(job.result, "snapshot", None)
                if snapshot is not None:
                    # Native snapshot semantics/decoder acceptance are deliberately not duplicated.
                    raise MaintenanceError(
                        "Native dataset snapshots require separate compatibility review"
                    )
            budget.check()
            return {
                "alembic_revision": "0001",
                "projects": len(projects),
                "jobs": count,
                "all_jobs_terminal": True,
                "validation": "stored_schema_and_local_artifact_inventory",
            }
    except (sqlite3.Error, ValueError, TypeError) as exc:
        if isinstance(exc, MaintenanceError):
            raise
        raise MaintenanceError("Stored database/record compatibility failed") from exc


def copy_files(workspace, destination, files, budget):
    for name, row in files.items():
        budget.check()
        target = destination / name
        if row["kind"] == "directory":
            target.mkdir(mode=0o700)
            continue
        with opened(workspace / name) as source, target.open("xb") as output:
            digest = hashlib.sha256()
            seen = 0
            while chunk := source.read(65536):
                budget.check()
                seen += len(chunk)
                if seen > row["bytes"]:
                    raise MaintenanceError("Workspace grew during backup")
                output.write(chunk)
                digest.update(chunk)
            if seen != row["bytes"] or digest.hexdigest() != row["sha256"]:
                raise MaintenanceError("Workspace changed during backup")
            os.fchmod(output.fileno(), row["mode"])
            output.flush()
            os.fsync(output.fileno())
    for name, row in reversed(list(files.items())):
        if row["kind"] == "directory":
            os.chmod(destination / name, row["mode"])
            sync_directory(destination / name)
    sync_directory(destination)


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prepare_backup(workspace: Path, output: Path, *, limits: Limits = Limits()) -> dict:
    """Validate a stopped desktop workspace and create a NEW complete backup.

    No migration or activation. Invalid preflight creates no output. Copy/publication
    failures retain an explicitly incomplete output directory for inspection; no receipt
    is success unless this function returns. Original data and marker bytes are unchanged.
    """
    workspace = directory(workspace)
    output = Path(output)
    parent = directory(output.parent)
    if (
        not output.is_absolute()
        or ".." in output.parts
        or output == workspace
        or output.is_relative_to(workspace)
        or workspace.is_relative_to(output)
        or output.exists()
        or output.is_symlink()
    ):
        raise MaintenanceError("Backup must be a new disjoint absolute directory")
    budget = Budget(limits)
    with locks(workspace):
        before = inventory(workspace, budget)
        original_marker = marker(workspace, budget)
        database = validate_database(workspace, before, budget)
        if inventory(workspace, budget) != before:
            raise MaintenanceError("Workspace changed during preflight")
        total = sum(row.get("bytes", 0) for row in before.values())
        if shutil.disk_usage(parent).free < total + limits.spare_bytes:
            raise MaintenanceError("Insufficient disk space for the complete backup")
        output.mkdir(mode=0o700)
        copied = output / "workspace"
        copied.mkdir(mode=0o700)
        copy_files(workspace, copied, before, budget)
        if inventory(copied, budget) != before or inventory(workspace, budget) != before:
            raise MaintenanceError("Backup or source inventory changed")
        if validate_database(copied, before, budget) != database:
            raise MaintenanceError("Copied database compatibility differs")
        receipt = {
            "schema_version": 1,
            "operation": "desktop.workspace.backup",
            "complete": True,
            "workspace_path_sha256": hashlib.sha256(os.fsencode(workspace)).hexdigest(),
            "marker": original_marker,
            "database": database,
            "files": before,
            "inventory_sha256": hashlib.sha256(canonical(before)).hexdigest(),
            "file_bytes": total,
            "limits": asdict(limits),
            "source_unchanged": True,
            "migration_performed": False,
            "activation_performed": False,
            "restore_or_upgrade_verified": False,
        }
        budget.check()
        receipt_path = output / "receipt.json"
        published_identity = None
        stream = None
        try:
            stream = receipt_path.open("xb")
            published_identity = os.fstat(stream.fileno())
            stream.write(canonical(receipt) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
            sync_directory(output)
            sync_directory(parent)
        except BaseException:
            # Newly owned output was exclusive; leave payload partial, never a success receipt.
            if published_identity is not None:
                try:
                    current = receipt_path.lstat()
                    if (current.st_dev, current.st_ino) == (
                        published_identity.st_dev,
                        published_identity.st_ino,
                    ) and stat.S_ISREG(current.st_mode):
                        receipt_path.unlink()
                except FileNotFoundError:
                    pass
            raise
        finally:
            if stream is not None:
                stream.close()
        return receipt
