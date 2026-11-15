"""Move a project between workspaces on this computer, and remove a project.

A project is its row in ``workspace.sqlite3`` and everything that names it: jobs, job
submissions and agent sessions (with their events, observations, proposals and receipts),
its datasets in ``dataset-library``, their records in ``dataset-jobs``, its job folders,
its diagnostic records, and the dataset snapshots those reference. Snapshots are stored
once per workspace by content, so one still used by another project is copied, not moved,
and is kept when a project is removed.

Moving within one disk renames folders, so it needs no extra space. The other workspace
must not be open in another OpenJensen app while a project leaves it.
"""

import json
import os
import re
import shutil
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

from filelock import FileLock, Timeout

ACTIVE = {"queued", "running", "cancelling"}
SESSION_TABLES = ("agent_events", "agent_observations", "agent_proposals", "agent_receipts")
# Integer keys the destination assigns itself.
AUTOINCREMENT = {"agent_events": "id", "diagnostic_records": "sequence"}
SNAPSHOT = re.compile(r"[0-9a-f]{64}")


class TransferError(Exception):
    def __init__(self, message, status=409):
        super().__init__(message)
        self.status = status


@dataclass
class Inventory:
    project: dict
    jobs: list = field(default_factory=list)
    submissions: list = field(default_factory=list)
    sessions: list = field(default_factory=list)
    session_rows: dict = field(default_factory=dict)
    datasets: list = field(default_factory=list)  # dataset-library folders
    dataset_jobs: list = field(default_factory=list)  # dataset-jobs record files
    job_folders: list = field(default_factory=list)
    snapshots: list = field(default_factory=list)
    diagnostics: list = field(default_factory=list)

    def summary(self):
        return {
            "id": self.project["id"],
            "name": self.project["name"],
            "datasets": len(self.datasets),
            "jobs": len(self.jobs),
        }


def connect(root, *, readonly=False):
    path = Path(root) / "workspace.sqlite3"
    if not path.is_file():
        raise TransferError("This folder is not an OpenJensen workspace.", 404)
    if readonly:
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=10)
    else:
        connection = sqlite3.connect(path, timeout=30, isolation_level=None)
    connection.row_factory = sqlite3.Row
    return connection


def rows(connection, sql, arguments=()):
    return [dict(row) for row in connection.execute(sql, arguments)]


def schema(connection):
    return connection.execute("SELECT version_num FROM alembic_version").fetchone()[0]


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except OSError, ValueError:
        return None


def project_datasets(root):
    """dataset-library folder -> project id."""
    library = Path(root) / "dataset-library"
    found = {}
    for folder in sorted(library.iterdir()) if library.is_dir() else []:
        record = read_json(folder / "record.json") if folder.is_dir() else None
        if isinstance(record, dict) and isinstance(record.get("project_id"), str):
            found[folder] = record["project_id"]
    return found


def project_dataset_jobs(root):
    """dataset-jobs record -> project id."""
    folder = Path(root) / "dataset-jobs"
    found = {}
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        record = read_json(path)
        if isinstance(record, dict) and isinstance(record.get("project_id"), str):
            found[path] = record["project_id"]
    return found


def referenced_text(root, jobs, datasets, dataset_jobs):
    """Every place a project can name a snapshot: job records, job request and result
    files, dataset records and dataset-job records."""
    parts = [
        job["record"] if isinstance(job["record"], str) else json.dumps(job["record"])
        for job in jobs
    ]
    for job in jobs:
        folder = Path(root) / "jobs" / job["id"]
        for path in folder.glob("*.json") if folder.is_dir() else []:
            parts.append(path.read_text(errors="ignore"))
    for folder in datasets:
        parts.append((folder / "record.json").read_text(errors="ignore"))
    for path in dataset_jobs:
        parts.append(path.read_text(errors="ignore"))
    return "\n".join(parts)


def snapshot_names(text):
    return set(SNAPSHOT.findall(text))


def inventory(root, project_id, connection=None):
    root = Path(root)
    own = connection is None
    connection = connection or connect(root, readonly=True)
    try:
        found = rows(connection, "SELECT * FROM projects WHERE id = ?", (project_id,))
        if not found:
            raise TransferError("That project is not in this workspace.", 404)
        result = Inventory(project=found[0])
        result.jobs = rows(connection, "SELECT * FROM jobs WHERE project_id = ?", (project_id,))
        result.submissions = rows(
            connection, "SELECT * FROM job_submissions WHERE project_id = ?", (project_id,)
        )
        result.sessions = rows(
            connection, "SELECT * FROM agent_sessions WHERE project_id = ?", (project_id,)
        )
        ids = [session["id"] for session in result.sessions]
        for table in SESSION_TABLES:
            result.session_rows[table] = [
                row
                for ident in ids
                for row in rows(connection, f"SELECT * FROM {table} WHERE session_id = ?", (ident,))
            ]
    finally:
        if own:
            connection.close()
    result.datasets = [
        folder for folder, owner in project_datasets(root).items() if owner == project_id
    ]
    result.dataset_jobs = [
        path for path, owner in project_dataset_jobs(root).items() if owner == project_id
    ]
    result.job_folders = [
        root / "jobs" / job["id"] for job in result.jobs if (root / "jobs" / job["id"]).is_dir()
    ]
    names = snapshot_names(referenced_text(root, result.jobs, result.datasets, result.dataset_jobs))
    snapshots = root / "dataset-snapshots"
    result.snapshots = [snapshots / name for name in sorted(names) if (snapshots / name).is_dir()]
    result.diagnostics = diagnostics(root, [job["id"] for job in result.jobs])
    return result


def diagnostics(root, job_ids):
    path = Path(root) / "job-observability.sqlite3"
    if not path.is_file() or not job_ids:
        return []
    with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as database:
        database.row_factory = sqlite3.Row
        try:
            return [
                row
                for ident in job_ids
                for row in rows(
                    database, "SELECT * FROM diagnostic_records WHERE job_id = ?", (ident,)
                )
            ]
        except sqlite3.OperationalError:
            return []


def snapshots_used_elsewhere(root, project_id, connection):
    """Snapshots that other projects of this workspace still reference."""
    others = rows(connection, "SELECT * FROM jobs WHERE project_id != ?", (project_id,))
    datasets = [folder for folder, owner in project_datasets(root).items() if owner != project_id]
    records = [path for path, owner in project_dataset_jobs(root).items() if owner != project_id]
    return snapshot_names(referenced_text(root, others, datasets, records))


def ensure_idle(found):
    busy = [job for job in found.jobs if job["status"] in ACTIVE]
    if busy:
        raise TransferError(
            f"{found.project['name']} has {len(busy)} job(s) still running. "
            "Wait for them to finish or cancel them first."
        )


def owned(root):
    """Hold the workspace's owner lock, as a running app does, while a project leaves it."""
    lock = FileLock(Path(root) / "owner.lock")
    try:
        lock.acquire(timeout=0)
    except Timeout:
        raise TransferError(
            "That workspace is open in another OpenJensen app. Close that app and try again."
        ) from None
    return lock


def move(source, destination, project_id):
    """Move one project from ``source`` into ``destination`` (the running app's workspace)."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise TransferError("The project is already in this workspace.")
    lock = owned(source)
    try:
        with closing(connect(source)) as origin, closing(connect(destination)) as target:
            if schema(origin) != schema(target):
                raise TransferError(
                    "That workspace was made by a different OpenJensen version. Open it once "
                    "with this version, then move the project."
                )
            found = inventory(source, project_id, origin)
            ensure_idle(found)
            refuse_conflicts(found, source, destination, target)
            shared = snapshots_used_elsewhere(source, project_id, origin)
            placed = []
            try:
                for path in found.datasets + found.dataset_jobs + found.job_folders:
                    placed.append(place(path, destination / path.relative_to(source), copy=False))
                for path in found.snapshots:
                    target_path = destination / path.relative_to(source)
                    if (
                        not target_path.exists()
                    ):  # content-addressed: an equal copy is already there
                        placed.append(place(path, target_path, copy=path.name in shared))
                rewrite_paths(found, source, destination)
                insert(target, found, source, destination)
                insert_diagnostics(destination, found.diagnostics)
            except BaseException:
                for record in reversed(placed):
                    undo(record)
                raise
            delete_rows(origin, found)
            delete_diagnostics(source, [job["id"] for job in found.jobs])
            # Originals copied across disks, and snapshots the destination already held.
            for kind, path, _target in placed:
                if kind == "copied-for-move":
                    delete(path)
            for path in found.snapshots:
                if path.name not in shared and path.exists():
                    delete(path)
            return found.summary()
    finally:
        lock.release()


def refuse_conflicts(found, source, destination, target):
    clash = rows(target, "SELECT id FROM projects WHERE id = ?", (found.project["id"],))
    for job in found.jobs:
        clash += rows(target, "SELECT id FROM jobs WHERE id = ?", (job["id"],))
    for session in found.sessions:
        clash += rows(target, "SELECT id FROM agent_sessions WHERE id = ?", (session["id"],))
    paths = [
        destination / path.relative_to(source)
        for path in found.datasets + found.dataset_jobs + found.job_folders
    ]
    if clash or any(path.exists() for path in paths):
        raise TransferError("This project, or part of it, is already in this workspace.")


def place(source, target, *, copy):
    """Move (rename) or copy one file or folder; return what undo needs."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if not copy:
        try:
            os.rename(source, target)
            return ("renamed", source, target)
        except OSError:
            pass  # another disk: copy, and delete the original after the move commits
    if source.is_dir():
        shutil.copytree(source, target, symlinks=True)
    else:
        shutil.copy2(source, target)
    if not copy:
        return ("copied-for-move", source, target)
    return ("copied", source, target)


def undo(record):
    kind, source, target = record
    if kind == "renamed":
        os.rename(target, source)
    elif target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink(missing_ok=True)


def rewrite_paths(found, source, destination):
    """Job folders keep the worker's request, which names files by absolute path."""
    old, new = str(source), str(destination)
    for job in found.jobs:
        folder = destination / "jobs" / job["id"]
        for path in folder.glob("*.json") if folder.is_dir() else []:
            text = path.read_text(errors="ignore")
            if old in text:
                path.write_text(text.replace(old, new))


def value(row, source, destination):
    return {
        key: item.replace(str(source), str(destination)) if isinstance(item, str) else item
        for key, item in row.items()
    }


def insert_rows(connection, table, records, source, destination):
    for record in records:
        record = value(record, source, destination)
        record.pop(AUTOINCREMENT.get(table, None), None)
        columns = ", ".join(record)
        marks = ", ".join("?" for _ in record)
        connection.execute(
            f"INSERT INTO {table} ({columns}) VALUES ({marks})", tuple(record.values())
        )


def insert(connection, found, source, destination):
    connection.execute("BEGIN IMMEDIATE")
    try:
        insert_rows(connection, "projects", [found.project], source, destination)
        insert_rows(connection, "jobs", found.jobs, source, destination)
        insert_rows(connection, "job_submissions", found.submissions, source, destination)
        insert_rows(connection, "agent_sessions", found.sessions, source, destination)
        for table in SESSION_TABLES:
            insert_rows(connection, table, found.session_rows.get(table, []), source, destination)
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise


def delete_rows(connection, found):
    project_id = found.project["id"]
    ids = [session["id"] for session in found.sessions]
    connection.execute("BEGIN IMMEDIATE")
    try:
        for table in SESSION_TABLES:
            for ident in ids:
                connection.execute(f"DELETE FROM {table} WHERE session_id = ?", (ident,))
        connection.execute("DELETE FROM agent_sessions WHERE project_id = ?", (project_id,))
        connection.execute("DELETE FROM job_submissions WHERE project_id = ?", (project_id,))
        connection.execute("DELETE FROM jobs WHERE project_id = ?", (project_id,))
        connection.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise


def insert_diagnostics(root, records):
    path = Path(root) / "job-observability.sqlite3"
    if not records or not path.is_file():
        return
    with closing(sqlite3.connect(path, timeout=30)) as database:
        try:
            with database:
                insert_rows(database, "diagnostic_records", records, "", "")
        except sqlite3.OperationalError:
            pass  # diagnostics are history; a schema difference must not block the move


def delete_diagnostics(root, job_ids):
    path = Path(root) / "job-observability.sqlite3"
    if not job_ids or not path.is_file():
        return
    with closing(sqlite3.connect(path, timeout=30)) as database:
        try:
            with database:
                for ident in job_ids:
                    database.execute("DELETE FROM diagnostic_records WHERE job_id = ?", (ident,))
        except sqlite3.OperationalError:
            pass


def remove(root, project_id):
    """Delete a project and its data from the workspace ``root``."""
    root = Path(root).resolve()
    with closing(connect(root)) as connection:
        found = inventory(root, project_id, connection)
        ensure_idle(found)
        shared = snapshots_used_elsewhere(root, project_id, connection)
        delete_rows(connection, found)
    delete_diagnostics(root, [job["id"] for job in found.jobs])
    for path in found.datasets + found.job_folders + found.dataset_jobs:
        delete(path)
    for path in found.snapshots:
        if path.name not in shared:
            delete(path)
    return found.summary()


def delete(path):
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def nearby(current, home=None):
    """Other OpenJensen workspaces on this computer and the projects each holds."""
    home = Path(home) if home else Path.home()
    current = Path(current).resolve()
    share = home / ".local" / "share"
    candidates = [home / ".jensen", *sorted(share.glob("*")), *sorted(share.glob("*/*"))]
    with closing(connect(current, readonly=True)) as own:
        version = schema(own)
        here = {row["id"] for row in rows(own, "SELECT id FROM projects")}
    found = []
    for root in candidates:
        try:
            root = root.resolve()
            if root == current or not (root / "workspace.sqlite3").is_file():
                continue
            with closing(connect(root, readonly=True)) as connection:
                projects = rows(
                    connection, "SELECT id, name, created_at FROM projects ORDER BY created_at"
                )
                same = schema(connection) == version
                counts = {
                    row["project_id"]: row["count"]
                    for row in rows(
                        connection,
                        "SELECT project_id, count(*) AS count FROM jobs GROUP BY project_id",
                    )
                }
        except OSError, sqlite3.Error, TransferError:
            continue
        if not projects:
            continue
        datasets = list(project_datasets(root).values())
        open_elsewhere = lock_held(root)
        found.append(
            {
                "path": str(root),
                "label": "~/" + str(root.relative_to(home))
                if root.is_relative_to(home)
                else str(root),
                "available": same and not open_elsewhere,
                "message": None
                if same and not open_elsewhere
                else "Open in another OpenJensen app. Close it to move projects from here."
                if open_elsewhere
                else "Made by a different OpenJensen version.",
                "projects": [
                    {
                        **project,
                        "datasets": datasets.count(project["id"]),
                        "jobs": counts.get(project["id"], 0),
                        # A backup of a workspace lists projects this app already holds.
                        "here": project["id"] in here,
                    }
                    for project in projects
                ],
            }
        )
    return found


def lock_held(root):
    lock = FileLock(Path(root) / "owner.lock")
    try:
        lock.acquire(timeout=0)
    except Timeout:
        return True
    lock.release()
    return False
