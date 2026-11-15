"""Move a project between two real workspaces, and remove one, with every piece it owns."""

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from filelock import FileLock
from vla_platform import project_transfer
from vla_platform.project_transfer_api import router
from vla_platform.projects import Projects
from vla_platform.storage import Storage

SHARED = "a" * 64  # a snapshot both projects of the source use
OWN = "b" * 64  # a snapshot only the moving project uses


def workspace(root: Path) -> Path:
    root.mkdir(parents=True)
    storage = Storage(root)
    asyncio.run(storage.initialize())
    asyncio.run(storage.close())
    with sqlite3.connect(root / "job-observability.sqlite3") as database:
        database.execute(
            "CREATE TABLE diagnostic_records (sequence INTEGER PRIMARY KEY AUTOINCREMENT, "
            "job_id TEXT NOT NULL, message TEXT)"
        )
    return root


def add_project(root, ident, name, *, job_status="succeeded", snapshots=(SHARED,)):
    job = f"job-{ident}"
    session = f"session-{ident}"
    with sqlite3.connect(root / "workspace.sqlite3") as database:
        database.execute(
            "INSERT INTO projects VALUES (?, ?, ?)", (ident, name, "2026-10-31T00:00:00+00:00")
        )
        record = {"id": job, "project_id": ident, "result": {"snapshots": list(snapshots)}}
        database.execute(
            "INSERT INTO jobs VALUES (?, ?, ?, ?)", (job, ident, job_status, json.dumps(record))
        )
        database.execute(
            "INSERT INTO job_submissions VALUES (?, ?, ?, 1, ?, '{}', ?, '{}', ?)",
            (ident, "policy.finetune", f"key-{ident}", "0" * 64, job, "2026-10-31"),
        )
        database.execute(
            "INSERT INTO agent_sessions VALUES (?, ?, 'closed', '{}', '{}')", (session, ident)
        )
        database.execute(
            "INSERT INTO agent_events (session_id, kind, created_at, record) "
            "VALUES (?, 'note', 'now', '{}')",
            (session,),
        )
        database.execute(
            "INSERT INTO agent_observations VALUES (?, ?, ?, '{}')",
            (f"obs-{ident}", session, ident),
        )
        database.execute(
            "INSERT INTO agent_proposals VALUES (?, ?, 'request', ?, '{}')",
            (f"proposal-{ident}", session, "1" * 64),
        )
        database.execute(
            "INSERT INTO agent_receipts VALUES (?, ?, ?, '{}')",
            (f"receipt-{ident}", session, f"proposal-{ident}"),
        )
    with sqlite3.connect(root / "job-observability.sqlite3") as database:
        database.execute(
            "INSERT INTO diagnostic_records (job_id, message) VALUES (?, 'ok')", (job,)
        )
    dataset = root / "dataset-library" / f"dataset-{ident}"
    (dataset / "source").mkdir(parents=True)
    (dataset / "source" / "episode 0.parquet").write_bytes(b"episode bytes")
    (dataset / "record.json").write_text(json.dumps({"id": dataset.name, "project_id": ident}))
    (root / "dataset-jobs").mkdir(exist_ok=True)
    (root / "dataset-jobs" / f"import-{ident}.json").write_text(
        json.dumps({"id": f"import-{ident}", "project_id": ident, "dataset_id": dataset.name})
    )
    folder = root / "jobs" / job
    folder.mkdir(parents=True)
    (folder / "request.json").write_text(
        json.dumps({"dataset": str(root / "dataset-snapshots" / snapshots[0] / "data")})
    )
    (folder / "checkpoint.bin").write_bytes(b"weights")
    for name in snapshots:
        (root / "dataset-snapshots" / name / "data").mkdir(parents=True, exist_ok=True)
        (root / "dataset-snapshots" / name / "data" / "file-000.parquet").write_bytes(b"frames")
    return job


def table(root, sql):
    with sqlite3.connect(root / "workspace.sqlite3") as database:
        return database.execute(sql).fetchall()


@pytest.fixture
def two(tmp_path):
    source = workspace(tmp_path / "old workspace")
    destination = workspace(tmp_path / "new")
    add_project(source, "pen", "SO101 pen pickup", snapshots=(SHARED, OWN))
    add_project(source, "pusht", "PushT benchmark")
    add_project(destination, "duck", "duck pickup", snapshots=("c" * 64,))
    return source, destination


def test_move_brings_every_piece_and_leaves_the_rest(two):
    source, destination = two
    moved = project_transfer.move(source, destination, "pen")
    assert moved == {"id": "pen", "name": "SO101 pen pickup", "datasets": 1, "jobs": 1}
    assert [row[0] for row in table(destination, "SELECT name FROM projects ORDER BY name")] == [
        "SO101 pen pickup",
        "duck pickup",
    ]
    for name in ("jobs", "job_submissions", "agent_sessions", "agent_observations"):
        assert table(destination, f"SELECT count(*) FROM {name}")[0][0] == 2
        assert table(source, f"SELECT count(*) FROM {name}")[0][0] == 1
    for name in ("agent_events", "agent_proposals", "agent_receipts"):
        assert table(destination, f"SELECT count(*) FROM {name}")[0][0] == 2
    assert (
        destination / "dataset-library/dataset-pen/source/episode 0.parquet"
    ).read_bytes() == b"episode bytes"
    assert not (source / "dataset-library/dataset-pen").exists()
    assert (destination / "dataset-jobs/import-pen.json").exists()
    assert (destination / "jobs/job-pen/checkpoint.bin").read_bytes() == b"weights"
    request = json.loads((destination / "jobs/job-pen/request.json").read_text())
    assert request["dataset"] == str(destination / "dataset-snapshots" / SHARED / "data")
    # The snapshot PushT still uses stays; the one only the pen project used moves.
    assert (source / "dataset-snapshots" / SHARED).is_dir()
    assert (destination / "dataset-snapshots" / SHARED).is_dir()
    assert not (source / "dataset-snapshots" / OWN).exists()
    assert (destination / "dataset-snapshots" / OWN).is_dir()
    with sqlite3.connect(destination / "job-observability.sqlite3") as database:
        assert database.execute(
            "SELECT job_id FROM diagnostic_records ORDER BY job_id"
        ).fetchall() == [("job-duck",), ("job-pen",)]
    with sqlite3.connect(source / "job-observability.sqlite3") as database:
        assert database.execute("SELECT job_id FROM diagnostic_records").fetchall() == [
            ("job-pusht",)
        ]
    # What stays behind still works as a workspace.
    assert table(source, "SELECT name FROM projects") == [("PushT benchmark",)]
    assert (source / "dataset-library/dataset-pusht/record.json").exists()


def test_move_refuses_a_workspace_open_in_another_app(two):
    source, destination = two
    lock = FileLock(source / "owner.lock")
    lock.acquire()
    try:
        with pytest.raises(project_transfer.TransferError, match="open in another OpenJensen app"):
            project_transfer.move(source, destination, "pen")
    finally:
        lock.release()
    assert table(destination, "SELECT count(*) FROM projects")[0][0] == 1


def test_move_refuses_running_jobs_and_a_project_already_here(tmp_path):
    source = workspace(tmp_path / "old")
    destination = workspace(tmp_path / "new")
    add_project(source, "busy", "Busy", job_status="running")
    with pytest.raises(project_transfer.TransferError, match="still running"):
        project_transfer.move(source, destination, "busy")
    add_project(source, "duck", "duck pickup")
    add_project(destination, "duck", "duck pickup")
    with pytest.raises(project_transfer.TransferError, match="already in this workspace"):
        project_transfer.move(source, destination, "duck")
    assert (source / "dataset-library/dataset-duck").exists()


def test_move_refuses_another_schema_version(two):
    source, destination = two
    with sqlite3.connect(source / "workspace.sqlite3") as database:
        database.execute("UPDATE alembic_version SET version_num = '0002'")
    with pytest.raises(project_transfer.TransferError, match="different OpenJensen version"):
        project_transfer.move(source, destination, "pen")


def test_failed_move_puts_everything_back(two, monkeypatch):
    source, destination = two

    def fail(*args):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(project_transfer, "insert", fail)
    with pytest.raises(sqlite3.OperationalError):
        project_transfer.move(source, destination, "pen")
    assert (source / "dataset-library/dataset-pen/source/episode 0.parquet").exists()
    assert (source / "jobs/job-pen/checkpoint.bin").exists()
    assert (source / "dataset-snapshots" / OWN).is_dir()
    assert not (destination / "dataset-library/dataset-pen").exists()
    assert not (destination / "dataset-snapshots" / OWN).exists()
    assert table(source, "SELECT count(*) FROM projects")[0][0] == 2
    assert table(destination, "SELECT count(*) FROM projects")[0][0] == 1


def test_remove_deletes_the_project_but_keeps_what_others_use(two):
    source, _ = two
    removed = project_transfer.remove(source, "pusht")
    assert removed["name"] == "PushT benchmark"
    assert table(source, "SELECT name FROM projects") == [("SO101 pen pickup",)]
    assert table(source, "SELECT count(*) FROM jobs WHERE project_id = 'pusht'")[0][0] == 0
    assert table(source, "SELECT count(*) FROM agent_events")[0][0] == 1
    assert not (source / "dataset-library/dataset-pusht").exists()
    assert not (source / "jobs/job-pusht").exists()
    assert not (source / "dataset-jobs/import-pusht.json").exists()
    assert (source / "dataset-snapshots" / SHARED).is_dir()  # the pen project still uses it
    project_transfer.remove(source, "pen")
    assert not (source / "dataset-snapshots" / SHARED).exists()
    assert not (source / "dataset-snapshots" / OWN).exists()
    assert table(source, "SELECT count(*) FROM projects")[0][0] == 0


def test_remove_refuses_while_a_job_runs(tmp_path):
    root = workspace(tmp_path / "w")
    add_project(root, "busy", "Busy", job_status="queued")
    with pytest.raises(project_transfer.TransferError, match="still running"):
        project_transfer.remove(root, "busy")
    assert (root / "dataset-library/dataset-busy").exists()


def test_nearby_lists_other_workspaces_on_this_computer(tmp_path):
    home = tmp_path / "home"
    share = home / ".local" / "share"
    current = workspace(share / "openjensen")
    other = workspace(share / "jensen" / "data")
    add_project(other, "pen", "SO101 pen pickup")
    workspace(share / "empty")  # no projects: nothing to move
    (share / "unrelated").mkdir()
    found = project_transfer.nearby(current, home)
    assert found == [
        {
            "path": str(other.resolve()),
            "label": "~/.local/share/jensen/data",
            "available": True,
            "message": None,
            "projects": [
                {
                    "id": "pen",
                    "name": "SO101 pen pickup",
                    "created_at": "2026-10-31T00:00:00+00:00",
                    "datasets": 1,
                    "jobs": 1,
                    "here": False,
                }
            ],
        }
    ]
    add_project(current, "pen", "SO101 pen pickup")  # e.g. moved earlier; the other is a backup
    [entry] = project_transfer.nearby(current, home)
    assert entry["projects"][0]["here"] is True
    lock = FileLock(other / "owner.lock")
    lock.acquire()
    try:
        [entry] = project_transfer.nearby(current, home)
        assert entry["available"] is False
        assert "Close it" in entry["message"]
    finally:
        lock.release()


def test_page_moves_a_project_in_and_removes_one(tmp_path, monkeypatch):
    home = tmp_path / "home"
    share = home / ".local" / "share"
    current = workspace(share / "openjensen")
    other = workspace(share / "jensen" / "data")
    add_project(other, "pen", "SO101 pen pickup")
    add_project(current, "duck", "duck pickup", snapshots=("c" * 64,))
    monkeypatch.setenv("HOME", str(home))
    storage = Storage(current)

    class Execution:
        class settings:
            data_dir = current

    app = FastAPI()
    app.include_router(router)
    app.state.execution = Execution()
    app.state.projects = Projects(storage)
    client = TestClient(app)

    [listed] = client.get("/api/v1/projects/nearby").json()
    assert listed["projects"][0]["name"] == "SO101 pen pickup"
    refused = client.post(
        "/api/v1/projects/move", json={"workspace": str(tmp_path), "project_id": "pen"}
    )
    assert refused.status_code == 404
    moved = client.post(
        "/api/v1/projects/move", json={"workspace": listed["path"], "project_id": "pen"}
    )
    assert moved.status_code == 200
    assert moved.json()["name"] == "SO101 pen pickup"
    assert client.get("/api/v1/projects/nearby").json() == []  # nothing left to move

    removed = client.delete("/api/v1/projects/duck")
    assert removed.status_code == 200
    assert removed.json() == {"id": "duck", "name": "duck pickup", "datasets": 1, "jobs": 1}
    assert [p.name for p in asyncio.run(app.state.projects.list())] == ["SO101 pen pickup"]
    assert client.delete("/api/v1/projects/duck").status_code == 404
    asyncio.run(storage.close())
