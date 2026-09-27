"""Disposable SQLite/files only; no API, runtime, provider, package build or GUI."""

import hashlib
import importlib.util
import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest
from filelock import FileLock

SIDE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "workspace_maintenance", SIDE / "workspace_maintenance.py"
)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "owned"
    root.mkdir(mode=0o700)
    (root / "owner.lock").touch(mode=0o600)
    (root / "desktop-owner.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "owner": "openjensen-desktop",
                "build_id": "a" * 40,
                "resources_sha256": "b" * 64,
                "payload_identity_sha256": "c" * 64,
            }
        )
    )
    with sqlite3.connect(root / "workspace.sqlite3") as db:
        db.executescript("""
            CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL PRIMARY KEY);
            INSERT INTO alembic_version VALUES ('0001');
            CREATE TABLE projects (id VARCHAR NOT NULL PRIMARY KEY,
                name VARCHAR NOT NULL, created_at VARCHAR NOT NULL);
            CREATE TABLE jobs (id VARCHAR NOT NULL PRIMARY KEY, project_id VARCHAR NOT NULL,
                status VARCHAR NOT NULL, record JSON NOT NULL);
            CREATE INDEX ix_jobs_project_id ON jobs (project_id);
            INSERT INTO projects VALUES
                ('project', 'Generated offline fixture', '2026-09-27T00:00:00+00:00');
        """)
        job = {
            "id": "job",
            "project_id": "project",
            "kind": "dataset.inspect",
            "status": "failed",
            "request": {
                "source": "huggingface",
                "repo_id": "fixture/generated",
                "revision": "main",
            },
            "created_at": "2026-09-27T00:00:00+00:00",
            "updated_at": "2026-09-27T00:00:00+00:00",
            "result": None,
            "error": "Generated terminal fixture; no network request.",
        }
        db.execute(
            "INSERT INTO jobs VALUES (?,?,?,?)", ("job", "project", "failed", json.dumps(job))
        )
    (root / "jobs/job").mkdir(parents=True)
    (root / "jobs/job/events.jsonl").write_text('{"event":"fixture"}\n')
    (root / "private-settings.json").write_text('{"example":"preserved bytes, never activated"}')
    return root


def hashes(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file() and not p.is_symlink()
    }


def update_job(root, edit):
    with sqlite3.connect(root / "workspace.sqlite3") as db:
        value = json.loads(db.execute("SELECT record FROM jobs").fetchone()[0])
        edit(value)
        db.execute("UPDATE jobs SET record=?", (json.dumps(value),))


def run(workspace, **kwargs):
    return m.prepare_backup(workspace, workspace.parent / "new-backup", **kwargs)


def test_complete_backup_preserves_database_all_files_modes_and_original(workspace):
    before = hashes(workspace)
    os.chmod(workspace / "private-settings.json", 0o600)
    (workspace / "empty").mkdir(mode=0o750)
    modules_before = set(sys.modules)
    result = run(workspace)
    assert result["complete"] is True
    assert result["database"]["jobs"] == result["database"]["projects"] == 1
    assert result["migration_performed"] is result["activation_performed"] is False
    copied = workspace.parent / "new-backup/workspace"
    assert hashes(copied) == before == hashes(workspace)
    assert copied.joinpath("private-settings.json").stat().st_mode & 0o777 == 0o600
    assert copied.joinpath("empty").stat().st_mode & 0o777 == 0o750
    assert json.loads((copied.parent / "receipt.json").read_bytes()) == result
    assert result["inventory_sha256"] == hashlib.sha256(m.canonical(result["files"])).hexdigest()
    assert not {
        "vla_platform.api",
        "vla_platform.execution",
        "vla_platform.lifecycle.service",
    } & (set(sys.modules) - modules_before)


@pytest.mark.parametrize("kind", ["directory", "file", "symlink"])
def test_existing_output_never_replaced(workspace, kind):
    output = workspace.parent / "new-backup"
    if kind == "directory":
        output.mkdir()
    elif kind == "file":
        output.write_text("existing")
    else:
        output.symlink_to(workspace.parent / "missing")
    before = hashes(workspace)
    with pytest.raises(m.MaintenanceError):
        run(workspace)
    assert hashes(workspace) == before
    assert output.exists() or output.is_symlink()


@pytest.mark.parametrize("lock_name", ["owner.lock", ".desktop-maintenance.lock"])
def test_existing_real_filelock_prevents_backup(workspace, lock_name):
    path = workspace / lock_name if lock_name == "owner.lock" else workspace.parent / lock_name
    before = hashes(workspace)
    with FileLock(path):
        with pytest.raises(m.MaintenanceError, match="owned"):
            run(workspace)
    assert not (workspace.parent / "new-backup").exists()
    assert hashes(workspace) == before
    # Also proves refusal releases any other acquired handle.
    assert run(workspace)["complete"] is True


@pytest.mark.parametrize(
    "fault", ["symlink", "fifo", "setuid", "journal", "remote-state", "missing-lock"]
)
def test_unsafe_entries_refused_before_output(workspace, fault):
    if fault == "symlink":
        (workspace / "link").symlink_to(workspace / "private-settings.json")
    elif fault == "fifo":
        os.mkfifo(workspace / "pipe")
    elif fault == "setuid":
        os.chmod(workspace / "private-settings.json", 0o4600)
    elif fault == "journal":
        (workspace / "workspace.sqlite3-wal").write_bytes(b"")
    elif fault == "remote-state":
        (workspace / "jobs/job/sky-state.json").write_text("{}")
    else:
        (workspace / "owner.lock").unlink()
    with pytest.raises((m.MaintenanceError, OSError)):
        run(workspace)
    assert not (workspace.parent / "new-backup").exists()


@pytest.mark.parametrize(
    "fault",
    [
        "revision",
        "extra-table",
        "extra-column",
        "index",
        "invalid-json",
        "nonfinite",
        "identity",
        "project",
        "status",
        "active",
        "invalid-kind",
    ],
)
def test_database_and_typed_record_refusals(workspace, fault):
    with sqlite3.connect(workspace / "workspace.sqlite3") as db:
        if fault == "revision":
            db.execute("UPDATE alembic_version SET version_num='9999'")
        elif fault == "extra-table":
            db.execute("CREATE TABLE ignored(value TEXT)")
        elif fault == "extra-column":
            db.execute("ALTER TABLE jobs ADD COLUMN ignored TEXT")
        elif fault == "index":
            db.execute("DROP INDEX ix_jobs_project_id")
            db.execute("CREATE INDEX ix_jobs_project_id ON jobs(status)")
        elif fault == "invalid-json":
            db.execute("UPDATE jobs SET record=?", ('{"id":',))
        elif fault == "nonfinite":
            db.execute("UPDATE jobs SET record=?", ('{"ignored":1e309}',))
        elif fault == "status":
            db.execute("UPDATE jobs SET status='succeeded'")
    if fault == "identity":
        update_job(workspace, lambda value: value.update(id="other"))
    elif fault == "project":
        update_job(workspace, lambda value: value.update(project_id="missing"))
    elif fault == "invalid-kind":
        update_job(workspace, lambda value: value.update(kind="policy.quantize"))
    elif fault == "active":
        update_job(workspace, lambda value: value.update(status="running"))
        with sqlite3.connect(workspace / "workspace.sqlite3") as db:
            db.execute("UPDATE jobs SET status='running'")
    before = hashes(workspace)
    with pytest.raises(m.MaintenanceError):
        run(workspace)
    assert hashes(workspace) == before
    assert not (workspace.parent / "new-backup").exists()


@pytest.mark.parametrize("fault", ["boolean-version", "owner", "unknown-field", "duplicate-key"])
def test_marker_is_checked_and_never_rewritten(workspace, fault):
    path = workspace / "desktop-owner.json"
    data = json.loads(path.read_bytes())
    if fault == "boolean-version":
        data["schema_version"] = True
    elif fault == "owner":
        data["owner"] = "other"
    elif fault == "unknown-field":
        data["ignored"] = 1
    path.write_text(json.dumps(data))
    if fault == "duplicate-key":
        path.write_text('{"schema_version":2,"schema_version":2}')
    before = hashes(workspace)
    with pytest.raises(m.MaintenanceError):
        run(workspace)
    assert hashes(workspace) == before


@pytest.mark.parametrize(
    "limits", [m.Limits(entries=2), m.Limits(bytes=100), m.Limits(json_bytes=10)]
)
def test_budgets_refuse_before_output(workspace, limits):
    with pytest.raises(m.MaintenanceError):
        run(workspace, limits=limits)
    assert not (workspace.parent / "new-backup").exists()


def test_low_disk_refuses_before_output(workspace, monkeypatch):
    from collections import namedtuple

    monkeypatch.setattr(
        m.shutil, "disk_usage", lambda _: namedtuple("disk", "total used free")(100, 99, 1)
    )
    with pytest.raises(m.MaintenanceError, match="disk"):
        run(workspace)
    assert not (workspace.parent / "new-backup").exists()


def test_copy_failure_preserves_partial_without_success_receipt(workspace, monkeypatch):
    before = hashes(workspace)

    def fail(*args):
        raise OSError("controlled copy failure")

    monkeypatch.setattr(m, "copy_files", fail)
    with pytest.raises(OSError):
        run(workspace)
    assert (workspace.parent / "new-backup/workspace").is_dir()
    assert not (workspace.parent / "new-backup/receipt.json").exists()
    assert hashes(workspace) == before


def test_source_mutation_is_not_accepted(workspace, monkeypatch):
    original = m.copy_files

    def changing(*args):
        original(*args)
        (workspace / "private-settings.json").write_text("concurrent change")

    monkeypatch.setattr(m, "copy_files", changing)
    with pytest.raises(m.MaintenanceError, match="changed"):
        run(workspace)
    assert not (workspace.parent / "new-backup/receipt.json").exists()


def test_receipt_flush_failure_cannot_leave_success(workspace, monkeypatch):
    original = m.sync_directory

    def fail(path):
        if path.name == "new-backup":
            raise OSError("controlled publication failure")
        original(path)

    monkeypatch.setattr(m, "sync_directory", fail)
    with pytest.raises(OSError):
        run(workspace)
    assert not (workspace.parent / "new-backup/receipt.json").exists()
    assert (workspace.parent / "new-backup/workspace").is_dir()


def test_original_is_only_opened_as_sqlite_read_only(workspace, monkeypatch):
    actual = m.sqlite3.connect
    observed = []

    def connect(address, **kwargs):
        observed.append(address)
        assert address.endswith("?mode=ro&immutable=1") and kwargs["uri"] is True
        return actual(address, **kwargs)

    monkeypatch.setattr(m.sqlite3, "connect", connect)
    run(workspace)
    assert len(observed) == 2
    assert observed[0].startswith(workspace.as_uri())


def test_actual_current_alembic_schema_accepted_without_api(workspace):
    import asyncio

    from vla_platform.storage import Storage

    (workspace / "workspace.sqlite3").unlink()

    async def initialize():
        storage = Storage(workspace)
        try:
            await storage.initialize()
        finally:
            await storage.close()

    asyncio.run(initialize())
    result = run(workspace)
    assert result["database"]["alembic_revision"] == "0001"
    assert result["database"]["projects"] == result["database"]["jobs"] == 0


def artifact(workspace):
    root = workspace / "jobs/job/operation/artifact"
    root.mkdir(parents=True)
    (root / "policy.bin").write_bytes(b"Generated inventory bytes, not a model")
    manifest = {
        "files": {"policy.bin": hashlib.sha256((root / "policy.bin").read_bytes()).hexdigest()}
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    record = {
        "id": "job:operation",
        "project_id": "project",
        "job_id": "job",
        "label": "Generated artifact",
        "format": "native_checkpoint",
        "path": root.relative_to(workspace).as_posix(),
        "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        "file_bytes": (root / "policy.bin").stat().st_size,
        "parent_ids": [],
        "metadata": {},
    }
    update_job(
        workspace,
        lambda value: value.update(
            kind="policy.quantize",
            request={"operation": "policy.quantize", "runtime_id": "fixture", "artifact_id": "old"},
            result={"artifacts": [record], "reports": [], "decision": "completed"},
        ),
    )
    return root


def test_complete_registered_artifact_inventory_is_preserved(workspace):
    root = artifact(workspace)
    assert run(workspace)["complete"] is True
    assert (
        workspace.parent / "new-backup/workspace" / root.relative_to(workspace) / "policy.bin"
    ).read_bytes() == (root / "policy.bin").read_bytes()


@pytest.mark.parametrize(
    "fault", ["missing", "changed", "extra", "manifest", "escape", "ownership", "size"]
)
def test_registered_artifact_failures_preserve_original(workspace, fault):
    root = artifact(workspace)
    if fault == "missing":
        (root / "policy.bin").unlink()
    elif fault == "changed":
        (root / "policy.bin").write_bytes(b"changed")
    elif fault == "extra":
        (root / "extra.bin").write_bytes(b"extra")
    elif fault == "manifest":
        (root / "manifest.json").write_text("{}")
    else:

        def edit(value):
            field = {"escape": "path", "ownership": "project_id", "size": "file_bytes"}[fault]
            value["result"]["artifacts"][0][field] = {
                "escape": "../outside",
                "ownership": "other",
                "size": 0,
            }[fault]

        update_job(workspace, edit)
    before = hashes(workspace)
    with pytest.raises(m.MaintenanceError):
        run(workspace)
    assert before == hashes(workspace)
    assert not (workspace.parent / "new-backup").exists()


def test_publication_failure_preserves_unrelated_replacement(workspace, monkeypatch):
    original = m.sync_directory

    def fail(path):
        if path.name == "new-backup":
            receipt = path / "receipt.json"
            receipt.unlink()
            receipt.write_text("unrelated replacement")
            raise KeyboardInterrupt
        original(path)

    monkeypatch.setattr(m, "sync_directory", fail)
    with pytest.raises(KeyboardInterrupt):
        run(workspace)
    assert (workspace.parent / "new-backup/receipt.json").read_text() == "unrelated replacement"


def test_sqlite_handles_are_closed_after_preflight_and_copy_validation(workspace, monkeypatch):
    actual = m.sqlite3.connect
    handles = []

    def connect(address, **kwargs):
        handle = actual(address, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(m.sqlite3, "connect", connect)
    run(workspace)
    assert len(handles) == 2
    for handle in handles:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            handle.execute("SELECT 1")


def test_native_snapshot_not_misrepresented_as_semantically_verified(workspace):
    digest = "a" * 64
    result = {
        "source": "local",
        "repo_id": None,
        "revision": "metadata-sha256:" + digest,
        "format": "lerobot_v3",
        "robot_type": None,
        "total_episodes": 1,
        "total_frames": 1,
        "fps": 1.0,
        "features": {"fixture": {"dtype": "float32"}},
        "license": None,
        "metadata_sha256": digest,
        "inspected_at": "2026-09-27T00:00:00+00:00",
        "warnings": [],
        "inspection_scope": "complete_snapshot",
        "snapshot": {
            "id": "sha256:" + digest,
            "manifest_sha256": digest,
            "total_bytes": 1,
            "file_count": 1,
            "total_episodes": 1,
            "total_frames": 1,
            "lineage_validated": False,
            "warnings": [],
        },
    }
    update_job(workspace, lambda value: value.update(result=result))
    with pytest.raises(m.MaintenanceError, match="snapshots require separate"):
        run(workspace)
    assert not (workspace.parent / "new-backup").exists()


def test_remote_target_refused_without_executing_recovery(workspace):
    target = {
        "project_id": "fixture-project",
        "workspace": "fixture",
        "sky_api_endpoint": "http://127.0.0.1:9999",
        "region": "us-east4",
        "accelerator": "L4",
        "disk_size_gb": 100,
        "idle_minutes": 1,
    }
    update_job(workspace, lambda value: value.update(compute_target=target))
    with pytest.raises(m.MaintenanceError, match="Remote target"):
        run(workspace)
    assert not (workspace.parent / "new-backup").exists()


def test_cooperative_copy_deadline_prevents_receipt(workspace, monkeypatch):
    original = m.copy_files

    def over_budget(*args):
        original(*args)
        args[-1].end = 0

    monkeypatch.setattr(m, "copy_files", over_budget)
    before = hashes(workspace)
    with pytest.raises(m.MaintenanceError, match="deadline"):
        run(workspace)
    assert before == hashes(workspace)
    assert not (workspace.parent / "new-backup/receipt.json").exists()


def test_marker_change_between_read_and_inventory_cannot_mix_receipt_identity(
    workspace, monkeypatch
):
    original = m.marker

    def changing(*args):
        value = original(*args)
        changed = {**value, "build_id": "d" * 40}
        (workspace / "desktop-owner.json").write_text(json.dumps(changed))
        return value

    monkeypatch.setattr(m, "marker", changing)
    with pytest.raises(m.MaintenanceError, match="changed"):
        run(workspace)
    assert not (workspace.parent / "new-backup/receipt.json").exists()
