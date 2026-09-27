"""Generated source-mode proof only; no API, runtime, provider, native build or GUI."""

import builtins
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

SIDE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIDE))
try:
    proof = importlib.import_module("workspace_roundtrip")
finally:
    sys.path.pop(0)

m = proof.maintenance


def test_actual_migration_then_read_only_roundtrip_preserves_all_records_and_modes(
    tmp_path, monkeypatch
):
    from vla_platform.storage import Storage

    original_initialize = Storage.initialize
    starts = []

    async def initialize(storage):
        starts.append(str(storage.engine.url.database))
        assert len(starts) == 1, "Migration cannot run again on sealed data"
        await original_initialize(storage)

    monkeypatch.setattr(Storage, "initialize", initialize)
    modules = set(sys.modules)
    output = tmp_path / "proof"
    result = proof.run(output)
    assert starts == [str(output / "original/workspace.sqlite3")]
    assert result["complete"] and result["fresh_roundtrip_exact"]
    assert result["original_unchanged"] and result["backup_unchanged"]
    for key in (
        "upgrade_performed",
        "activation_performed",
        "api_or_execution_started",
        "native_launch_verified",
        "frozen_payload_verified",
    ):
        assert result[key] is False
    assert result["record_identity"]["database"] == {
        "alembic_revision": "0001",
        "projects": 1,
        "jobs": 2,
        "all_jobs_terminal": True,
        "validation": "stored_schema_and_local_artifact_inventory",
    }
    for name in ("original", "backup/workspace", "candidate", "roundtrip"):
        root = output / name
        intake = result["record_identity"]["jobs"][0]["record"]
        info = root / intake["request"]["path"] / "meta/info.json"
        assert proof.sha256(info.read_bytes()) == intake["result"]["metadata_sha256"]
        assert m.inventory(root, m.Budget(m.Limits())) == result["files"]
        assert (root / "empty-directory").stat().st_mode & 0o777 == 0o750
        assert (root / "private-settings-sentinel.json").stat().st_mode & 0o777 == 0o600
        assert (
            proof.record_identity(root, result["files"], m.Budget(m.Limits()))
            == result["record_identity"]
        )
    assert not set(proof.FORBIDDEN_IMPORTS) & (set(sys.modules) - modules)
    assert m.read_json(output / "receipt.json", m.Limits().json_bytes) == result


@pytest.mark.parametrize("kind", ["directory", "file", "symlink"])
def test_output_must_be_new_and_existing_bytes_survive(tmp_path, kind):
    output = tmp_path / "existing"
    target = tmp_path / "untouched"
    target.write_bytes(b"preserve")
    if kind == "directory":
        output.mkdir()
    elif kind == "file":
        output.write_bytes(b"existing")
    else:
        output.symlink_to(target)
    with pytest.raises(FileExistsError):
        proof.run(output)
    assert target.read_bytes() == b"preserve"
    assert output.exists()


@pytest.mark.parametrize("name", proof.FORBIDDEN_IMPORTS)
def test_application_imports_refused_even_when_cached(name, monkeypatch):
    monkeypatch.setitem(sys.modules, name, object())
    previous = builtins.__import__
    with proof.no_application_imports():
        with pytest.raises(m.MaintenanceError, match="startup"):
            __import__(name)
    assert builtins.__import__ is previous


@pytest.mark.parametrize(
    "event",
    [
        "socket.connect",
        "socket.bind",
        "socket.getaddrinfo",
        "urllib.Request",
        "subprocess.Popen",
        "os.system",
        "os.exec",
        "os.posix_spawn",
        "os.fork",
    ],
)
def test_cli_external_action_guard(event):
    with pytest.raises(m.MaintenanceError, match="forbidden"):
        proof.deny_external_actions(event, ())
    proof.deny_external_actions("open", ())


def test_sealed_original_mutation_cannot_publish_success(tmp_path, monkeypatch):
    actual = proof.fresh_copy
    output = tmp_path / "proof"

    def change(*args, **kwargs):
        result = actual(*args, **kwargs)
        (output / "original/private-settings-sentinel.json").write_text("fault injection")
        return result

    monkeypatch.setattr(proof, "fresh_copy", change)
    with pytest.raises(m.MaintenanceError, match="inventory changed"):
        proof.run(output)
    assert not (output / "receipt.json").exists()
    backup = m.read_json(output / "sealed-backup.json", m.Limits().json_bytes)
    assert m.inventory(output / "backup", m.Budget(m.Limits())) == backup


def test_published_backup_receipt_is_bound_before_copy(tmp_path, monkeypatch):
    actual = m.prepare_backup

    def change(workspace, destination):
        result = actual(workspace, destination)
        (destination / "receipt.json").write_text("{}")
        return result

    monkeypatch.setattr(m, "prepare_backup", change)
    output = tmp_path / "proof"
    with pytest.raises(m.MaintenanceError, match="receipt differs"):
        proof.run(output)
    assert not (output / "candidate").exists()
    assert not (output / "receipt.json").exists()


def test_partial_candidate_failure_preserves_original_and_backup(tmp_path, monkeypatch):
    actual = m.copy_files

    def fail(source, destination, files, budget):
        if destination.name == "candidate":
            (destination / "partial").write_bytes(b"generated incomplete copy")
            raise OSError("generated copy failure")
        return actual(source, destination, files, budget)

    monkeypatch.setattr(m, "copy_files", fail)
    output = tmp_path / "proof"
    with pytest.raises(OSError, match="copy failure"):
        proof.run(output)
    sealed = m.read_json(output / "sealed-original.json", m.Limits().json_bytes)
    assert m.inventory(output / "original", m.Budget(m.Limits())) == sealed["files"]
    assert (output / "candidate/partial").is_file()
    assert (output / "backup/receipt.json").is_file()
    assert not (output / "receipt.json").exists()


def test_candidate_record_mutation_is_rejected(tmp_path, monkeypatch):
    actual = m.copy_files

    def change(source, destination, files, budget):
        actual(source, destination, files, budget)
        if destination.name == "candidate":
            with sqlite3.connect(destination / "workspace.sqlite3") as db:
                db.execute("UPDATE jobs SET status='running'")

    monkeypatch.setattr(m, "copy_files", change)
    output = tmp_path / "proof"
    with pytest.raises(m.MaintenanceError, match="inventory changed"):
        proof.run(output)
    assert not (output / "receipt.json").exists()


@pytest.mark.parametrize("replace", [False, True])
def test_publication_failure_removes_only_owned_receipt(tmp_path, monkeypatch, replace):
    path = tmp_path / "receipt.json"

    def fail(parent):
        if replace:
            path.unlink()
            path.write_bytes(b"unrelated replacement")
        raise OSError("generated directory fsync failure")

    monkeypatch.setattr(m, "sync_directory", fail)
    with pytest.raises(OSError, match="fsync failure"):
        proof.write_new(path, {"complete": True})
    if replace:
        assert path.read_bytes() == b"unrelated replacement"
    else:
        assert not path.exists()
