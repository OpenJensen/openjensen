"""Offline disposable transactions; payload bytes are generated, never executed."""

import importlib
import json
import os
import sys
from pathlib import Path

import pytest
from filelock import FileLock
from test_prepare_local_tauri import fixture as payload_fixture
from test_workspace_maintenance import workspace  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    t = importlib.import_module("workspace_transaction")
finally:
    sys.path.pop(0)
m = t.m


@pytest.fixture
def setup(workspace, tmp_path):  # noqa: F811
    root = workspace.with_name("desktop-v1")
    workspace.rename(root)
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    old = t.PayloadFiles(*payload_fixture(old_dir))
    new = t.PayloadFiles(*payload_fixture(new_dir))
    executable = new.root / "firebird-sidecar"
    executable.write_bytes(executable.read_bytes() + b"different generated payload")
    from payload_inventory import inventory

    manifest = inventory(new.root)
    new.manifest.write_text(json.dumps(manifest))
    acceptance = json.loads(new.acceptance.read_text())
    acceptance["payload_identity_sha256"] = manifest["identity_sha256"]
    new.acceptance.write_text(json.dumps(acceptance))
    marker = t.accepted_payload(old, m.Budget(m.Limits()))["marker"]
    (root / "desktop-owner.json").write_bytes(json.dumps(marker, indent=3).encode())
    (root / "desktop-owner.json").chmod(0o640)
    return root, tmp_path / "backup", old, new


def prepare(setup):
    return t.prepare_handoff(*setup)


def snapshot(root):
    return m.inventory(root, m.Budget(m.Limits()))


def test_complete_handoff_changes_only_marker_and_exact_unstarted_reversal(setup):
    root, backup, _, new = setup
    before = snapshot(root)
    prepared = prepare(setup)
    assert snapshot(root) == before == snapshot(backup / "workspace")
    backup_before = snapshot(backup)
    committed = t.commit_handoff(root)
    assert (
        m.marker(root, m.Budget(m.Limits()))
        == t.accepted_payload(new, m.Budget(m.Limits()))["marker"]
    )
    after = snapshot(root)
    assert [name for name in before if before[name] != after[name]] == ["desktop-owner.json"]
    assert after["desktop-owner.json"]["mode"] == 0o640
    assert committed["inventory_sha256"] == t.digest_inventory(after)
    assert snapshot(backup) == backup_before
    result = t.reverse_unstarted_handoff(root)
    assert snapshot(root) == before
    assert result["marker"] == prepared["old_payload"]["marker"]
    assert snapshot(backup) == backup_before
    with pytest.raises(m.MaintenanceError, match="incomplete"):
        t.commit_handoff(root)
    with pytest.raises(m.MaintenanceError, match="incomplete"):
        t.reverse_unstarted_handoff(root)


@pytest.mark.parametrize("lock_name", ["owner.lock", ".desktop-maintenance.lock"])
def test_actual_flock_excludes_preparation(setup, lock_name):
    root = setup[0]
    target = root / lock_name if lock_name == "owner.lock" else root.parent / lock_name
    with FileLock(target):
        with pytest.raises(m.MaintenanceError, match="owned"):
            prepare(setup)
    assert not setup[1].exists()
    assert not (root.parent / t.JOURNAL).exists()


@pytest.mark.parametrize("phase", ["commit", "reverse"])
@pytest.mark.parametrize(
    "fault", ["data", "backup", "payload", "started", "extra", "intent", "record"]
)
def test_changes_and_uncertain_states_refuse_without_marker_change(setup, phase, fault):
    root, backup, _, new = setup
    prepare(setup)
    journal = root.parent / t.JOURNAL
    if phase == "reverse":
        t.commit_handoff(root)
    if fault == "data":
        (root / "private-settings.json").write_text("changed")
    elif fault == "backup":
        (backup / "workspace/private-settings.json").write_text("changed")
    elif fault == "payload":
        (new.root / "firebird-sidecar").write_bytes(b"changed")
    elif fault == "started":
        (journal / "start-intent.json").write_text("{}")
    elif fault == "extra":
        (journal / "unknown").write_text("{}")
    elif fault == "intent":
        (
            journal / ("reverse-intent.json" if phase == "reverse" else "commit-intent.json")
        ).write_text("{}")
    else:
        value = json.loads((journal / "prepared.json").read_text())
        value["workspace_path_sha256"] = "0" * 64
        (journal / "prepared.json").write_text(json.dumps(value))
    before = (root / "desktop-owner.json").read_bytes()
    with pytest.raises((m.MaintenanceError, ValueError)):
        (t.commit_handoff if phase == "commit" else t.reverse_unstarted_handoff)(root)
    assert (root / "desktop-owner.json").read_bytes() == before


@pytest.mark.parametrize("phase", ["commit", "reverse"])
@pytest.mark.parametrize(
    "barrier",
    [
        "intent",
        "temporary_flush",
        "replace_before",
        "replace_after",
        "directory_flush",
        "completion",
    ],
)
def test_publication_failures_remain_unresolved_and_do_not_retry(
    setup, monkeypatch, phase, barrier
):
    root = setup[0]
    prepare(setup)
    if phase == "reverse":
        t.commit_handoff(root)
    journal = root.parent / t.JOURNAL
    actual_publish, actual_fsync, actual_replace, actual_sync = (
        t.publish,
        os.fsync,
        os.replace,
        m.sync_directory,
    )
    intent = "commit-intent.json" if phase == "commit" else "reverse-intent.json"
    completion = "committed.json" if phase == "commit" else "reversed.json"

    def publish(path, value):
        if (barrier == "intent" and path.name == intent) or (
            barrier == "completion" and path.name == completion
        ):
            raise OSError("publication fault")
        return actual_publish(path, value)

    def fsync(fd):
        if barrier == "temporary_flush" and (journal / "marker.pending").exists():
            raise OSError("temporary flush fault")
        return actual_fsync(fd)

    def replace(src, dst):
        if barrier == "replace_before":
            raise OSError("replacement fault")
        result = actual_replace(src, dst)
        if barrier == "replace_after":
            raise KeyboardInterrupt("interrupted after replacement")
        return result

    def sync(path):
        if barrier == "directory_flush" and path == root:
            raise OSError("directory flush fault")
        return actual_sync(path)

    with monkeypatch.context() as patch:
        patch.setattr(t, "publish", publish)
        patch.setattr(os, "fsync", fsync)
        patch.setattr(os, "replace", replace)
        patch.setattr(m, "sync_directory", sync)
        with pytest.raises((OSError, KeyboardInterrupt)):
            (t.commit_handoff if phase == "commit" else t.reverse_unstarted_handoff)(root)
    assert not (journal / completion).exists()
    if barrier != "intent":
        with pytest.raises(m.MaintenanceError, match="incomplete"):
            (t.commit_handoff if phase == "commit" else t.reverse_unstarted_handoff)(root)
    # A failure before any durable intent/marker change is still only prepared, never startup-ready.
    assert (journal / "prepared.json").exists()
    assert (
        snapshot(setup[1] / "workspace")
        == json.loads((journal / "prepared.json").read_text())["files"]
    )


@pytest.mark.parametrize("replace", [False, True])
def test_receipt_flush_failure_removes_only_owned_inode(tmp_path, monkeypatch, replace):
    target = tmp_path / "receipt.json"
    actual = os.fsync

    def fail(fd):
        if replace:
            target.unlink()
            target.write_text("unrelated replacement")
        raise OSError("fsync failed")

    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(OSError):
        t.publish(target, {"test": True})
    monkeypatch.setattr(os, "fsync", actual)
    if replace:
        assert target.read_text() == "unrelated replacement"
    else:
        assert not target.exists()


def test_partial_preparation_blocks_new_prepare_and_preserves_backup(setup, monkeypatch):
    def fail(*_):
        raise OSError("prepared receipt flush")

    monkeypatch.setattr(t, "publish", fail)
    before = snapshot(setup[0])
    with pytest.raises(OSError):
        prepare(setup)
    assert snapshot(setup[0]) == before == snapshot(setup[1] / "workspace")
    assert (setup[0].parent / t.JOURNAL).is_dir()
    with pytest.raises(m.MaintenanceError, match="existing transaction"):
        prepare(setup)


def test_bad_acceptance_and_payload_cannot_create_backup(setup):
    report = json.loads(setup[3].acceptance.read_text())
    report[t.PROOFS[0]] = 1
    setup[3].acceptance.write_text(json.dumps(report))
    with pytest.raises(m.MaintenanceError, match="acceptance"):
        prepare(setup)
    assert not setup[1].exists()


def test_low_disk_refuses_before_backup_or_journal(setup, monkeypatch):
    from collections import namedtuple

    monkeypatch.setattr(m.shutil, "disk_usage", lambda _: namedtuple("Usage", "free")(0))
    with pytest.raises(m.MaintenanceError, match="disk space"):
        prepare(setup)
    assert not setup[1].exists()
    assert not (setup[0].parent / t.JOURNAL).exists()


@pytest.mark.parametrize("fault", ["transaction_id", "extra", "limits", "mode-bool", "mode-float"])
def test_rehashed_prepared_type_changes_refuse_before_marker_mutation(setup, fault):
    root = setup[0]
    prepare(setup)
    path = root.parent / t.JOURNAL / "prepared.json"
    value = json.loads(path.read_text())
    if fault == "transaction_id":
        value["transaction_id"] = "arbitrary"
    elif fault == "extra":
        value["extra"] = 1
    elif fault == "limits":
        value["limits"]["seconds"] = True
    elif fault == "mode-bool":
        value["files"]["desktop-owner.json"]["mode"] = True
    else:
        value["files"]["desktop-owner.json"]["mode"] = float(
            value["files"]["desktop-owner.json"]["mode"]
        )
    path.write_text(json.dumps(value))
    before = snapshot(root)
    with pytest.raises(m.MaintenanceError):
        t.commit_handoff(root)
    assert snapshot(root) == before
    assert not path.with_name("commit-intent.json").exists()


def test_payload_change_during_marker_publication_prevents_completion(setup, monkeypatch):
    root = setup[0]
    prepare(setup)
    actual = t.replace_marker

    def change(*args):
        actual(*args)
        (setup[3].root / "firebird-sidecar").write_bytes(b"changed after validation")

    monkeypatch.setattr(t, "replace_marker", change)
    with pytest.raises(ValueError):
        t.commit_handoff(root)
    journal = root.parent / t.JOURNAL
    assert (journal / "commit-intent.json").exists()
    assert not (journal / "committed.json").exists()


def test_unknown_journal_entry_refuses_without_reading_unbounded_directory(tmp_path, monkeypatch):
    class Entries:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def __iter__(self):
            yield type("Entry", (), {"name": "unexpected"})()
            raise AssertionError("journal reader continued beyond its fixed allowlist")

    monkeypatch.setattr(os, "scandir", lambda _: Entries())
    with pytest.raises(m.MaintenanceError, match="unrecognized"):
        t.journal_names(tmp_path, ["prepared.json"])
