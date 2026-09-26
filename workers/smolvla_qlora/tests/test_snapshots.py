import hashlib
import json

import pytest

from firebird_vla import checkpoint, snapshots


def native(root, step):
    path = root / f"checkpoint-{step:06d}"
    path.mkdir(parents=True)
    (path / "weights.bin").write_bytes(f"weights at {step}".encode())
    checkpoint.write_json(
        path / "manifest.json",
        {
            "schema_version": 1,
            "step": step,
            "files": {"weights.bin": checkpoint.sha256(path / "weights.bin")},
        },
    )
    return path


def test_snapshots_are_immutable_atomic_and_indexed(tmp_path, monkeypatch):
    logs, training = tmp_path / "logs", tmp_path / "training"
    logs.mkdir()
    source = native(training, 5)
    monkeypatch.setattr(snapshots, "PART_BYTES", 4096)
    snapshots.publish_checkpoint(source, logs)
    published = logs / "checkpoint-snapshots" / source.name
    descriptor = json.loads((published / "firebird-output.json").read_text())
    index = json.loads((logs / "checkpoint-index" / "index.json").read_text())
    assert index["checkpoints"][0]["name"] == "checkpoint-000005"
    assert index["checkpoints"][0]["step"] == 5
    content = b"".join((published / part["name"]).read_bytes() for part in descriptor["parts"])
    assert hashlib.sha256(content).hexdigest() == descriptor["sha256"]
    assert all(part["size"] <= 4096 for part in descriptor["parts"])
    assert not list(training.glob(".snapshot-*"))
    with pytest.raises(FileExistsError):
        snapshots.publish_checkpoint(source, logs)


def test_corrupt_or_misnamed_checkpoint_never_enters_index(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    source = native(tmp_path / "training", 5)
    (source / "weights.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        snapshots.publish_checkpoint(source, logs)
    assert not (logs / "checkpoint-index" / "index.json").exists()
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["files"]["weights.bin"] = checkpoint.sha256(source / "weights.bin")
    manifest["step"] = 10
    checkpoint.write_json(source / "manifest.json", manifest)
    with pytest.raises(ValueError, match="optimizer step"):
        snapshots.publish_checkpoint(source, logs)


def test_only_acknowledged_older_remote_copies_are_pruned(tmp_path):
    logs, training = tmp_path / "cloud-logs", tmp_path / "cloud-training"
    logs.mkdir()
    sources = [native(training, step) for step in (5, 10, 15, 20)]
    for source in sources:
        snapshots.publish_checkpoint(source, logs)
    receipts = {source.name: checkpoint.sha256(source / "manifest.json") for source in sources}
    local_backup = tmp_path / "local-backup"
    local_backup.mkdir()
    (local_backup / "checkpoint-000005").write_text("host copy")
    snapshots.acknowledge(logs, training, {sources[0].name: receipts[sources[0].name]})
    assert not sources[0].exists()
    assert sources[1].exists()  # Unacknowledged, despite being older than newest two.
    assert sources[2].exists() and sources[3].exists()
    assert (local_backup / "checkpoint-000005").read_text() == "host copy"
    snapshots.acknowledge(logs, training, receipts)
    assert not sources[1].exists()
    assert sources[2].exists() and sources[3].exists()
    snapshots.acknowledge(logs, training, receipts)  # Lost acknowledgement responses are retryable.
    index = json.loads((logs / "checkpoint-index" / "index.json").read_text())
    assert [item["step"] for item in index["checkpoints"] if item.get("remote_pruned")] == [5, 10]


@pytest.mark.parametrize("receipt", [{"../other": "0" * 64}, {"checkpoint-000005": "0" * 64}])
def test_unverified_receipts_cannot_delete_checkpoints(tmp_path, receipt):
    logs, training = tmp_path / "logs", tmp_path / "training"
    logs.mkdir()
    for step in (5, 10, 15):
        snapshots.publish_checkpoint(native(training, step), logs)
    with pytest.raises(ValueError, match="Unverified"):
        snapshots.acknowledge(logs, training, receipt)
    assert len(list(training.glob("checkpoint-*"))) == 3


def test_latest_json_replace_failure_preserves_previous_value(tmp_path, monkeypatch):
    path = tmp_path / "latest.json"
    checkpoint.write_json(path, {"step": 5})

    def fail(*args):
        raise OSError("simulated interruption before commit")

    monkeypatch.setattr(checkpoint.os, "replace", fail)
    with pytest.raises(OSError):
        checkpoint.write_json(path, {"step": 10})
    assert json.loads(path.read_text()) == {"step": 5}
    assert list(tmp_path.iterdir()) == [path]


def test_low_cloud_disk_fails_without_publishing_partial_checkpoint(tmp_path, monkeypatch):
    logs = tmp_path / "logs"
    logs.mkdir()
    source = native(tmp_path / "training", 5)
    monkeypatch.setattr(snapshots.shutil, "disk_usage", lambda _: type("Disk", (), {"free": 0})())
    with pytest.raises(ValueError, match="cloud disk"):
        snapshots.publish_checkpoint(source, logs)
    assert source.is_dir()
    assert not (logs / "checkpoint-index" / "index.json").exists()
