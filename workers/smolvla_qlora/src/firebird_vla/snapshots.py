"""Immutable checkpoint transfers; only committed descriptors enter the live index."""

import base64
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .checkpoint import verify_bundle, write_json

PART_BYTES = 32 * 1024**2
MAX_BYTES = 20 * 1024**3
MAX_FILES = 10000


@contextmanager
def index_lock(index_dir):
    with (index_dir / ".lock").open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def publish_checkpoint(checkpoint: Path, log_root: Path):
    if not re.fullmatch(r"checkpoint-\d{6,}", checkpoint.name):
        raise ValueError("Invalid checkpoint snapshot name")
    manifest = verify_bundle(checkpoint)
    if os.getenv("FIREBIRD_GCS_PREFIX"):
        from cloud_storage import publish_training_checkpoint
        publish_training_checkpoint(checkpoint)
        return
    if (
        type(manifest.get("step")) is not int
        or manifest["step"] < 1
        or manifest["step"] != int(checkpoint.name[11:])
    ):
        raise ValueError("Checkpoint snapshot requires an optimizer step")
    snapshots = log_root / "checkpoint-snapshots"
    snapshots.mkdir(exist_ok=True)
    destination = snapshots / checkpoint.name
    if destination.exists():
        raise FileExistsError("Checkpoint snapshots are immutable")
    files = [path for path in sorted(checkpoint.rglob("*")) if path.is_file()]
    size = sum(path.stat().st_size for path in files)
    if size > MAX_BYTES or len(files) > MAX_FILES:
        raise ValueError("Checkpoint snapshot exceeds its transfer limit")
    index_dir = log_root / "checkpoint-index"
    index_dir.mkdir(exist_ok=True)
    index_path = index_dir / "index.json"
    if shutil.disk_usage(checkpoint.parent).free < size * 2 + 1024**3:
        raise ValueError("Not enough cloud disk space to publish the next checkpoint safely")
    staging = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=checkpoint.parent))
    try:
        archive_path = staging / "checkpoint.tar"
        with tarfile.open(archive_path, "w") as archive:
            for path in files:
                archive.add(path, arcname=path.relative_to(checkpoint).as_posix(), recursive=False)
        parts, digest = [], hashlib.sha256()
        with archive_path.open("rb") as stream:
            while chunk := stream.read(PART_BYTES):
                name = f"firebird-output.part-{len(parts):05d}"
                (staging / name).write_bytes(chunk)
                parts.append({"name": name, "size": len(chunk)})
                digest.update(chunk)
        descriptor = {
            "checkpoint": checkpoint.name,
            "step": manifest["step"],
            "size": archive_path.stat().st_size,
            "files": len(files),
            "parts": parts,
            "sha256": digest.hexdigest(),
            "manifest_sha256": hashlib.sha256(
                (checkpoint / "manifest.json").read_bytes()
            ).hexdigest(),
        }
        write_json(staging / "firebird-output.json", descriptor)
        archive_path.unlink()
        # Snapshot data is visible before its small index entry. A sync racing
        # the rename can miss this snapshot once but can never publish a partial one.
        with index_lock(index_dir):
            index = (
                json.loads(index_path.read_text()) if index_path.exists() else {"checkpoints": []}
            )
            entries = index["checkpoints"]
            if not isinstance(entries, list) or len(entries) >= 10000:
                raise ValueError("Invalid or oversized checkpoint transfer index")
            os.rename(staging, destination)
            entries.append(
                {
                    "name": checkpoint.name,
                    "step": manifest["step"],
                    "file_bytes": size,
                    "manifest_sha256": descriptor["manifest_sha256"],
                }
            )
            write_json(index_path, {"checkpoints": entries})
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def acknowledge(log_root: Path, training_root: Path, receipts: dict):
    """Drop older remote copies only after the host verified their exact manifest."""
    if not isinstance(receipts, dict) or len(receipts) > 256:
        raise ValueError("Invalid checkpoint acknowledgements")
    index_dir = log_root / "checkpoint-index"
    with index_lock(index_dir):
        index = json.loads((index_dir / "index.json").read_text())
        entries = {item["name"]: item for item in index["checkpoints"]}
        for name, digest in receipts.items():
            if (
                not re.fullmatch(r"checkpoint-\d{6,}", name)
                or name not in entries
                or entries[name]["manifest_sha256"] != digest
            ):
                raise ValueError("Unverified checkpoint acknowledgement")
        for name in receipts:
            entries[name]["acknowledged"] = True
        retained = set(sorted(entries, key=lambda name: entries[name]["step"])[-2:])
        for name, entry in entries.items():
            if name in retained or not entry.get("acknowledged") or entry.get("remote_pruned"):
                continue
            native = training_root / name
            transfer = log_root / "checkpoint-snapshots" / name
            if (
                native.is_symlink()
                or transfer.is_symlink()
                or not native.resolve().is_relative_to(training_root.resolve())
                or not transfer.resolve().is_relative_to(
                    (log_root / "checkpoint-snapshots").resolve()
                )
            ):
                raise ValueError("Unsafe checkpoint retention path")
            if native.exists():
                if (
                    hashlib.sha256((native / "manifest.json").read_bytes()).hexdigest()
                    != entry["manifest_sha256"]
                ):
                    raise ValueError("Remote checkpoint manifest changed after acknowledgement")
                shutil.rmtree(native)
            if transfer.exists():
                shutil.rmtree(transfer)
            entry["remote_pruned"] = True
        write_json(index_dir / "index.json", {"checkpoints": list(entries.values())})


def main():
    if len(sys.argv) != 2 or len(sys.argv[1]) > 65536:
        raise ValueError("Invalid checkpoint acknowledgement command")
    receipts = json.loads(base64.urlsafe_b64decode(sys.argv[1]).decode())
    acknowledge(
        Path.home() / "sky_logs" / "1-firebird-training",
        Path.cwd() / "output" / "training",
        receipts,
    )


if __name__ == "__main__":
    main()
