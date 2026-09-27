"""Generated, offline same-format proof; never an upgrade or application entrypoint.

Only the newly created fixture may use Storage.initialize(), before its baseline is
sealed. The original, verified backup and two fresh copies are subsequently read-only.
No existing workspace path is accepted. Partial output is retained on every failure.
"""

import argparse
import asyncio
import builtins
import hashlib
import json
import os
import sqlite3
import sys
from contextlib import closing, contextmanager
from pathlib import Path

import workspace_maintenance as maintenance

STAMP = "2026-09-27T00:00:00+00:00"
PROJECT = "generated-desktop-roundtrip"
FORBIDDEN_IMPORTS = (
    "vla_platform.api",
    "vla_platform.execution",
    "vla_platform.lifecycle.service",
)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_new(path: Path, value: object) -> None:
    """Publish a fresh generated evidence file without replacing any path."""
    owned = None
    stream = None
    try:
        stream = path.open("xb")
        opened = os.fstat(stream.fileno())
        owned = (opened.st_dev, opened.st_ino)
        stream.write(maintenance.canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
        maintenance.sync_directory(path.parent)
    except BaseException:
        if owned is not None:
            try:
                current = path.lstat()
                if (current.st_dev, current.st_ino) == owned:
                    path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        if stream is not None:
            stream.close()


@contextmanager
def no_application_imports():
    """Reject both cached and new imports of API or execution controllers."""
    previous = builtins.__import__

    def guarded(name, *args, **kwargs):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_IMPORTS):
            raise maintenance.MaintenanceError("Application startup is forbidden in this proof")
        return previous(name, *args, **kwargs)

    builtins.__import__ = guarded
    try:
        yield
    finally:
        builtins.__import__ = previous


def deny_external_actions(event: str, args: tuple) -> None:
    """CLI-only audit hook: no network requests or child execution in this proof."""
    if event in {
        "socket.connect",
        "socket.bind",
        "socket.getaddrinfo",
        "urllib.Request",
        "subprocess.Popen",
        "os.system",
        "os.exec",
        "os.posix_spawn",
        "os.fork",
    }:
        raise maintenance.MaintenanceError("Network and process execution are forbidden")


async def seed(workspace: Path) -> None:
    """Create the real current migration and generated records before sealing."""
    from vla_platform.contracts import DatasetProfile, IntakeRequest, Job, Project
    from vla_platform.lifecycle.contracts import LifecycleResult, PolicyArtifact, PolicyRequest
    from vla_platform.storage import Storage

    workspace.mkdir(mode=0o700)
    storage = Storage(workspace)
    try:
        await storage.initialize()
    finally:
        await storage.close()
    (workspace / "owner.lock").touch(mode=0o600)
    write_new(
        workspace / "desktop-owner.json",
        {
            "schema_version": 2,
            "owner": "openjensen-desktop",
            # Generated marker, not an assertion that a real payload was executed.
            "build_id": "a" * 40,
            "resources_sha256": "b" * 64,
            "payload_identity_sha256": "c" * 64,
        },
    )
    metadata = {
        "codebase_version": "v3.0",
        "robot_type": "generated_fixture",
        "total_episodes": 1,
        "total_frames": 2,
        "fps": 10,
        "features": {"observation.state": {"dtype": "float32", "shape": [6]}},
    }
    metadata_path = workspace / "generated-dataset/meta/info.json"
    metadata_path.parent.mkdir(parents=True, mode=0o700)
    write_new(metadata_path, metadata)
    metadata_hash = sha256(metadata_path.read_bytes())
    profile = DatasetProfile(
        source="local",
        revision="metadata-sha256:" + metadata_hash,
        format="lerobot_v3",
        robot_type="generated_fixture",
        total_episodes=1,
        total_frames=2,
        fps=10,
        features={"observation.state": {"dtype": "float32", "shape": [6]}},
        metadata_sha256=metadata_hash,
        inspected_at=STAMP,
        warnings=["Generated metadata-only fixture; no recorded frames or model evidence."],
    )
    package = workspace / "jobs/package/artifact"
    package.mkdir(parents=True, mode=0o700)
    payload = b"Generated text package. Not model weights.\n"
    (package / "generated.txt").write_bytes(payload)
    manifest = {
        "schema_version": 1,
        "metadata": {"generated_fixture": True},
        "files": {"generated.txt": sha256(payload)},
    }
    write_new(package / "manifest.json", manifest)
    artifact = PolicyArtifact(
        id="package:operation",
        project_id=PROJECT,
        job_id="package",
        label="Generated text package, not policy weights",
        format="deployment_package",
        path="jobs/package/artifact",
        manifest_sha256=sha256((package / "manifest.json").read_bytes()),
        file_bytes=len(payload),
        metadata={"generated_fixture": True, "quality_verified": False},
    )
    records = [
        Job(
            id="intake",
            project_id=PROJECT,
            kind="dataset.inspect",
            status="succeeded",
            request=IntakeRequest(source="local", path="generated-dataset"),
            result=profile,
            created_at=STAMP,
            updated_at=STAMP,
        ),
        Job(
            id="package",
            project_id=PROJECT,
            kind="policy.import",
            status="succeeded",
            request=PolicyRequest(
                operation="policy.import", runtime_id="never-executed", source_id="generated-text"
            ),
            result=LifecycleResult(artifacts=[artifact], selected_artifact_id=artifact.id),
            created_at=STAMP,
            updated_at=STAMP,
        ),
    ]
    project = Project(id=PROJECT, name="Generated desktop round-trip proof", created_at=STAMP)
    with sqlite3.connect(workspace / "workspace.sqlite3") as database:
        database.execute(
            "INSERT INTO projects VALUES (?,?,?)", (project.id, project.name, project.created_at)
        )
        for job in records:
            database.execute(
                "INSERT INTO jobs VALUES (?,?,?,?)",
                (job.id, job.project_id, job.status, job.model_dump_json()),
            )
            folder = workspace / "jobs" / job.id
            folder.mkdir(parents=True, exist_ok=True)
            write_new(
                folder / "events.jsonl",
                {
                    "sequence": 1,
                    "stage": "fixture",
                    "message": "Generated record; no operation executed",
                    "timestamp": STAMP,
                    "data": {},
                },
            )
    write_new(
        workspace / "private-settings-sentinel.json",
        {"sentinel": "Generated bytes; not credentials; never activated"},
    )
    (workspace / "private-settings-sentinel.json").chmod(0o600)
    (workspace / "empty-directory").mkdir(mode=0o750)


def record_identity(workspace: Path, files: dict, budget: maintenance.Budget) -> dict:
    """Compare indexed IDs and full typed JSON, not just database row counts."""
    from vla_platform.contracts import Job, Project

    summary = maintenance.validate_database(workspace, files, budget)
    uri = (workspace / "workspace.sqlite3").as_uri() + "?mode=ro&immutable=1"
    with closing(sqlite3.connect(uri, uri=True, timeout=0)) as database:
        database.execute("PRAGMA query_only=ON")
        database.execute("PRAGMA trusted_schema=OFF")
        projects = [
            Project.model_validate(
                dict(zip(("id", "name", "created_at"), row, strict=True)), strict=True
            ).model_dump(mode="json")
            for row in database.execute("SELECT id,name,created_at FROM projects ORDER BY id")
        ]
        jobs = []
        for ident, project, status, raw in database.execute(
            "SELECT id,project_id,status,record FROM jobs ORDER BY id"
        ):
            budget.check()
            job = Job.model_validate(maintenance.parse_json(raw.encode()), strict=True)
            jobs.append(
                {
                    "index": [ident, project, status],
                    "stored_json_sha256": sha256(raw.encode()),
                    "record": job.model_dump(mode="json"),
                }
            )
    return {"database": summary, "projects": projects, "jobs": jobs}


def verify_tree(path: Path, expected: dict, budget: maintenance.Budget) -> None:
    if maintenance.inventory(path, budget) != expected:
        raise maintenance.MaintenanceError("Sealed workspace inventory changed")


def fresh_copy(
    backup: Path,
    target: Path,
    expected: dict,
    records: dict,
    backup_tree: dict,
    budget: maintenance.Budget,
) -> dict:
    """Copy a verified backup only to an exclusive fresh sibling directory."""
    verify_tree(backup, backup_tree, budget)
    target.mkdir(mode=0o700)  # Never replace or merge an existing destination.
    maintenance.copy_files(backup / "workspace", target, expected, budget)
    verify_tree(target, expected, budget)
    identity = record_identity(target, expected, budget)
    if identity != records:
        raise maintenance.MaintenanceError("Copied record identity differs")
    verify_tree(backup, backup_tree, budget)
    return identity


def run(output: Path) -> dict:
    """Create a new generated fixture and prove two exact fresh-copy round trips."""
    output = Path(output)
    maintenance.directory(output.parent)
    if not output.is_absolute() or ".." in output.parts:
        raise maintenance.MaintenanceError("An absolute new proof directory is required")
    output.mkdir(mode=0o700)
    budget = maintenance.Budget(maintenance.Limits())
    with no_application_imports():
        original = output / "original"
        asyncio.run(seed(original))
        baseline = maintenance.inventory(original, budget)
        records = record_identity(original, baseline, budget)
        write_new(output / "sealed-original.json", {"files": baseline, "records": records})
        backup = output / "backup"
        receipt = maintenance.prepare_backup(original, backup)
        if receipt["files"] != baseline or receipt["marker"] != maintenance.marker(
            original, budget
        ):
            raise maintenance.MaintenanceError("Backup receipt differs from sealed original")
        with maintenance.opened(backup / "receipt.json") as stream:
            saved = stream.read(budget.limits.json_bytes + 1)
        if saved != maintenance.canonical(receipt) + b"\n":
            raise maintenance.MaintenanceError("Published backup receipt differs")
        backup_tree = maintenance.inventory(backup, budget)
        write_new(output / "sealed-backup.json", backup_tree)
        outcomes = {}
        for label in ("candidate", "roundtrip"):
            outcomes[label] = fresh_copy(
                backup, output / label, baseline, records, backup_tree, budget
            )
        # Recheck all sealed trees together after both copies and all SQLite reads.
        verify_tree(original, baseline, budget)
        verify_tree(backup, backup_tree, budget)
        for label in outcomes:
            verify_tree(output / label, baseline, budget)
        source_paths = {Path(__file__), Path(maintenance.__file__)}
        source_paths.update(
            Path(module.__file__)
            for name, module in list(sys.modules.items())
            if name.startswith("vla_platform.") and getattr(module, "__file__", None)
        )
        root = Path(__file__).resolve().parents[3]
        source = {
            str(path.resolve().relative_to(root)): sha256(path.read_bytes())
            for path in sorted(source_paths)
        }
        result = {
            "schema_version": 1,
            "operation": "desktop.workspace.generated_roundtrip",
            "complete": True,
            "fixture_kind": "generated_metadata_and_text_artifact",
            "python": sys.version,
            "source_files": source,
            "fixture_payload_identity": "generated_only_no_real_payload",
            "files": baseline,
            "inventory_sha256": sha256(maintenance.canonical(baseline)),
            "record_identity": records,
            "record_identity_sha256": sha256(maintenance.canonical(records)),
            "backup_receipt_sha256": sha256((backup / "receipt.json").read_bytes()),
            "sealed_backup_inventory_sha256": sha256(maintenance.canonical(backup_tree)),
            "original_unchanged": True,
            "backup_unchanged": True,
            "candidate_exact": True,
            "fresh_roundtrip_exact": True,
            "migration_performed_on_sealed_data": False,
            "upgrade_performed": False,
            "activation_performed": False,
            "api_or_execution_started": False,
            "native_launch_verified": False,
            "frozen_payload_verified": False,
            "claim": (
                "Current source accepts generated revision0001 records and an exact backup "
                "can be copied to fresh directories; not an upgrade or in-place restore."
            ),
        }
        budget.check()
        write_new(output / "receipt.json", result)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sys.addaudithook(deny_external_actions)
    result = run(args.output)
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "receipt": str(args.output / "receipt.json"),
                "inventory_sha256": result["inventory_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
