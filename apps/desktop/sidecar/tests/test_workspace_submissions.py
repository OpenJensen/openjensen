"""Read-only backup of durable request identities; generated SQLite fixtures only."""

import hashlib
import json
import sqlite3

import pytest
from test_workspace_maintenance import hashes, m, run, workspace  # noqa: F401


def add_submissions(root):
    from vla_platform.contracts import IntakeRequest, Job

    original = IntakeRequest(repo_id="fixture/generated").model_dump(mode="json")
    with sqlite3.connect(root / "workspace.sqlite3") as db:
        db.executescript("""
            UPDATE alembic_version SET version_num='0002';
            CREATE TABLE job_submissions (
                project_id VARCHAR NOT NULL,
                operation VARCHAR NOT NULL,
                idempotency_key VARCHAR NOT NULL,
                request_sha256 VARCHAR NOT NULL,
                fingerprint_version INTEGER NOT NULL,
                request_record JSON NOT NULL,
                job_id VARCHAR NOT NULL,
                accepted_response JSON NOT NULL,
                created_at VARCHAR NOT NULL,
                PRIMARY KEY (project_id, operation, idempotency_key),
                FOREIGN KEY(job_id) REFERENCES jobs(id)
            );
            CREATE INDEX ix_job_submissions_job_id ON job_submissions(job_id);
        """)
        value = json.loads(db.execute("SELECT record FROM jobs").fetchone()[0])
        value["request"]["revision"] = "a" * 40
        current = Job.model_validate(value)
        db.execute("UPDATE jobs SET record=?", (current.model_dump_json(),))
        accepted = current.model_copy(update={"status": "queued", "error": None})
        db.execute(
            "INSERT INTO job_submissions VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "project",
                "dataset.inspect",
                "attempt-1",
                hashlib.sha256(m.canonical(original)).hexdigest(),
                1,
                json.dumps(original),
                "job",
                accepted.model_dump_json(),
                "2026-09-27T01:00:00+00:00",
            ),
        )


def test_backup_preserves_original_request_and_enriched_accepted_job(workspace):  # noqa: F811
    add_submissions(workspace)
    before = hashes(workspace)
    result = run(workspace)
    assert result["database"]["alembic_revision"] == "0002"
    assert result["database"]["job_submissions"] == 1
    assert hashes(workspace) == before
    assert hashes(workspace.parent / "new-backup/workspace") == before
    with sqlite3.connect(workspace.parent / "new-backup/workspace/workspace.sqlite3") as db:
        request, accepted = db.execute(
            "SELECT request_record,accepted_response FROM job_submissions"
        ).fetchone()
    assert json.loads(request)["revision"] == "main"
    assert json.loads(accepted)["request"]["revision"] == "a" * 40
    assert json.loads(accepted)["status"] == "queued"


def test_multiple_keys_can_preserve_one_cached_terminal_job(workspace):  # noqa: F811
    add_submissions(workspace)
    with sqlite3.connect(workspace / "workspace.sqlite3") as db:
        current = db.execute("SELECT record FROM jobs").fetchone()[0]
        db.execute("UPDATE job_submissions SET accepted_response=?", (current,))
        db.execute("""
            INSERT INTO job_submissions
            SELECT project_id,operation,'attempt-2',request_sha256,fingerprint_version,
                   request_record,job_id,accepted_response,created_at
            FROM job_submissions
        """)
    result = run(workspace)
    assert result["database"]["job_submissions"] == 2
    assert result["database"]["jobs"] == 1


@pytest.mark.parametrize(
    "fault",
    [
        "fingerprint",
        "version",
        "key",
        "project",
        "operation",
        "missing-job",
        "accepted-id",
        "accepted-request",
        "original-operation",
        "timestamp",
        "duplicate-json",
        "unnormalized-request",
        "index",
        "revision",
        "extra-table",
    ],
)
def test_invalid_submission_identity_refused_without_touching_original(workspace, fault):  # noqa: F811
    add_submissions(workspace)
    with sqlite3.connect(workspace / "workspace.sqlite3") as db:
        columns = {
            "fingerprint": ("request_sha256", "0" * 64),
            "version": ("fingerprint_version", 2),
            "key": ("idempotency_key", "bad key"),
            "project": ("project_id", "other"),
            "operation": ("operation", "policy.run"),
            "missing-job": ("job_id", "missing"),
            "timestamp": ("created_at", "today"),
            "duplicate-json": ("request_record", '{"revision":"main","revision":"other"}'),
        }
        if fault in columns:
            name, value = columns[fault]
            db.execute(f"UPDATE job_submissions SET {name}=?", (value,))
        elif fault.startswith("accepted-"):
            value = json.loads(
                db.execute("SELECT accepted_response FROM job_submissions").fetchone()[0]
            )
            if fault == "accepted-id":
                value["id"] = "other"
            else:
                value["request"]["repo_id"] = "other/dataset"
            db.execute("UPDATE job_submissions SET accepted_response=?", (json.dumps(value),))
        elif fault in {"original-operation", "unnormalized-request"}:
            value = json.loads(
                db.execute("SELECT request_record FROM job_submissions").fetchone()[0]
            )
            if fault == "original-operation":
                value = {"operation": "policy.run", "runtime_id": "fixture"}
            else:
                value["repo_id"] = " fixture/generated "
            db.execute(
                "UPDATE job_submissions SET request_record=?,request_sha256=?",
                (
                    json.dumps(value),
                    hashlib.sha256(m.canonical(value)).hexdigest(),
                ),
            )
        elif fault == "index":
            db.executescript(
                "DROP INDEX ix_job_submissions_job_id; "
                "CREATE INDEX ix_job_submissions_job_id ON job_submissions(operation);"
            )
        elif fault == "revision":
            db.execute("UPDATE alembic_version SET version_num='9999'")
        else:
            db.execute("CREATE TABLE unreviewed (id TEXT)")
    before = hashes(workspace)
    with pytest.raises(m.MaintenanceError):
        run(workspace)
    assert hashes(workspace) == before
    assert not (workspace.parent / "new-backup").exists()


def test_fingerprint_bound_uses_canonical_json_not_storage_whitespace(workspace):  # noqa: F811
    add_submissions(workspace)
    with sqlite3.connect(workspace / "workspace.sqlite3") as db:
        raw = db.execute("SELECT request_record FROM job_submissions").fetchone()[0]
        # Serializer whitespace is not part of the canonical 1 MiB fingerprint bound.
        db.execute("UPDATE job_submissions SET request_record=?", (raw + " " * (1024 * 1024),))
    assert run(workspace)["database"]["job_submissions"] == 1
