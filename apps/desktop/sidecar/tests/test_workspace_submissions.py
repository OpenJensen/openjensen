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


def add_teaching_submission(root, status="failed"):
    """Generated saved records only; no session or capture is launched."""
    from vla_platform.contracts import Job
    from vla_platform.teaching_sessions.contracts import TeachingCaptureRequest

    add_submissions(root)
    request = TeachingCaptureRequest(profile_id="local-isaac", profile_sha256="a" * 64).model_dump(
        mode="json"
    )
    with sqlite3.connect(root / "workspace.sqlite3") as db:
        value = json.loads(db.execute("SELECT record FROM jobs").fetchone()[0])
        value.update(kind="teaching.capture", status=status, request=request, result=None)
        current = Job.model_validate(value)
        accepted = current.model_copy(update={"status": "queued", "error": None})
        db.execute("UPDATE jobs SET status=?,record=?", (status, current.model_dump_json()))
        db.execute(
            "UPDATE job_submissions SET operation=?,request_record=?,request_sha256=?,"
            "accepted_response=?",
            (
                "teaching.capture",
                json.dumps(request),
                hashlib.sha256(m.canonical(request)).hexdigest(),
                accepted.model_dump_json(),
            ),
        )
    return request


@pytest.mark.parametrize("status", ["failed", "cancelled", "interrupted"])
def test_terminal_teaching_key_and_exact_recipe_preserved(workspace, status):  # noqa: F811
    request = add_teaching_submission(workspace, status)
    before = hashes(workspace)
    result = run(workspace)
    assert result["database"]["job_submissions"] == result["database"]["jobs"] == 1
    assert hashes(workspace) == before == hashes(workspace.parent / "new-backup/workspace")
    with sqlite3.connect(workspace.parent / "new-backup/workspace/workspace.sqlite3") as db:
        row = db.execute(
            "SELECT operation,idempotency_key,request_record,request_sha256,accepted_response "
            "FROM job_submissions"
        ).fetchone()
        saved = json.loads(db.execute("SELECT record FROM jobs").fetchone()[0])
    accepted = json.loads(row[4])
    assert row[:2] == ("teaching.capture", "attempt-1")
    assert json.loads(row[2]) == accepted["request"] == saved["request"] == request
    assert row[3] == hashlib.sha256(m.canonical(request)).hexdigest()
    assert accepted["status"] == "queued" and saved["status"] == status
    assert saved["result"] is None


@pytest.mark.parametrize(
    "fault",
    [
        "original-profile",
        "original-profile-hash",
        "original-timeout",
        "accepted-profile",
        "fingerprint",
        "operation",
    ],
)
def test_teaching_submission_mismatch_preserves_original(workspace, fault):  # noqa: F811
    request = add_teaching_submission(workspace)
    with sqlite3.connect(workspace / "workspace.sqlite3") as db:
        if fault.startswith("original-"):
            key, value = {
                "original-profile": ("profile_id", "another-profile"),
                "original-profile-hash": ("profile_sha256", "b" * 64),
                "original-timeout": ("timeout_seconds", 301),
            }[fault]
            request[key] = value
            db.execute(
                "UPDATE job_submissions SET request_record=?,request_sha256=?",
                (json.dumps(request), hashlib.sha256(m.canonical(request)).hexdigest()),
            )
        elif fault == "accepted-profile":
            accepted = json.loads(
                db.execute("SELECT accepted_response FROM job_submissions").fetchone()[0]
            )
            accepted["request"]["profile_id"] = "another-profile"
            db.execute("UPDATE job_submissions SET accepted_response=?", (json.dumps(accepted),))
        elif fault == "fingerprint":
            db.execute("UPDATE job_submissions SET request_sha256=?", ("0" * 64,))
        else:
            db.execute("UPDATE job_submissions SET operation='policy.run'")
    before = hashes(workspace)
    with pytest.raises(m.MaintenanceError):
        run(workspace)
    assert hashes(workspace) == before
    assert not (workspace.parent / "new-backup").exists()


@pytest.mark.parametrize("status", ["queued", "running"])
def test_active_teaching_still_refuses_offline_maintenance(workspace, status):  # noqa: F811
    add_teaching_submission(workspace, status)
    before = hashes(workspace)
    with pytest.raises(m.MaintenanceError, match="Active jobs"):
        run(workspace)
    assert hashes(workspace) == before
    assert not (workspace.parent / "new-backup").exists()


def test_published_teaching_refusal_preserves_external_capture(workspace):  # noqa: F811
    from vla_platform.contracts import Job
    from vla_platform.teaching_sessions.contracts import TeachingCaptureResult

    add_teaching_submission(workspace, "succeeded")
    session_id = "c" * 32
    result = TeachingCaptureResult(
        profile_id="local-isaac",
        profile_sha256="a" * 64,
        recording_configuration_sha256="b" * 64,
        session_id=session_id,
        session_sha256="d" * 64,
        inventory_sha256="e" * 64,
        episodes=[
            {
                "episode_id": "f" * 32,
                "receipt_sha256": "0" * 64,
                "frames": 1,
                "termination": "finish",
                "outcome": "unknown",
            }
        ],
        origin="synthetic",
        lineage_group="generated-maintenance-fixture",
    )
    external = workspace.parent / "external-captures" / session_id
    external.mkdir(parents=True)
    (external / "session.json").write_text('{"fixture":"opaque external capture"}')
    (external / "retained-frame.rgb").write_bytes(b"generated external fixture bytes")
    with sqlite3.connect(workspace / "workspace.sqlite3") as db:
        value = json.loads(db.execute("SELECT record FROM jobs").fetchone()[0])
        value["result"] = result.model_dump(mode="json")
        db.execute("UPDATE jobs SET record=?", (Job.model_validate(value).model_dump_json(),))
    before, external_before = hashes(workspace), hashes(external)
    with pytest.raises(m.MaintenanceError, match="external catalog/inventory"):
        run(workspace)
    assert hashes(workspace) == before and hashes(external) == external_before
    assert not (workspace.parent / "new-backup").exists()
