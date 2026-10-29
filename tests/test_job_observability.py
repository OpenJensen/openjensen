"""Durable diagnostic contracts; no workers, model downloads or provider calls."""

import io
import json
import sqlite3
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from vla_platform.job_observability import (
    JobObservability,
    ObservabilityError,
    ObservabilityPolicy,
    job_key,
    redact,
)
from vla_platform.job_observability_api import router


def test_output_is_redacted_before_persistence(tmp_path):
    store = JobObservability(tmp_path)
    store.record(
        "job-1",
        "\x1b[31mfailed\x1b[0m token=private-value bearer private-bearer "
        "https://operator:password@compute.example.invalid/download?signature=private "
        "/home/example/private/model.bin",
        kind="log",
        stream="stderr",
        data={"env": {"PASSWORD": "hidden"}, "hostname": "private-host", "step": 3},
    )
    row = store.page("job-1").records[0]
    assert row.stream == "stderr" and row.kind == "log"
    assert row.job_id == "core:job-1"
    assert row.data == {"env": "[redacted]", "hostname": "[redacted]", "step": 3}
    with sqlite3.connect(store.path) as connection:
        raw = repr(connection.execute("SELECT message,data FROM diagnostic_records").fetchall())
    for private in ("private-value", "private-bearer", "compute.example.invalid", "/home/example"):
        assert private not in raw
    assert "\x1b" not in raw


def test_page_reconnect_and_reopen_do_not_duplicate_records(tmp_path):
    store = JobObservability(tmp_path)
    written = [
        store.record("scene:scene-1", f"stage {index}", attempt_id="2") for index in range(7)
    ]
    first = store.page("scene:scene-1", limit=3)
    assert first.has_more and len(first.records) == 3
    second = JobObservability(tmp_path).page("scene:scene-1", after=first.next_cursor, limit=3)
    third = store.page("scene:scene-1", after=second.next_cursor, limit=3)
    actual = first.records + second.records + third.records
    assert [row.sequence for row in actual] == [row.sequence for row in written]
    assert all(row.attempt_id == "2" for row in actual)
    assert not third.has_more
    assert store.page("scene:scene-1", after=third.next_cursor).records == []


def test_filters_advance_past_unmatched_records(tmp_path):
    store = JobObservability(tmp_path)
    store.record("job-1", "loaded", kind="log", stream="stdout")
    target = store.record("job-1", "GPU failed", kind="log", stream="stderr", severity="error")
    final = store.record("job-1", "cleanup complete")
    result = store.page("job-1", severity="error", stream="stderr", query="gpu")
    assert [row.sequence for row in result.records] == [target.sequence]
    assert result.next_cursor == final.sequence
    assert store.page("job-1", query="absent").next_cursor == final.sequence


def test_retention_is_bounded_and_final_error_is_independent(tmp_path):
    policy = ObservabilityPolicy(max_log_bytes=65536, max_event_records=100)
    store = JobObservability(tmp_path, policy)
    store.failure(
        "job-1",
        stage="loading",
        code="MODEL_LOAD_FAILED",
        message="Model was not loaded",
        cause="Exit status 1",
        exit_code=1,
        cleanup="succeeded",
    )
    for index in range(70):
        store.record(
            "job-1", str(index) + "x" * 2000, kind="log", severity="error", stream="stderr"
        )
    page = store.page("job-1", limit=500)
    assert page.truncated and page.discarded_records > 0
    assert any(row.component == "diagnostics" and row.severity == "warning" for row in page.records)
    with sqlite3.connect(store.path) as connection:
        retained = connection.execute(
            "SELECT SUM(bytes) FROM diagnostic_records WHERE kind='log'"
        ).fetchone()[0]
    assert retained <= policy.max_log_bytes
    failure = store.get_failure("job-1")
    assert failure.code == "MODEL_LOAD_FAILED" and failure.exit_code == 1
    assert failure.cleanup == "succeeded"


def test_event_retention_has_explicit_marker(tmp_path):
    store = JobObservability(tmp_path, ObservabilityPolicy(max_event_records=100))
    for index in range(110):
        store.record("job-1", str(index))
    page = store.page("job-1", limit=500)
    assert page.truncated and page.discarded_records == 10
    assert len(page.records) == 101


def test_record_bounds_and_unicode_are_safe(tmp_path):
    store = JobObservability(tmp_path, ObservabilityPolicy(max_record_bytes=1024))
    row = store.record("job-1", "\ud800" + "😃" * 2000, data={"large": "y" * 5000})
    assert "[record truncated]" in row.message
    assert len(row.message.encode()) < 1024
    assert row.data == {"record_data_truncated": True}


def test_export_contains_only_redacted_bounded_evidence(tmp_path):
    store = JobObservability(tmp_path)
    store.record("job-1", "Downloading from https://private.example.invalid/model")
    store.failure(
        "job-1", stage="download", message="token=hidden-value", cause="/home/example/model missing"
    )
    archive = store.export(
        "job-1",
        {
            "id": "job-1",
            "environment": {"SECRET": "hidden"},
            "inputs": {"path": "/private/data"},
            "application_version": "test",
        },
    )
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        assert set(bundle.namelist()) == {
            "job.json",
            "failure.json",
            "diagnostics.jsonl",
            "retention.json",
        }
        contents = "\n".join(bundle.read(name).decode() for name in bundle.namelist())
        assert json.loads(bundle.read("job.json"))["application_version"] == "test"
    assert "hidden-value" not in contents
    assert "private.example.invalid" not in contents
    assert "/home/example" not in contents
    assert "/private/data" not in contents


def test_project_and_job_records_are_isolated(tmp_path):
    store = JobObservability(tmp_path)
    store.record("job-1", "First project")
    store.record("job-2", "Second project")
    assert [row.message for row in store.page("job-1").records] == ["First project"]
    assert not store.page("job-3").available


def test_concurrent_writers_have_ordered_unique_cursors(tmp_path):
    def append(index):
        return JobObservability(tmp_path).record("job-1", str(index)).sequence

    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(append, range(24)))
    assert len(set(values)) == 24
    assert [row.sequence for row in JobObservability(tmp_path).page("job-1").records] == sorted(
        values
    )


def test_storage_failure_is_explicit_but_safe_hooks_do_not_raise(tmp_path):
    target = tmp_path / "private.sqlite3"
    target.write_text("keep")
    (tmp_path / "job-observability.sqlite3").symlink_to(target)
    store = JobObservability(tmp_path)
    with pytest.raises(ObservabilityError):
        store.record("job-1", "must not write")
    with pytest.raises(ObservabilityError):
        store.page("job-1")
    assert store.try_record("job-1", "must not write") is None
    assert store.try_failure("job-1", stage="run", message="must not write") is None
    assert target.read_text() == "keep"


@pytest.mark.parametrize("value", ["../elsewhere", "job/../other", "", "a" * 200])
def test_job_identity_cannot_be_a_file_path(value):
    with pytest.raises(ValueError):
        job_key(value)


def test_private_keys_and_control_sequences_are_removed():
    assert "private-data" not in redact(
        "-----BEGIN PRIVATE KEY-----\nprivate-data\n-----END PRIVATE KEY-----"
    )
    assert redact("safe\x00\x1b]0;untrusted\x07message") == "safemessage"


@pytest.fixture
def api(tmp_path, monkeypatch):
    app = FastAPI()
    app.include_router(router)
    app.state.job_observability = JobObservability(tmp_path)

    async def project(ident):
        return object() if ident in {"project-1", "project-2"} else None

    app.state.projects = SimpleNamespace(get=project)
    app.state.execution = SimpleNamespace(settings=SimpleNamespace(data_dir=tmp_path))

    async def job(state, project_id, key):
        if project_id != "project-1" or key != "core:job-1":
            return None
        return SimpleNamespace(
            id="core:job-1", source_id="job-1", model_dump=lambda **kwargs: {"id": "core:job-1"}
        )

    monkeypatch.setattr("vla_platform.job_observability_api.get_job", job)
    with TestClient(app) as client:
        yield client, app.state.job_observability


def test_api_checks_project_before_log_and_export_access(api):
    client, store = api
    store.record("job-1", "private evidence")
    for suffix in ("logs", "failure", "diagnostics.zip"):
        assert (
            client.get(f"/api/v1/projects/project-2/work-jobs/core:job-1/{suffix}").status_code
            == 404
        )
        assert (
            client.get(f"/api/v1/projects/missing/work-jobs/core:job-1/{suffix}").status_code == 404
        )
    assert (
        client.get("/api/v1/projects/project-1/work-jobs/core:job-1/logs").json()["records"][0][
            "message"
        ]
        == "private evidence"
    )
    assert client.get("/api/v1/projects/project-1/work-jobs/core:job-1/failure").json() is None
    exported = client.get("/api/v1/projects/project-1/work-jobs/core:job-1/diagnostics.zip")
    assert exported.status_code == 200 and exported.headers["content-type"] == "application/zip"


def test_api_rejects_unbounded_log_reads(api):
    client, _ = api
    base = "/api/v1/projects/project-1/work-jobs/core:job-1/logs"
    for query in ("limit=501", "limit=0", "after=-1", "query=" + "x" * 201, "stream=other"):
        assert client.get(base + "?" + query).status_code == 422


def test_age_retention_preserves_failure_summary_and_marks_removed_history(tmp_path):
    store = JobObservability(tmp_path, ObservabilityPolicy(retention_days=1))
    store.record("job-1", "old output", kind="log")
    store.failure("job-1", stage="run", message="retained final error")
    old = (datetime.now(UTC) - timedelta(days=2)).isoformat()
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE diagnostic_records SET timestamp=?", (old,))
    assert store.expire() == 2
    page = store.page("job-1")
    assert page.available and page.truncated and page.discarded_records == 2
    assert page.records == []
    assert store.get_failure("job-1").message == "retained final error"


def test_private_key_lines_are_redacted_across_worker_messages(tmp_path):
    store = JobObservability(tmp_path)
    store.record("job-1", "-----BEGIN PRIVATE KEY-----", kind="log", stream="stderr")
    store.record("job-1", "base64-private-material", kind="log", stream="stderr")
    store.record("job-1", "-----END PRIVATE KEY-----", kind="log", stream="stderr")
    store.record("job-1", "resumed output", kind="log", stream="stderr")
    messages = [row.message for row in store.page("job-1").records]
    assert "base64-private-material" not in messages
    assert messages[-1] == "resumed output"


def test_generic_failure_does_not_erase_provider_failure_evidence(tmp_path):
    store = JobObservability(tmp_path)
    store.failure(
        "job-1",
        stage="worker",
        code="WORKER_EXIT",
        message="Worker exited with status 2",
        cause="A bounded worker error",
        exit_code=2,
        cleanup="succeeded",
    )
    merged = store.failure("job-1", stage="failed", message="Job failed")
    assert merged.code == "WORKER_EXIT" and merged.stage == "worker"
    assert merged.exit_code == 2 and merged.cleanup == "succeeded"
    assert merged.cause == "A bounded worker error"
    assert store.get_failure("job-1") == merged
    new_attempt = store.failure(
        "job-1", stage="run", message="Another attempt failed", attempt_id="2"
    )
    assert new_attempt.exit_code is None and new_attempt.cleanup == "unknown"


@pytest.mark.parametrize("malformed", ["{", '{"max_log_bytes":1}', '{"retention_days":"unknown"}'])
def test_invalid_retention_policy_does_not_disable_workspace_access(
    tmp_path, monkeypatch, malformed
):
    from vla_platform.api import create_app
    from vla_platform.execution import Execution
    from vla_platform.settings import Settings

    path = tmp_path / "job-observability-settings.json"
    path.write_text(malformed)
    monkeypatch.setattr(Execution, "_schedule", lambda self, ident: None)
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        assert client.get("/api/v1/health").status_code == 200
        project = client.post("/api/v1/projects", json={"name": "Accessible workspace"})
        assert project.status_code == 201
        ident = project.json()["id"]
        assert client.get(f"/api/v1/projects/{ident}/jobs").json() == []
        imported = client.post(
            f"/api/v1/projects/{ident}/dataset-uploads", json={"name": "An upload"}
        )
        assert imported.status_code == 201
        jobs = client.get(f"/api/v1/projects/{ident}/work-jobs")
        assert jobs.status_code == 200 and len(jobs.json()) == 1
        key = jobs.json()[0]["id"]
        for suffix in ("logs", "failure", "diagnostics.zip"):
            response = client.get(f"/api/v1/projects/{ident}/work-jobs/{key}/{suffix}")
            assert response.status_code == 503
            assert "diagnostic" in response.json()["detail"].lower()
            assert "max_log_bytes" not in response.text
        assert client.get("/api/v1/job-diagnostics-settings").status_code == 503
    assert path.read_text() == malformed
    assert not (tmp_path / "job-observability.sqlite3").exists()


def test_invalid_retention_policy_never_uses_an_implicit_write_policy(tmp_path):
    (tmp_path / "job-observability-settings.json").write_text("{")
    store = JobObservability(tmp_path)
    assert store.try_record("job-1", "must not be persisted") is None
    assert store.try_failure("job-1", stage="run", message="must not be persisted") is None
    with pytest.raises(ObservabilityError):
        store.record("job-1", "strict write")
    with pytest.raises(ObservabilityError):
        store.page("job-1")
    assert not store.path.exists()


@pytest.mark.parametrize(
    "header",
    [
        "Authorization: Basic dXNlcjpwYXNz",
        "Proxy-Authorization: Basic dXNlcjpwYXNz",
        "authorization=basic dXNlcjpwYXNz",
    ],
)
def test_basic_authentication_credentials_never_persist_or_export(tmp_path, header):
    store = JobObservability(tmp_path)
    store.record("job-1", header, kind="log", stream="stderr")
    assert "dXNlcjpwYXNz" not in store.page("job-1").records[0].message
    with zipfile.ZipFile(io.BytesIO(store.export("job-1", {"message": header}))) as archive:
        for name in archive.namelist():
            assert b"dXNlcjpwYXNz" not in archive.read(name)


def test_sensitive_dictionary_keys_are_redacted_before_storage(tmp_path):
    store = JobObservability(tmp_path)
    record = store.record(
        "job-1",
        "metadata",
        data={"api_key=private-value": "ignored", "password=another-value": "ignored"},
    )
    assert "private-value" not in record.model_dump_json()
    assert "another-value" not in record.model_dump_json()
    assert "private-value" not in store.page("job-1").model_dump_json()
