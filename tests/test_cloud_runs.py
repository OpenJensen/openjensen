import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from vla_platform.api import create_app
from vla_platform.cloud_runs import MAX_LOG_BYTES, MAX_SNAPSHOT_BYTES, read_cloud_runs
from vla_platform.settings import Settings


def snapshot(**changes):
    return {
        "schema_version": 1,
        "run_id": "test-run",
        "label": "Synthetic cloud feed test",
        "cluster": "test-group",
        "job_id": "7",
        "collected_at": datetime.now(UTC).isoformat(),
        "status": "SUCCEEDED",
        "collection_error": None,
        "logs": {"isaac": "rollout process finished\n", "vla": "policy server started\n"},
        "outcomes": {
            "rollout_completed": True,
            "pickup_success": None,
            "calibration": "unverified",
        },
        **changes,
    }


def write(directory, data=None, name="test-run.json"):
    path = directory / name
    path.write_text(json.dumps(data or snapshot()), encoding="utf-8")
    return path


def test_cloud_feed_disabled_by_default_and_read_only(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        response = client.get("/api/v1/cloud-runs")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.json()["enabled"] is False
        assert response.json()["runs"] == response.json()["errors"] == []
        assert client.post("/api/v1/cloud-runs").status_code == 405
        assert client.get("/api/v1/cloud-runs", headers={"Host": "bad.invalid"}).status_code == 400
        assert (
            client.get("/api/v1/cloud-runs", headers={"Origin": "https://bad.invalid"}).status_code
            == 403
        )


def test_api_returns_only_configured_snapshots_and_distinct_outcomes(tmp_path):
    directory = tmp_path / "feed"
    directory.mkdir()
    write(directory)
    with TestClient(
        create_app(Settings(data_dir=tmp_path / "app", cloud_runs_dir=directory))
    ) as client:
        response = client.get(
            "/api/v1/cloud-runs", params={"path": "/not/operator/approved", "command": "ignored"}
        )
    data = response.json()
    assert data["enabled"] is True and data["errors"] == []
    run = data["runs"][0]
    assert run["stale"] is False and 0 <= run["age_seconds"] < 10
    assert run["status"] == "SUCCEEDED"
    assert run["outcomes"] == {
        "rollout_completed": True,
        "pickup_success": None,
        "calibration": "unverified",
    }
    assert run["logs"]["isaac"] == "rollout process finished\n"
    assert str(directory) not in response.text


def test_stale_collection_error_and_no_observation_are_explicit(tmp_path):
    collected = (datetime.now(UTC) - timedelta(seconds=100)).isoformat()
    write(
        tmp_path, snapshot(collected_at=collected, collection_error="Remote collection unavailable")
    )
    feed = read_cloud_runs(tmp_path)
    assert feed.runs[0].stale and feed.runs[0].age_seconds >= 100
    assert feed.runs[0].collection_error == "Remote collection unavailable"
    assert feed.runs[0].collected_at.isoformat() == collected
    write(
        tmp_path,
        snapshot(collected_at=None, status="UNKNOWN", collection_error="Initial collection failed"),
    )
    run = read_cloud_runs(tmp_path).runs[0]
    assert run.collected_at is None and run.age_seconds is None and run.stale


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"run_id": "../escape"},
        {"run_id": "other"},
        {"job_id": "not-numeric"},
        {"collected_at": "2026-09-26T00:00:00"},
        {"collected_at": 123},
        {"credentials": "must-not-appear-in-errors"},
        {"outcomes": {"pickup_success": "true"}},
        {"logs": {"isaac": "é" * (MAX_LOG_BYTES // 2 + 1), "vla": ""}},
        {"logs": {"isaac": "x" * (MAX_LOG_BYTES + 1), "vla": ""}},
        {"collected_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat()},
    ],
)
def test_invalid_snapshot_is_reported_without_exposing_its_content(tmp_path, change):
    write(tmp_path, snapshot(**change))
    feed = read_cloud_runs(tmp_path)
    assert not feed.runs and len(feed.errors) == 1
    assert feed.errors[0].run_id == "test-run"
    assert "must-not-appear" not in feed.model_dump_json()


def test_malformed_new_snapshot_does_not_hide_error_or_other_run(tmp_path):
    write(tmp_path)
    (tmp_path / "newer.json").write_text('{"schema_version":1,')
    (tmp_path / ".writing.tmp").write_text("incomplete private staging")
    feed = read_cloud_runs(tmp_path)
    assert [run.run_id for run in feed.runs] == ["test-run"]
    assert [error.run_id for error in feed.errors] == ["newer"]


def test_atomic_replacement_is_visible_on_next_read(tmp_path):
    write(tmp_path, snapshot(status="RUNNING"))
    assert read_cloud_runs(tmp_path).runs[0].status == "RUNNING"
    replacement = write(
        tmp_path,
        snapshot(status="FAILED", logs={"isaac": "explicit failure", "vla": ""}),
        ".snapshot.tmp",
    )
    os.replace(replacement, tmp_path / "test-run.json")
    assert read_cloud_runs(tmp_path).runs[0].status == "FAILED"
    assert read_cloud_runs(tmp_path).runs[0].logs.isaac == "explicit failure"


def test_file_and_directory_bounds_fail_visibly(tmp_path):
    (tmp_path / "test-run.json").write_bytes(b"x" * (MAX_SNAPSHOT_BYTES + 1))
    assert "size limit" in read_cloud_runs(tmp_path).errors[0].message
    for index in range(50):
        write(tmp_path, snapshot(run_id=f"run-{index}"), f"run-{index}.json")
    feed = read_cloud_runs(tmp_path)
    assert not feed.runs and "50-run limit" in feed.errors[0].message
    missing = read_cloud_runs(tmp_path / "missing")
    assert missing.enabled and missing.errors and not missing.runs


def test_entry_limit_invalid_filename_and_duplicate_json_are_rejected(tmp_path):
    write(tmp_path, name="invalid id.json")
    feed = read_cloud_runs(tmp_path)
    assert feed.errors[0].run_id is None and "invalid run ID" in feed.errors[0].message
    (tmp_path / "test-run.json").write_text('{"schema_version":1,"schema_version":1}')
    assert len(read_cloud_runs(tmp_path).errors) == 2
    for index in range(200):
        (tmp_path / f".{index}.tmp").touch()
    feed = read_cloud_runs(tmp_path)
    assert not feed.runs and "entry limit" in feed.errors[0].message


def test_json_escaping_at_log_limit_is_supported(tmp_path):
    value = "\x00" * MAX_LOG_BYTES
    path = write(tmp_path, snapshot(logs={"isaac": value, "vla": value}))
    assert 128 * 1024 < path.stat().st_size < MAX_SNAPSHOT_BYTES
    assert read_cloud_runs(tmp_path).runs[0].logs.isaac == value


def test_symlink_snapshot_and_directory_are_refused(tmp_path):
    target = tmp_path / "real"
    target.mkdir()
    snapshot_path = write(target)
    links = tmp_path / "links"
    links.mkdir()
    try:
        (links / "test-run.json").symlink_to(snapshot_path)
        (tmp_path / "linked-root").symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks requires platform permission")
    assert not read_cloud_runs(links).runs
    assert "symlinks" in read_cloud_runs(links).errors[0].message
    assert not read_cloud_runs(tmp_path / "linked-root").runs
    assert "symlinks" in read_cloud_runs(tmp_path / "linked-root").errors[0].message


def test_settings_environment_does_not_resolve_away_a_symlink(tmp_path, monkeypatch):
    path = tmp_path / "feed"
    monkeypatch.setenv("FIREBIRD_CLOUD_RUNS_DIR", str(path))
    assert Settings.from_env().cloud_runs_dir == path
    monkeypatch.delenv("FIREBIRD_CLOUD_RUNS_DIR")
    assert Settings.from_env().cloud_runs_dir is None


def test_atomic_replacement_during_open_is_retried_without_a_false_feed_error(
    tmp_path, monkeypatch
):
    target = write(tmp_path, snapshot(status="RUNNING"))
    pending = write(tmp_path, snapshot(status="SUCCEEDED"), ".pending.tmp")
    original_open = os.open
    replaced = False

    def replace_then_open(path, flags, *args, **kwargs):
        nonlocal replaced
        if Path(path) == target and not replaced:
            os.replace(pending, target)
            replaced = True
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", replace_then_open)
    feed = read_cloud_runs(tmp_path)
    assert replaced and feed.errors == []
    assert feed.runs[0].status == "SUCCEEDED"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO check")
def test_nonregular_snapshot_is_rejected_without_blocking(tmp_path):
    os.mkfifo(tmp_path / "test-run.json")
    feed = read_cloud_runs(tmp_path)
    assert not feed.runs and "regular file" in feed.errors[0].message


def test_snapshot_encoding_must_be_utf8(tmp_path):
    (tmp_path / "test-run.json").write_bytes(json.dumps(snapshot()).encode("utf-16"))
    feed = read_cloud_runs(tmp_path)
    assert not feed.runs and "invalid versioned JSON" in feed.errors[0].message
