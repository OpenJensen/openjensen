"""Public naming migration must preserve alpha workspace configuration and data."""

import asyncio
import shutil
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from vla_platform.api import create_app
from vla_platform.environment import getenv
from vla_platform.lifecycle.service import Lifecycle
from vla_platform.settings import Settings


def test_canonical_environment_wins_without_mutating_alpha_config(monkeypatch):
    monkeypatch.setenv("FIREBIRD_API_URL", "http://localhost:8123")
    monkeypatch.delenv("JENSEN_API_URL", raising=False)
    assert getenv("JENSEN_API_URL") == "http://localhost:8123"
    monkeypatch.setenv("JENSEN_API_URL", "http://localhost:8234")
    assert getenv("FIREBIRD_API_URL") == "http://localhost:8234"
    monkeypatch.setenv("JENSEN_API_URL", "")
    assert getenv("JENSEN_API_URL", "fallback") == ""


def test_default_workspace_keeps_existing_alpha_data(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in ("JENSEN_DATA_DIR", "FIREBIRD_DATA_DIR"):
        monkeypatch.delenv(name, raising=False)
    assert Settings.from_env().data_dir == tmp_path / ".jensen"
    (tmp_path / ".firebird").mkdir()
    assert Settings.from_env().data_dir == tmp_path / ".firebird"
    (tmp_path / ".jensen").mkdir()
    assert Settings.from_env().data_dir == tmp_path / ".jensen"
    monkeypatch.setenv("FIREBIRD_DATA_DIR", str(tmp_path / "explicit-alpha"))
    assert Settings.from_env().data_dir == tmp_path / "explicit-alpha"
    monkeypatch.setenv("JENSEN_DATA_DIR", str(tmp_path / "explicit-current"))
    assert Settings.from_env().data_dir == tmp_path / "explicit-current"


def test_stopped_workspace_backup_restore_keeps_project_and_upload_bytes(tmp_path):
    source = tmp_path / ".firebird"
    with TestClient(create_app(Settings(data_dir=source))) as client:
        project = client.post("/api/v1/projects", json={"name": "Alpha workspace"}).json()
        prefix = f"/api/v1/projects/{project['id']}"
        upload = client.post(prefix + "/dataset-uploads", json={"name": "Kept source"}).json()
        response = client.put(
            f"/api/v1/dataset-uploads/{upload['id']}/file?path=data.txt",
            content=b"Original robot data fixture",
        )
        assert response.status_code == 200
    backup = tmp_path / "backup"
    shutil.copytree(source, backup)
    restored = tmp_path / ".jensen"
    shutil.copytree(backup, restored)
    with TestClient(create_app(Settings(data_dir=restored))) as client:
        assert client.get("/api/v1/projects").json() == [project]
        response = client.get("/api/v1/datasets", params={"project_id": project["id"]})
        assert response.status_code == 200
        records = response.json()
        assert (
            next(record for record in records if record["id"] == upload["id"])["status"]
            == "uploading"
        )
    for root in (source, backup, restored):
        assert (
            root / "dataset-library" / upload["id"] / "source/data.txt"
        ).read_bytes() == b"Original robot data fixture"


@pytest.mark.parametrize("failure", ["exit", "spawn", "timeout"])
def test_docker_cleanup_failure_is_visible_after_native_process_cleanup(monkeypatch, failure):
    process = SimpleNamespace(pid=999999, returncode=0, wait=AsyncMock(return_value=0))
    removal = SimpleNamespace(wait=AsyncMock(return_value=1), kill=lambda: None)
    if failure == "spawn":
        spawn = AsyncMock(side_effect=OSError("Docker unavailable"))
    else:
        spawn = AsyncMock(return_value=removal)
        if failure == "timeout":
            removal.wait = AsyncMock(side_effect=[TimeoutError, 0])
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    import vla_platform.lifecycle.service as service

    if hasattr(service.os, "killpg"):
        monkeypatch.setattr(service.os, "killpg", lambda *_args: None)
    with pytest.raises(
        RuntimeError, match="Worker cleanup needs attention.*Resource deletion is not confirmed"
    ):
        asyncio.run(Lifecycle.__new__(Lifecycle).stop(process, "jensen-owned-fixture"))
    assert process.wait.await_count == 1
