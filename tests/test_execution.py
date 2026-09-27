import asyncio
import sqlite3

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import insert
from vla_platform.api import create_app
from vla_platform.contracts import IntakeRequest, Job, now
from vla_platform.execution import Execution
from vla_platform.settings import Settings
from vla_platform.storage import Storage, jobs


def test_workspace_has_exactly_one_owner(tmp_path):
    settings = Settings(data_dir=tmp_path)
    with TestClient(create_app(settings)):
        with pytest.raises(RuntimeError, match="already owns"):
            with TestClient(create_app(settings)):
                pass
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/health").status_code == 200


def test_restart_reconciles_abandoned_job(tmp_path):
    async def seed():
        storage = Storage(tmp_path)
        await storage.initialize()
        record = Job(
            id="abandoned",
            project_id="p",
            status="running",
            request=IntakeRequest(repo_id="fixture/data"),
            created_at=now(),
            updated_at=now(),
        )
        async with storage.engine.begin() as connection:
            await connection.execute(
                insert(jobs).values(
                    id=record.id,
                    project_id=record.project_id,
                    status=record.status,
                    record=record.model_dump(),
                )
            )
        await storage.close()

    asyncio.run(seed())
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        record = client.get("/api/v1/jobs/abandoned").json()
        assert record["status"] == "interrupted"
        assert record["result"] is None


def test_cancellation_cannot_publish_late_success(tmp_path):
    async def exercise():
        storage = Storage(tmp_path)
        await storage.initialize()
        execution = Execution(storage, Settings(data_dir=tmp_path))
        # Hold all slots so cancellation deterministically exercises a queued worker.
        execution.slots = asyncio.Semaphore(0)
        record = await execution.submit("p", IntakeRequest(repo_id="fixture/data"))
        result = await execution.cancel(record.id)
        assert result.status == "cancelled"
        assert not (tmp_path / "jobs" / record.id / "result.json").exists()
        await execution.close()
        assert (await execution.get(record.id)).status == "cancelled"
        await storage.close()

    asyncio.run(exercise())


def test_initial_migration_is_versioned(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path))):
        with sqlite3.connect(tmp_path / "workspace.sqlite3") as database:
            assert database.execute("select version_num from alembic_version").fetchone() == (
                "0002",
            )


def test_startup_failure_preserves_error_and_releases_owner(tmp_path, monkeypatch):
    original = Storage.initialize

    async def fail(_):
        raise RuntimeError("original migration failure")

    monkeypatch.setattr(Storage, "initialize", fail)
    with pytest.raises(RuntimeError, match="original migration failure"):
        with TestClient(create_app(Settings(data_dir=tmp_path))):
            pass
    monkeypatch.setattr(Storage, "initialize", original)
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        assert client.get("/api/v1/health").status_code == 200
