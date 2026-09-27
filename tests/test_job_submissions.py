"""Durable submission identity: real disposable SQLite, no model/provider work."""

import asyncio
import sqlite3
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, update
from vla_platform.api import create_app
from vla_platform.contracts import IntakeRequest
from vla_platform.execution import Execution
from vla_platform.lifecycle.contracts import PolicyRequest
from vla_platform.settings import Settings
from vla_platform.storage import Storage, job_submissions, jobs
from vla_platform.submissions import SubmissionConflict, SubmissionUnavailable, lookup


@asynccontextmanager
async def owner(tmp_path):
    storage = Storage(tmp_path)
    await storage.initialize()
    execution = Execution(storage, Settings(data_dir=tmp_path, local_root=tmp_path))
    launched = []

    async def record(job_id):
        launched.append(job_id)

    execution.run = record
    try:
        yield execution, launched
    finally:
        await execution.close()
        await storage.close()


def request(path="generated"):
    return IntakeRequest(source="local", path=path)


def test_concurrent_key_accepts_one_job_and_launch(tmp_path):
    async def exercise():
        storage = Storage(tmp_path)
        await storage.initialize()
        execution = Execution(storage, Settings(data_dir=tmp_path, local_root=tmp_path))
        launched = []

        async def record(job_id):
            launched.append(job_id)

        execution.run = record
        try:
            records = await asyncio.gather(
                *[
                    execution.submit(
                        "project",
                        IntakeRequest(source="local", path="generated"),
                        idempotency_key="same-request",
                    )
                    for _ in range(12)
                ]
            )
            await asyncio.sleep(0)
            assert len({job.id for job in records}) == 1
            assert launched == [records[0].id]
            assert len(await execution.list("project")) == 1
        finally:
            await execution.close()
            await storage.close()

    asyncio.run(exercise())


def test_retry_checks_original_before_any_source_resolution(tmp_path):
    async def exercise():
        async with owner(tmp_path) as (execution, launched):
            first = await execution.submit("p", request(), idempotency_key="key")

            async def unavailable(_):
                pytest.fail("An accepted retry must not resolve a changed or unavailable source")

            execution.inspections.prepare = unavailable
            execution.settings = replace(execution.settings, local_root=None)
            again = await execution.submit("p", request(), idempotency_key="key")
            assert again.id == first.id
            with pytest.raises(SubmissionConflict):
                await execution.submit("p", request("changed"), idempotency_key="key")
            assert launched == [first.id]

    asyncio.run(exercise())


def test_restart_returns_interrupted_job_without_dispatch(tmp_path):
    async def exercise():
        async with owner(tmp_path) as (execution, _):
            first = await execution.submit("p", request(), idempotency_key="key")
        async with owner(tmp_path) as (execution, launched):
            await execution.reconcile()
            retried = await execution.submit("p", request(), idempotency_key="key")
            assert retried.id == first.id and retried.status == "interrupted"
            assert launched == []

    asyncio.run(exercise())


def test_scope_separates_project_and_operation_and_preserves_original_recipe(tmp_path):
    async def exercise():
        async with owner(tmp_path) as (execution, launched):

            async def validate(_project, payload):
                payload.artifact_id = "server-resolved"

            execution.lifecycle.validate = validate
            first = await execution.submit("p", request(), idempotency_key="key")
            other = await execution.submit("other", request(), idempotency_key="key")
            recipe = PolicyRequest(
                operation="policy.run", runtime_id="fixture", artifact_id="original"
            )
            policy = await execution.submit("p", recipe, idempotency_key="key")
            assert recipe.artifact_id == "original"
            retried = await execution.submit("p", recipe, idempotency_key="key")
            assert retried.id == policy.id and retried.request.artifact_id == "server-resolved"
            assert len({first.id, other.id, policy.id}) == 3
            assert len(launched) == 3

    asyncio.run(exercise())


def test_different_keys_reuse_existing_metadata_cache(tmp_path):
    async def exercise():
        async with owner(tmp_path) as (execution, launched):

            async def prepare(value):
                return value.model_copy(
                    update={"path": str(tmp_path / "data"), "revision": "fixed"}
                )

            execution.inspections.prepare = prepare
            first = await execution.inspections.submit("p", request())
            first.status = "running"
            await execution.save(first)
            records = await asyncio.gather(
                *[
                    execution.inspections.submit("p", request(), idempotency_key=f"key-{i}")
                    for i in range(8)
                ]
            )
            assert {item.id for item in records} == {first.id}
            assert launched == [first.id]
            async with execution.storage.engine.connect() as conn:
                rows = (await conn.execute(select(job_submissions))).mappings().all()
            assert len(rows) == 8
            assert all(row["accepted_response"]["status"] == "running" for row in rows)

    asyncio.run(exercise())


def test_transaction_failure_rolls_back_both_job_and_key(tmp_path):
    async def exercise():
        async with owner(tmp_path) as (execution, launched):
            async with execution.storage.engine.begin() as conn:
                await conn.exec_driver_sql(
                    "CREATE TRIGGER refuse_submission BEFORE INSERT ON job_submissions "
                    "BEGIN SELECT RAISE(ABORT, 'fixture disk failure'); END"
                )
            with pytest.raises(Exception, match="fixture disk failure"):
                await execution.submit("p", request(), idempotency_key="key")
            assert await execution.list("p") == [] and launched == []
            assert await lookup(execution.storage, "p", "dataset.inspect", "key") is None

    asyncio.run(exercise())


def test_failure_after_commit_never_schedules_on_retry(tmp_path):
    async def exercise():
        async with owner(tmp_path) as (execution, launched):

            def fail(_):
                raise RuntimeError("fixture failure after commit")

            execution._schedule = fail
            with pytest.raises(RuntimeError, match="after commit"):
                await execution.submit("p", request(), idempotency_key="key")
            accepted = await lookup(execution.storage, "p", "dataset.inspect", "key")
            assert accepted is not None
            retry = await execution.submit("p", request(), idempotency_key="key")
            assert retry.id == accepted.id and launched == []

    asyncio.run(exercise())


def test_cancellation_drains_acceptance_even_with_repeated_cancel(tmp_path):
    async def exercise():
        async with owner(tmp_path) as (execution, launched):
            entered, release = asyncio.Event(), asyncio.Event()
            original = execution._commit_and_schedule

            async def gated(*args, **kwargs):
                entered.set()
                await release.wait()
                return await original(*args, **kwargs)

            execution._commit_and_schedule = gated
            caller = asyncio.create_task(execution.submit("p", request(), idempotency_key="key"))
            await asyncio.wait_for(entered.wait(), 2)
            caller.cancel()
            await asyncio.sleep(0)
            caller.cancel()
            await asyncio.sleep(0)
            assert not caller.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(caller, 2)
            job = await execution.submit("p", request(), idempotency_key="key")
            assert launched == [job.id]

    asyncio.run(exercise())


def test_database_uniqueness_across_independent_submission_owners(tmp_path):
    async def exercise():
        async with owner(tmp_path) as (first, a):
            async with owner(tmp_path) as (second, b):
                records = await asyncio.gather(
                    *[
                        instance.submit("p", request(), idempotency_key="key")
                        for instance in [first, second]
                    ]
                )
                assert records[0].id == records[1].id
                assert len(a + b) == 1 and len(await first.list("p")) == 1

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "corruption", ["digest", "response", "version", "missing", "project", "request"]
)
def test_corrupt_saved_identity_fails_closed_without_new_job(tmp_path, corruption):
    async def exercise():
        async with owner(tmp_path) as (execution, launched):
            first = await execution.submit("p", request(), idempotency_key="key")
            async with execution.storage.engine.begin() as conn:
                if corruption == "project":
                    await conn.execute(update(jobs).values(project_id="foreign"))
                else:
                    updates = {
                        "digest": {"request_sha256": "0" * 64},
                        "response": {"accepted_response": {**first.model_dump(), "id": "foreign"}},
                        "version": {"fingerprint_version": 2},
                        "missing": {"job_id": "absent"},
                        "request": {"request_record": {}},
                    }[corruption]
                    await conn.execute(update(job_submissions).values(**updates))
            with pytest.raises(SubmissionUnavailable):
                await execution.submit("p", request(), idempotency_key="key")
            assert launched == [first.id]

    asyncio.run(exercise())


@pytest.mark.parametrize("key", ["", " spaced", "x y", "a" * 129, "a\n", "/bad", "é", True, 123])
def test_bad_key_never_accepts_or_dispatches(tmp_path, key):
    async def exercise():
        async with owner(tmp_path) as (execution, launched):
            with pytest.raises(ValueError):
                await execution.submit("p", request(), idempotency_key=key)
            assert await execution.list("p") == [] and launched == []

    asyncio.run(exercise())


def test_api_key_lookup_conflict_and_restart(tmp_path, monkeypatch):
    launched = []

    async def record(_self, job_id):
        launched.append(job_id)

    monkeypatch.setattr(Execution, "run", record)
    settings = Settings(data_dir=tmp_path, local_root=tmp_path)
    with TestClient(create_app(settings)) as client:
        project = client.post("/api/v1/projects", json={"name": "Generated"}).json()["id"]
        url = f"/api/v1/projects/{project}/intakes"
        headers = {"Idempotency-Key": "key"}
        first = client.post(url, json={"source": "local", "path": "generated"}, headers=headers)
        assert first.status_code == 202 and first.headers["idempotency-key"] == "key"
        # Explicit/default fields normalize to the same request.
        retry = client.post(url, json=request().model_dump(), headers=headers)
        assert retry.status_code == 202 and retry.json()["id"] == first.json()["id"]
        assert (
            client.post(url, json=request("changed").model_dump(), headers=headers).status_code
            == 409
        )
        assert (
            client.post(
                url,
                json=request().model_dump(),
                headers=[("Idempotency-Key", "key"), ("Idempotency-Key", "other")],
            ).status_code
            == 422
        )
        lookup_url = f"/api/v1/projects/{project}/submissions/key?operation=dataset.inspect"
        saved = client.get(lookup_url)
        assert saved.status_code == 200 and saved.headers["cache-control"] == "no-store"
        assert (
            client.get(lookup_url.replace("submissions/key", "submissions/missing")).status_code
            == 404
        )
        assert client.get(lookup_url.replace(project, "foreign")).status_code == 404
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        retry = client.post(url, json=request().model_dump(), headers=headers)
        assert retry.status_code == 202 and retry.json()["status"] == "interrupted"
        assert retry.json()["id"] == first.json()["id"]
    assert launched == [first.json()["id"]]


def test_existing_0001_job_survives_additive_migration(tmp_path):
    from pathlib import Path

    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine
    from vla_platform import storage as module
    from vla_platform.contracts import Job, now

    engine = create_engine(f"sqlite:///{tmp_path / 'workspace.sqlite3'}")
    config = Config()
    config.set_main_option("script_location", str(Path(module.__file__).parent / "migrations"))
    with engine.begin() as conn:
        config.attributes["connection"] = conn
        command.upgrade(config, "0001")
    engine.dispose()
    old = Job(id="original", project_id="p", request=request(), created_at=now(), updated_at=now())
    with sqlite3.connect(tmp_path / "workspace.sqlite3") as db:
        db.execute(
            "INSERT INTO jobs VALUES (?,?,?,?)",
            (old.id, old.project_id, old.status, old.model_dump_json()),
        )

    async def exercise():
        async with owner(tmp_path) as (execution, _):
            assert (await execution.get(old.id)).model_dump() == old.model_dump()

    asyncio.run(exercise())
    with sqlite3.connect(tmp_path / "workspace.sqlite3") as db:
        assert db.execute("SELECT version_num FROM alembic_version").fetchone() == ("0002",)
        assert db.execute("SELECT COUNT(*) FROM job_submissions").fetchone() == (0,)


def test_hub_branch_retry_preserves_first_resolution_without_network(tmp_path, monkeypatch):
    from vla_platform.datasets import cache

    calls = []

    async def resolve(value):
        calls.append(value.revision)
        return "a" * 40

    monkeypatch.setattr(cache, "resolve_hub_revision", resolve)

    async def exercise():
        async with owner(tmp_path) as (execution, launched):
            original = IntakeRequest(repo_id="generated/data")
            first = await execution.inspections.submit("p", original, idempotency_key="key")
            assert first.request.revision == "a" * 40 and original.revision == "main"

            async def fail(_):
                pytest.fail("A retry must not resolve a mutable branch again")

            monkeypatch.setattr(cache, "resolve_hub_revision", fail)
            retried = await execution.inspections.submit("p", original, idempotency_key="key")
            assert retried.id == first.id and calls == ["main"] and launched == [first.id]

    asyncio.run(exercise())


@pytest.mark.parametrize("status", ["failed", "cancelled", "interrupted", "succeeded"])
def test_terminal_simulation_receipt_retains_unknown_cleanup_without_resubmission(tmp_path, status):
    from vla_platform.lifecycle.contracts import SimulationTarget

    async def exercise():
        async with owner(tmp_path) as (execution, launched):
            admissions = []

            async def validate(*_):
                admissions.append(True)
                return SimulationTarget(
                    profile_id="fixture", profile_sha256="a" * 64, source_manifest_sha256="b" * 64
                )

            execution.lifecycle.validate = validate
            recipe = PolicyRequest(
                operation="policy.run",
                runtime_id="fixture",
                artifact_id="model",
                simulation={"profile_id": "fixture", "experimental": True},
            )
            first = await execution.submit("p", recipe, idempotency_key="key")
            first.status, first.error = status, "Owned cloud cleanup remains unknown"
            first.simulation_target.model_id = "sha256:" + "c" * 64
            await execution.save(first)
            again = await execution.submit("p", recipe, idempotency_key="key")
            assert again.id == first.id and again.status == status
            assert (
                again.error == first.error
                and again.simulation_target.model_id == first.simulation_target.model_id
            )
            assert admissions == [True] and launched == [first.id]

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "route,payload",
    [
        (
            "policy-jobs",
            {"operation": "policy.run", "runtime_id": "fixture", "artifact_id": "model"},
        ),
        (
            "augmentations",
            {
                "source_job_id": "dataset",
                "episode_indices": [0],
                "camera_key": "observation.images.front",
            },
        ),
    ],
)
def test_api_policy_and_augmentation_keys_are_durable(tmp_path, monkeypatch, route, payload):
    from vla_platform.augmentation.service import Augmentation
    from vla_platform.lifecycle.service import Lifecycle

    calls = []

    async def admit(*_):
        calls.append("admit")

    async def run(*_):
        calls.append("run")

    monkeypatch.setattr(Lifecycle, "validate", admit)
    monkeypatch.setattr(Augmentation, "validate", admit)
    monkeypatch.setattr(Execution, "run", run)
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        project = client.post("/api/v1/projects", json={"name": "Generated"}).json()["id"]
        url = f"/api/v1/projects/{project}/{route}"
        first = client.post(url, json=payload, headers={"Idempotency-Key": "key"})
        assert first.status_code == 202, first.text
        again = client.post(url, json=payload, headers={"Idempotency-Key": "key"})
        assert again.status_code == 202 and again.json()["id"] == first.json()["id"]
        operation = first.json()["kind"]
        saved = client.get(f"/api/v1/projects/{project}/submissions/key?operation={operation}")
        assert saved.status_code == 200 and saved.json()["id"] == first.json()["id"]
    assert calls == ["admit", "run"]


def test_closing_drains_inflight_acceptance_before_worker_shutdown(tmp_path):
    async def exercise():
        async with owner(tmp_path) as (execution, launched):
            entered, release = asyncio.Event(), asyncio.Event()
            original = execution._commit_and_schedule

            async def gated(*args, **kwargs):
                entered.set()
                await release.wait()
                return await original(*args, **kwargs)

            execution._commit_and_schedule = gated
            submit = asyncio.create_task(execution.submit("p", request(), idempotency_key="key"))
            await asyncio.wait_for(entered.wait(), 2)
            closing = asyncio.create_task(execution.close())
            await asyncio.sleep(0)
            assert not closing.done()
            release.set()
            job = await asyncio.wait_for(submit, 2)
            await asyncio.wait_for(closing, 2)
            current = await lookup(execution.storage, "p", "dataset.inspect", "key")
            assert current.id == job.id and current.status == "interrupted"
            assert not execution.tasks and not execution._accepting

    asyncio.run(exercise())


@pytest.mark.parametrize("column", ["request_record", "accepted_response", "job_record"])
def test_malformed_persisted_json_is_unavailable_not_a_new_submission(tmp_path, column):
    async def exercise():
        async with owner(tmp_path) as (execution, launched):
            first = await execution.submit("p", request(), idempotency_key="key")
            async with execution.storage.engine.begin() as conn:
                statement = (
                    "UPDATE jobs SET record = ?"
                    if column == "job_record"
                    else f"UPDATE job_submissions SET {column} = ?"
                )
                await conn.exec_driver_sql(statement, ("{broken",))
            with pytest.raises(SubmissionUnavailable):
                await execution.submit("p", request(), idempotency_key="key")
            assert launched == [first.id]
            # Restore the owned fixture so ordinary shutdown reconciliation can parse it.
            async with execution.storage.engine.begin() as conn:
                if column == "job_record":
                    await conn.execute(update(jobs).values(record=first.model_dump()))

    asyncio.run(exercise())


def test_api_corrupt_binding_returns_503_for_retry_and_lookup(tmp_path, monkeypatch):
    launched = []

    async def record(_self, job_id):
        launched.append(job_id)

    monkeypatch.setattr(Execution, "run", record)
    with TestClient(create_app(Settings(data_dir=tmp_path, local_root=tmp_path))) as client:
        project = client.post("/api/v1/projects", json={"name": "Generated"}).json()["id"]
        url = f"/api/v1/projects/{project}/intakes"
        headers = {"Idempotency-Key": "key"}
        first = client.post(url, json=request().model_dump(), headers=headers)
        assert first.status_code == 202
        with sqlite3.connect(tmp_path / "workspace.sqlite3") as db:
            db.execute("UPDATE job_submissions SET request_record = ?", ("{broken",))
        retry = client.post(url, json=request().model_dump(), headers=headers)
        saved = client.get(f"/api/v1/projects/{project}/submissions/key?operation=dataset.inspect")
        assert retry.status_code == saved.status_code == 503
        assert retry.json() == saved.json()
        assert "no new work" in retry.json()["detail"]
        assert len(client.get(f"/api/v1/projects/{project}/jobs").json()) == 1
    assert launched == [first.json()["id"]]
