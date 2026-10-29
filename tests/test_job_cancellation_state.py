"""Cancellation records follow owned task outcomes, independent of error wording."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from sqlalchemy import insert
from vla_platform.compute.contracts import ExecutionBinding
from vla_platform.contracts import IntakeRequest, Job, JobCancellation, WorkerResult, now
from vla_platform.execution import Execution
from vla_platform.lifecycle.contracts import PolicyRequest
from vla_platform.scenes.contracts import SceneRequest
from vla_platform.settings import Settings
from vla_platform.storage import Storage, jobs
from vla_platform.workspace_jobs import core_job


@asynccontextmanager
async def workspace(tmp_path):
    storage = Storage(tmp_path)
    await storage.initialize()
    execution = Execution(storage, Settings(data_dir=tmp_path))
    try:
        yield execution
    finally:
        await execution.close()
        await storage.close()


async def seed(execution, ident="job-1", request=None, **updates):
    request = request or IntakeRequest(repo_id="fixture/data")
    record = Job(
        id=ident,
        project_id="project",
        kind=getattr(request, "operation", "dataset.inspect"),
        status="running",
        request=request,
        created_at=now(),
        updated_at=now(),
        **updates,
    )
    async with execution.storage.engine.begin() as connection:
        await connection.execute(
            insert(jobs).values(
                id=record.id,
                project_id=record.project_id,
                status=record.status,
                record=record.model_dump(),
            )
        )
    return record


@pytest.mark.parametrize(
    "message", ["Termination acknowledgement was lost", "The adapter report said cleanup succeeded"]
)
def test_exception_from_cancelled_task_never_claims_success(tmp_path, message):
    async def exercise():
        async with workspace(tmp_path) as execution:
            job = await seed(execution)
            started = asyncio.Event()

            async def owned():
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    raise RuntimeError(message) from None

            execution.tasks[job.id] = asyncio.create_task(owned())
            await started.wait()
            cancelled = await execution.cancel(job.id)
            assert cancelled.status == "cancelled"
            assert cancelled.cancellation.cleanup == "failed"
            assert cancelled.cancellation.finished_at is not None
            assert "could not be confirmed" in cancelled.error
            assert core_job(cancelled).status == "waiting"
            failure = execution.observability.get_failure(job.id)
            assert failure.code == "CANCELLATION_UNCONFIRMED" and failure.cleanup == "failed"

    asyncio.run(exercise())


def test_normal_owned_task_cancellation_confirms_cleanup(tmp_path):
    async def exercise():
        async with workspace(tmp_path) as execution:
            job = await seed(execution)
            started, cleaned = asyncio.Event(), asyncio.Event()

            async def owned():
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cleaned.set()

            execution.tasks[job.id] = asyncio.create_task(owned())
            await started.wait()
            cancelled = await execution.cancel(job.id)
            assert cleaned.is_set() and cancelled.cancellation.cleanup == "succeeded"
            assert core_job(cancelled).status == "cancelled"

    asyncio.run(exercise())


@pytest.mark.parametrize("family", ["policy", "scene"])
def test_operation_handler_preserves_exception_after_cancellation(tmp_path, monkeypatch, family):
    async def exercise():
        async with workspace(tmp_path) as execution:
            started = asyncio.Event()

            async def worker(*args, **kwargs):
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    raise RuntimeError("Termination acknowledgement unavailable") from None

            if family == "policy":
                request = PolicyRequest(
                    operation="policy.finetune",
                    runtime_id="fixture",
                    dataset_job_id="source",
                    training={"steps": 1},
                )
                monkeypatch.setattr(execution.lifecycle, "run", worker)
                job = await seed(execution, request=request)
                task = asyncio.create_task(execution.run_policy(job))
            else:
                await seed(execution, ident="source")
                request = SceneRequest(request_id="a" * 32, source_job_id="source")
                job = await seed(execution, request=request)
                execution.scene_explorer = object()
                monkeypatch.setattr(execution.scenes, "run_job", worker)
                task = asyncio.create_task(execution.run_scene(job))
            execution.tasks[job.id] = task
            await started.wait()
            cancelled = await execution.cancel(job.id)
            assert cancelled.status == "cancelled" and cancelled.cancellation.cleanup == "failed"
            assert core_job(cancelled).status == "waiting"
            assert "Termination acknowledgement unavailable" in cancelled.error

    asyncio.run(exercise())


def test_metadata_error_after_cancel_marks_cleanup_unconfirmed(tmp_path):
    async def exercise():
        async with workspace(tmp_path) as execution:
            job = await seed(execution)
            job.status, job.cancellation = "cancelled", JobCancellation(requested_at=now())
            await execution.save(job)
            await execution.finish(
                job.id, WorkerResult(error="Worker termination was not acknowledged")
            )
            saved = await execution.get(job.id)
            assert saved.cancellation.cleanup == "failed"
            assert core_job(saved).status == "waiting"

    asyncio.run(exercise())


def test_explicit_cleanup_failure_is_not_overwritten_by_task_return(tmp_path):
    async def exercise():
        async with workspace(tmp_path) as execution:
            job = await seed(execution)
            started = asyncio.Event()

            async def owned():
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    async with execution.lock:
                        current = await execution.get(job.id)
                        current.cancellation.cleanup = "failed"
                        current.error = "Execution stopped; resource release remains unverified."
                        await execution.save(current)

            execution.tasks[job.id] = asyncio.create_task(owned())
            await started.wait()
            cancelled = await execution.cancel(job.id)
            assert cancelled.cancellation.cleanup == "failed"
            assert "unverified" in cancelled.error
            assert core_job(cancelled).status == "waiting"

    asyncio.run(exercise())


@pytest.mark.parametrize("initial", ["pending", "failed"])
def test_local_restart_never_confirms_unverified_cleanup(tmp_path, monkeypatch, initial):
    async def exercise():
        async with workspace(tmp_path) as execution:

            async def recover(data_dir):
                return []

            async def simulations():
                return None

            monkeypatch.setattr("vla_platform.lifecycle.sky_runner.recover", recover)
            monkeypatch.setattr(execution, "reconcile_simulations", simulations)
            job = await seed(execution)
            job.status = "cancelled"
            job.cancellation = JobCancellation(requested_at=now(), cleanup=initial)
            job.error = "Execution cleanup remains unverified."
            await execution.save(job)
            await execution.reconcile()
            saved = await execution.get(job.id)
            assert saved.cancellation.cleanup == "failed"
            assert core_job(saved).status == "waiting"

    asyncio.run(exercise())


@pytest.mark.parametrize("recovery_failed", [False, True])
def test_remote_cleanup_failure_can_be_reconciled_after_restart(
    tmp_path, monkeypatch, recovery_failed
):
    async def exercise():
        async with workspace(tmp_path) as execution:

            async def recover(data_dir):
                return (
                    [{"job_id": "job-1", "error": "Cloud cleanup needs attention"}]
                    if recovery_failed
                    else []
                )

            async def simulations():
                return None

            monkeypatch.setattr("vla_platform.lifecycle.sky_runner.recover", recover)
            monkeypatch.setattr(execution, "reconcile_simulations", simulations)
            job = await seed(execution)
            job.status = "cancelled"
            job.execution = ExecutionBinding(
                adapter="skypilot",
                infrastructure="gcp",
                target_id="fixture",
                label="Configured compute",
            )
            job.cancellation = JobCancellation(requested_at=now(), cleanup="failed")
            job.error = "Execution cleanup remains unverified."
            await execution.save(job)
            await execution.reconcile()
            saved = await execution.get(job.id)
            assert saved.cancellation.cleanup == ("failed" if recovery_failed else "succeeded")
            assert core_job(saved).status == ("waiting" if recovery_failed else "cancelled")

    asyncio.run(exercise())


def test_decision_supervisor_exception_after_cancellation_remains_unconfirmed(
    tmp_path, monkeypatch
):
    import threading

    from vla_platform import decision_jobs
    from vla_platform.decision_contracts import DecisionRequest

    async def exercise():
        async with workspace(tmp_path) as execution:
            request = DecisionRequest(
                schema_version=1,
                state="A fixture state",
                instructions="Compare criteria",
                criteria=[
                    {"id": "one", "text": "First criterion"},
                    {"id": "two", "text": "Second criterion"},
                ],
            )
            job = await seed(execution, request=request)
            started = asyncio.Event()

            async def supervised(payload, paths, connection):
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    raise RuntimeError("Scorer termination acknowledgement was lost") from None

            lock = threading.Lock()
            monkeypatch.setattr(decision_jobs, "configuration", lambda: ())
            monkeypatch.setattr(decision_jobs, "supervised", supervised)
            monkeypatch.setattr(decision_jobs, "_busy", lock)
            execution.tasks[job.id] = asyncio.create_task(decision_jobs.run(execution, job))
            await started.wait()
            cancelled = await execution.cancel(job.id)
            assert cancelled.cancellation.cleanup == "failed"
            assert core_job(cancelled).status == "waiting"
            assert "Scorer termination acknowledgement was lost" in cancelled.error
            assert not lock.locked()
            failure = execution.observability.get_failure(job.id)
            assert failure.code == "CANCELLATION_UNCONFIRMED" and failure.cleanup == "failed"
            assert any(
                row.severity == "error" for row in execution.observability.page(job.id).records
            )

    asyncio.run(exercise())
