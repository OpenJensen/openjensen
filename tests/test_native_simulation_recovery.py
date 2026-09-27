"""Restart reconciliation uses saved identities; no cloud or model execution."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import insert
from vla_platform.contracts import Job, now
from vla_platform.execution import Execution
from vla_platform.lifecycle import simulation, sky_runner
from vla_platform.lifecycle.contracts import (
    LifecycleResult,
    PolicyArtifact,
    PolicyRequest,
    SimulationTarget,
)
from vla_platform.settings import Settings
from vla_platform.storage import Storage, jobs


@asynccontextmanager
async def workspace(tmp_path, monkeypatch):
    import sys

    import vla_platform.lifecycle

    storage = Storage(tmp_path)
    await storage.initialize()
    execution = Execution(storage, Settings(data_dir=tmp_path))
    calls = []
    profile = SimpleNamespace(id="cup", identity_hash=lambda: "a" * 64)

    async def recover(profile, directory):
        calls.append(directory)
        return {
            "status": "cancellation_requested",
            "resource_deletion": "unverified",
            "resubmitted": False,
        }

    runner = SimpleNamespace(recover=recover)
    monkeypatch.setitem(sys.modules, "vla_platform.lifecycle.isaac_runner", runner)
    monkeypatch.setattr(vla_platform.lifecycle, "isaac_runner", runner, raising=False)
    monkeypatch.setattr(simulation, "profiles", lambda _: (profile,))

    async def sky_recover(_):
        return []

    monkeypatch.setattr(sky_runner, "recover", sky_recover)
    try:
        yield execution, profile, runner, calls
    finally:
        await storage.close()


async def saved(execution, *, status="running", target=True, directory=True, ident=None):
    stamp = now()
    ident = ident or str(uuid4())
    request = PolicyRequest(
        operation="policy.run" if target else "policy.import",
        runtime_id="cup",
        artifact_id="source" if target else None,
        source_id=None if target else "f" * 32,
        simulation={"profile_id": "cup", "experimental": target},
    )
    job = Job(
        id=ident,
        project_id="p",
        kind=request.operation,
        status=status,
        request=request,
        created_at=stamp,
        updated_at=stamp,
        error="Earlier evidence",
        simulation_target=SimulationTarget(
            profile_id="cup",
            profile_sha256="a" * 64,
            source_manifest_sha256="b" * 64,
            model_id="sha256:" + "c" * 64,
        )
        if target
        else None,
    )
    async with execution.storage.engine.begin() as connection:
        await connection.execute(
            insert(jobs).values(id=job.id, project_id="p", status=status, record=job.model_dump())
        )
    if directory:
        root = execution.settings.data_dir / "jobs" / job.id / "simulation" / "execution"
        root.mkdir(parents=True)
        (root / "request.json").write_text(
            json.dumps(
                {"profile_id": "cup", "profile_sha256": "a" * 64, "model_id": "sha256:" + "c" * 64}
            )
        )
    return job


def test_interrupted_owned_job_cancels_without_resubmit_or_late_adoption(tmp_path, monkeypatch):
    async def exercise():
        async with workspace(tmp_path, monkeypatch) as (execution, _, _, calls):
            original = await saved(execution)
            late = tmp_path / "jobs" / original.id / "simulation" / "execution" / "report.json"
            late.write_text('{"execution_status":"succeeded","task_success":true}')
            await execution.reconcile()
            job = await execution.get(original.id)
            assert job.status == "interrupted" and job.result is None
            assert job.error.startswith("Earlier evidence")
            assert "cancellation_requested" in job.error and "unverified" in job.error
            assert len(calls) == 1 and not execution.tasks
            assert late.exists()
            events = execution.lifecycle.events(original.id)
            assert any(e.stage == "simulation_recovery" for e in events)

    asyncio.run(exercise())


@pytest.mark.parametrize("status", ["failed", "cancelled", "interrupted"])
def test_terminal_status_is_retained(tmp_path, monkeypatch, status):
    async def exercise():
        async with workspace(tmp_path, monkeypatch) as (execution, _, _, calls):
            original = await saved(execution, status=status)
            await execution.reconcile()
            assert (await execution.get(original.id)).status == status
            assert len(calls) == 1

    asyncio.run(exercise())


def test_import_and_unowned_directories_never_recover(tmp_path, monkeypatch):
    async def exercise():
        async with workspace(tmp_path, monkeypatch) as (execution, _, _, calls):
            await saved(execution, target=False)
            await saved(execution, directory=False)
            (tmp_path / "jobs" / str(uuid4()) / "simulation" / "execution").mkdir(parents=True)
            await execution.reconcile()
            assert calls == []

    asyncio.run(exercise())


def test_successful_published_simulation_record_is_not_cancelled(tmp_path, monkeypatch):
    async def exercise():
        async with workspace(tmp_path, monkeypatch) as (execution, _, _, calls):
            job = await saved(execution, status="succeeded")
            job.result = LifecycleResult(
                decision="diagnostics_only",
                reports=[{"stage": "simulation", "execution_status": "succeeded"}],
                artifacts=[
                    PolicyArtifact(
                        id=job.id + ":simulation",
                        project_id="p",
                        job_id=job.id,
                        label="Cup report",
                        format="simulation_record",
                        path="jobs/" + job.id + "/simulation/record",
                        manifest_sha256="d" * 64,
                        file_bytes=1,
                        metadata={"model_id": "sha256:" + "c" * 64},
                    )
                ],
            )
            await execution.save(job)
            before = job.model_dump()
            await execution.reconcile()
            assert calls == []
            assert (await execution.get(job.id)).model_dump() == before

    asyncio.run(exercise())


@pytest.mark.parametrize("problem", ["missing", "changed", "model_mismatch", "symlink"])
def test_unverifiable_target_requires_operator_without_cloud(tmp_path, monkeypatch, problem):
    async def exercise():
        async with workspace(tmp_path, monkeypatch) as (execution, profile, _, calls):
            job = await saved(execution, status="failed")
            root = tmp_path / "jobs" / job.id / "simulation" / "execution"
            if problem == "missing":
                monkeypatch.setattr(simulation, "profiles", lambda _: ())
            elif problem == "changed":
                profile.identity_hash = lambda: "e" * 64
            elif problem == "model_mismatch":
                (root / "request.json").write_text(
                    json.dumps(
                        {
                            "profile_id": "cup",
                            "profile_sha256": "a" * 64,
                            "model_id": "sha256:" + "e" * 64,
                        }
                    )
                )
            else:
                renamed = root.with_name("preserved")
                root.rename(renamed)
                root.symlink_to(renamed, target_is_directory=True)
            await execution.reconcile()
            assert calls == []
            assert "cleanup_unknown" in (await execution.get(job.id)).error

    asyncio.run(exercise())


def test_runner_failure_is_redacted_and_other_jobs_still_recover(tmp_path, monkeypatch):
    async def exercise():
        async with workspace(tmp_path, monkeypatch) as (execution, _, runner, calls):
            failed = await saved(execution, status="failed")
            good = await saved(execution, status="failed")

            async def recover(profile, directory):
                calls.append(directory)
                if failed.id in str(directory):
                    raise RuntimeError("SECRET CREDENTIAL CONTENT")
                return {
                    "status": "cancellation_requested",
                    "resource_deletion": "unverified",
                    "resubmitted": False,
                }

            runner.recover = recover
            await execution.reconcile()
            assert len(calls) == 2
            assert "cleanup_unknown" in (await execution.get(failed.id)).error
            assert "SECRET" not in (await execution.get(failed.id)).error
            assert "cancellation_requested" in (await execution.get(good.id)).error

    asyncio.run(exercise())


def test_timeout_is_bounded_and_records_unknown(tmp_path, monkeypatch):
    async def exercise():
        async with workspace(tmp_path, monkeypatch) as (execution, _, runner, _):
            job = await saved(execution, status="failed")
            cancelled = asyncio.Event()

            async def recover(*_):
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

            runner.recover = recover
            monkeypatch.setattr("vla_platform.execution.SIMULATION_RECOVERY_SECONDS", 0.01)
            await asyncio.wait_for(execution.reconcile(), 1)
            assert cancelled.is_set()
            assert "cleanup_unknown" in (await execution.get(job.id)).error

    asyncio.run(exercise())


def test_event_failure_does_not_block_startup_or_other_jobs(tmp_path, monkeypatch):
    async def exercise():
        async with workspace(tmp_path, monkeypatch) as (execution, _, _, calls):
            job = await saved(execution, status="failed")

            async def fail(*_, **__):
                raise OSError("disk unavailable")

            monkeypatch.setattr(execution.lifecycle, "event", fail)
            await execution.reconcile()
            assert len(calls) == 1
            assert "cancellation_requested" in (await execution.get(job.id)).error

    asyncio.run(exercise())


def test_global_budget_marks_remaining_jobs_unknown_without_attempting(tmp_path, monkeypatch):
    async def exercise():
        async with workspace(tmp_path, monkeypatch) as (execution, _, _, calls):
            job = await saved(execution, status="failed")
            monkeypatch.setattr("vla_platform.execution.SIMULATION_RECOVERY_BUDGET_SECONDS", 0)
            await execution.reconcile()
            assert calls == []
            current = await execution.get(job.id)
            assert current.status == "failed" and "cleanup_unknown" in current.error
            event = execution.lifecycle.events(job.id)[-1]
            assert event.data["reason"] == "startup_recovery_budget_exhausted"

    asyncio.run(exercise())


def test_noncanonical_job_identity_never_escapes_workspace(tmp_path, monkeypatch):
    async def exercise():
        async with workspace(tmp_path, monkeypatch) as (execution, _, _, calls):
            job = await saved(execution, status="failed", ident="../escaped", directory=False)
            await execution.reconcile()
            assert calls == []
            assert "cleanup_unknown" in (await execution.get(job.id)).error
            assert not (tmp_path / "escaped").exists()

    asyncio.run(exercise())


def test_same_process_does_not_repeat_acknowledged_cancellation(tmp_path, monkeypatch):
    async def exercise():
        async with workspace(tmp_path, monkeypatch) as (execution, _, _, calls):
            job = await saved(execution, status="failed")
            await execution.reconcile()
            before = (await execution.get(job.id)).model_dump()
            await execution.reconcile()
            assert len(calls) == 1
            assert (await execution.get(job.id)).model_dump() == before

    asyncio.run(exercise())


def test_recovery_rejects_resource_deletion_claim(tmp_path, monkeypatch):
    async def exercise():
        async with workspace(tmp_path, monkeypatch) as (execution, _, runner, _):
            job = await saved(execution, status="failed")

            async def wrong(*_):
                return {
                    "status": "cancellation_requested",
                    "resubmitted": False,
                    "resource_deletion": "verified",
                }

            runner.recover = wrong
            await execution.reconcile()
            assert "cleanup_unknown" in (await execution.get(job.id)).error

    asyncio.run(exercise())
