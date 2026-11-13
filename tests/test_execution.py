import asyncio
import hashlib
import json
import sqlite3
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import insert
from vla_platform.api import create_app
from vla_platform.contracts import IntakeRequest, Job, now
from vla_platform.execution import Execution
from vla_platform.lifecycle import sky_runner
from vla_platform.lifecycle.contracts import CloudExecutionTarget, PolicyRequest
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
                "0003",
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


# --- A cloud machine outlives the application: restarts while one is in use ---

CLOUD_TARGET = {
    "project_id": "billing-project",
    "workspace": "fixture-workspace",
    "sky_api_endpoint": "http://127.0.0.1:46580",
    "region": "us-central1",
    "accelerator": "A100",
    "gpu_count": 1,
    "instance_type": "a2-highgpu-1g",
    "disk_size_gb": 200,
    "idle_minutes": 10,
}


class Machine:
    """What the application's cloud commands find: the machines of the seeded jobs.

    ``status`` is what the cluster answers for a job (101 while it runs, 0 once it has
    succeeded, 100 once it has failed). ``calls`` records every command per job.
    """

    def __init__(self):
        self.status = 101
        self.reachable = True
        self.calls = []
        self.following = asyncio.Event()
        self.ended = asyncio.Event()

    def end(self, status=0):
        self.status = status
        self.ended.set()

    def commands(self, job_id):
        return [name for owner, name in self.calls if owner == job_id]

    async def command(self, argv, directory, *, timeout, **kwargs):
        job_id = directory.parent.name
        if "cloud_storage.py" in argv[1]:
            # Checkpoints and the result reach the application through the bucket.
            self.calls.append((job_id, "sync"))
            if self.reachable and self.status != 101:
                self.publish_result(job_id, directory)
            return (0, "") if self.reachable else (1, "unreachable")
        name = next(
            (item for item in ("--follow", "--status") if item in argv),
            argv[1] if argv[1] in {"launch", "down", "cancel"} else " ".join(argv[1:3]),
        )
        self.calls.append((job_id, name))
        if not self.reachable:
            async with asyncio.timeout(timeout):
                await asyncio.Event().wait()
        if name == "api status":
            return 0, "[]"
        if name == "--status":
            return self.status, ""
        if name == "--follow":
            self.following.set()
            if self.status == 101:
                await kwargs["on_event"]('_telemetry:{"step": 7, "loss": 0.25}')
                await self.ended.wait()
        return 0, ""

    def publish_result(self, job_id, stage):
        if (stage / "result.json").exists():
            return
        result = {"schema_version": 1, "job_id": job_id, "report": {"scope": "fixture"}}
        if self.status:
            result["error"] = "CUDA out of memory on the cloud machine"
        else:
            bundle = stage / "bundle"
            bundle.mkdir()
            weights = b"fixture checkpoint, not model weights"
            (bundle / "checkpoint.bin").write_bytes(weights)
            (bundle / "manifest.json").write_text(
                json.dumps(
                    {
                        "files": {"checkpoint.bin": hashlib.sha256(weights).hexdigest()},
                        "metadata": {"fixture_only": True},
                    }
                )
            )
            result["artifact"] = {
                "path": str(bundle),
                "label": "Cloud fixture checkpoint",
                "format": "training_checkpoint",
            }
        (stage / "result.json").write_text(json.dumps(result))


@pytest.fixture
def machine(tmp_path, monkeypatch):
    worker = tmp_path / "worker-src"
    (worker / "src" / "jensen_vla").mkdir(parents=True)
    (worker / "src" / "jensen_vla" / "application.py").write_text("# fixture only\n")
    (worker / "pyproject.toml").write_text("[project]\nname = 'fixture'\nversion = '1'\n")
    (worker / "requirements-smolvla-linux.txt").write_text("setuptools==80.10.2\n")
    cloud = Machine()
    monkeypatch.setattr(sky_runner, "worker_root", lambda: worker)
    monkeypatch.setattr(sky_runner, "executable", lambda: "/fixture/sky")
    monkeypatch.setattr(sky_runner.cloud_compute_catalog, "sky_python", lambda _: "/fixture/python")
    monkeypatch.setattr(sky_runner, "_command", cloud.command)
    return cloud


async def killed_during(workspace, phase, job_id="cloud-training"):
    """Leave what a killed application leaves: a running job and its machine's record."""
    request = PolicyRequest.model_validate(
        {
            "operation": "policy.finetune",
            "runtime_id": "skypilot-gcp-A100",
            "dataset_job_id": "dataset",
            "training": {"steps": 1},
        }
    )
    stamp = now()
    record = Job(
        id=job_id,
        project_id="p",
        kind="policy.finetune",
        status="running",
        request=request,
        compute_target=CloudExecutionTarget.model_validate(CLOUD_TARGET),
        created_at=stamp,
        updated_at=stamp,
    )
    storage = Storage(workspace)
    await storage.initialize()
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
    stage = workspace / "jobs" / job_id / "operation"
    payload = {
        "schema_version": 1,
        "operation": "policy.finetune",
        "job_id": job_id,
        "parameters": request.model_dump(),
        "final": False,
        "artifact": None,
        "source": None,
    }
    target = {**CLOUD_TARGET, "storage_uri": f"gs://fixture-bucket/jobs/{job_id}/operation"}
    _, state = sky_runner.prepare(payload, stage, target)
    (stage / "request.json").write_text(json.dumps(payload))
    state.update(phase=phase, request_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    sky_runner._write_json(stage / sky_runner.STATE_NAME, state)
    return stage


@asynccontextmanager
async def started(workspace):
    """The application's job supervisor from startup recovery to shutdown."""
    storage = Storage(workspace)
    await storage.initialize()
    execution = Execution(storage, Settings(data_dir=workspace))
    try:
        await asyncio.wait_for(execution.reconcile(), 10)
        yield execution
    finally:
        await execution.close()
        await storage.close()


def dispatch_phase(stage):
    return json.loads((stage / sky_runner.STATE_NAME).read_text())["phase"]


def test_restart_while_a_cloud_machine_trains_follows_the_run_to_its_result(tmp_path, machine):
    async def scenario():
        stage = await killed_during(tmp_path, "training")
        async with started(tmp_path) as execution:
            job = await execution.get("cloud-training")
            assert job.status == "running" and job.error is None
            await asyncio.wait_for(machine.following.wait(), 5)
            assert machine.commands(job.id) == ["--status", "--follow"]
            assert dispatch_phase(stage) == "training"
            machine.end()
            await asyncio.wait_for(execution.tasks[job.id], 5)
            job = await execution.get(job.id)
            assert job.status == "succeeded", job.error
            assert [artifact.id for artifact in job.result.artifacts] == [f"{job.id}:operation"]
            commands = machine.commands(job.id)
            assert "launch" not in commands and commands.count("down") == 1
            assert commands.index("sync") < commands.index("down")
            assert dispatch_phase(stage) == "cleaned_up"
            messages = [event.message for event in execution.lifecycle.events(job.id)]
            assert "Following the cloud run again after an application restart" in messages
            assert "Optimizer step 7" in messages
            assert execution.cloud_cleanup_pending is False

    asyncio.run(scenario())


@pytest.mark.parametrize("ended,outcome", [(0, "succeeded"), (100, "failed")])
def test_cloud_run_that_ended_while_the_application_was_down_is_collected(
    tmp_path, machine, ended, outcome
):
    async def scenario():
        stage = await killed_during(tmp_path, "training")
        machine.end(ended)
        async with started(tmp_path) as execution:
            await asyncio.wait_for(execution.tasks["cloud-training"], 5)
            job = await execution.get("cloud-training")
            assert job.status == outcome, job.error
            if ended:
                assert "CUDA out of memory on the cloud machine" in job.error
                assert job.result is None
            else:
                assert job.result.artifacts[0].format == "training_checkpoint"
            assert machine.commands(job.id).count("down") == 1
            assert dispatch_phase(stage) == "cleaned_up"

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["prepared", "launching", "collecting"])
def test_restart_outside_training_still_interrupts_the_job_and_stops_its_machine(
    tmp_path, machine, phase
):
    async def scenario():
        stage = await killed_during(tmp_path, phase)
        async with started(tmp_path) as execution:
            job = await execution.get("cloud-training")
            assert job.status == "interrupted"
            assert "recovered running job" in job.error
            assert job.id not in execution.tasks
            commands = machine.commands(job.id)
            assert "--status" not in commands and "--follow" not in commands
            assert commands[-1] == "down"
            assert dispatch_phase(stage) == "cleaned_up"

    asyncio.run(scenario())


@pytest.mark.parametrize("answer", [102, 103, 1])
def test_restart_stops_a_machine_whose_cluster_no_longer_knows_the_job(tmp_path, machine, answer):
    async def scenario():
        stage = await killed_during(tmp_path, "training")
        machine.status = answer
        async with started(tmp_path) as execution:
            job = await execution.get("cloud-training")
            assert job.status == "interrupted"
            assert job.id not in execution.tasks
            assert machine.commands(job.id)[0] == "--status"
            assert machine.commands(job.id)[-1] == "down"
            assert dispatch_phase(stage) == "cleaned_up"

    asyncio.run(scenario())


def test_unreachable_cloud_at_startup_neither_hangs_startup_nor_forgets_the_machine(
    tmp_path, machine, monkeypatch
):
    monkeypatch.setattr(sky_runner, "REATTACH_PROBE_SECONDS", 0.05)
    monkeypatch.setattr(sky_runner, "RECOVERY_BUDGET_SECONDS", 0.05)

    async def scenario():
        stage = await killed_during(tmp_path, "training")
        machine.reachable = False
        storage = Storage(tmp_path)
        await storage.initialize()
        execution = Execution(storage, Settings(data_dir=tmp_path))
        try:
            await asyncio.wait_for(execution.reconcile(), 5)
            job = await execution.get("cloud-training")
            assert job.status == "interrupted"
            assert job.id not in execution.tasks
            # Nothing confirmed the machine stopped, so it stays on the books: the job
            # says so, its record keeps the cluster, and new cloud work is refused.
            state = json.loads((stage / sky_runner.STATE_NAME).read_text())
            assert state["phase"] == "cleanup_failed" and state["cluster"].startswith("fb-")
            assert "Cloud cleanup needs attention" in job.error
            assert execution.cloud_cleanup_pending is True
        finally:
            machine.reachable = True
            await execution.close()
            await storage.close()

    asyncio.run(scenario())


def test_stopping_the_application_leaves_a_training_machine_for_the_next_start(tmp_path, machine):
    async def scenario():
        stage = await killed_during(tmp_path, "training")
        async with started(tmp_path) as execution:
            await asyncio.wait_for(machine.following.wait(), 5)
        # Stopped normally while the machine trains: nothing was torn down.
        assert "down" not in machine.commands("cloud-training")
        assert "cancel" not in machine.commands("cloud-training")
        assert dispatch_phase(stage) == "training"
        machine.following.clear()
        async with started(tmp_path) as execution:
            job = await execution.get("cloud-training")
            assert job.status == "running" and job.error is None
            messages = [event.message for event in execution.lifecycle.events(job.id)]
            assert any("keeps training" in message for message in messages)
            await asyncio.wait_for(machine.following.wait(), 5)
            machine.end()
            await asyncio.wait_for(execution.tasks[job.id], 5)
            assert (await execution.get(job.id)).status == "succeeded"
            assert machine.commands(job.id).count("down") == 1

    asyncio.run(scenario())


def test_cancelling_a_followed_cloud_run_stops_its_machine(tmp_path, machine):
    async def scenario():
        stage = await killed_during(tmp_path, "training")
        async with started(tmp_path) as execution:
            await asyncio.wait_for(machine.following.wait(), 5)
            cancelled = await execution.cancel("cloud-training")
            assert cancelled.status == "cancelled"
            assert cancelled.cancellation.cleanup == "succeeded"
            assert machine.commands("cloud-training")[-1] == "down"
            assert dispatch_phase(stage) == "cleaned_up"

    asyncio.run(scenario())


def test_a_restart_keeps_no_more_cloud_machines_than_the_cloud_limit(tmp_path, machine):
    async def scenario():
        await killed_during(tmp_path, "training", "first")
        await killed_during(tmp_path, "training", "second")
        await killed_during(tmp_path, "training", "third")
        async with started(tmp_path) as execution:
            statuses = {
                name: (await execution.get(name)).status for name in ("first", "second", "third")
            }
            assert statuses == {"first": "running", "second": "running", "third": "interrupted"}
            assert set(execution.tasks) == {"first", "second"}
            assert execution.cloud_slots.locked()
            assert machine.commands("third")[-1] == "down"
            assert "down" not in machine.commands("first") + machine.commands("second")
            machine.end()
            await asyncio.wait_for(asyncio.gather(*execution.tasks.values()), 5)
            assert not execution.cloud_slots.locked()

    asyncio.run(scenario())
