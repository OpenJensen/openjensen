"""Start queues promptly; cloud preparation happens inside the owned background job."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from sqlalchemy import insert
from vla_platform import cloud_compute_catalog as catalog
from vla_platform.contracts import IntakeRequest, Job, now
from vla_platform.datasets.inspect import profile
from vla_platform.execution import Execution
from vla_platform.lifecycle import sky_runner
from vla_platform.lifecycle.contracts import PolicyRequest
from vla_platform.settings import Settings
from vla_platform.storage import Storage, jobs


def recipe(**updates):
    return PolicyRequest.model_validate(
        {
            "operation": "policy.finetune",
            "runtime_id": "skypilot-gcp-A100",
            "dataset_job_id": "dataset",
            "training_method": "lora",
            "training": {"steps": 10, "warmup_steps": 0, "batch_size": 1, "save_every": 5},
            **updates,
        }
    )


@pytest.fixture
def one_click(tmp_path, monkeypatch):
    directory = tmp_path / "workspace"
    directory.mkdir()
    connection_path = directory / "cloud-connections.json"
    connection = {"project_id": "billing-project", "region": "us-central1"}
    connection_path.write_text(json.dumps({"version": 1, "providers": {"gcp": connection}}))
    (directory / "compute-settings.json").write_text(
        json.dumps(
            {
                "version": 1,
                "local": {"enabled": False, "label": "Local"},
                "gcp": {
                    "enabled": True,
                    "default_gpu": "A100",
                    "disk_size_gb": 200,
                    "idle_minutes": 10,
                },
            }
        )
    )
    state = SimpleNamespace(
        prepare_calls=[],
        runner_calls=[],
        blocked=False,
        ignore_cancel=False,
        started=None,
        release=None,
    )
    endpoint = "http://127.0.0.1:46580"

    async def configured_endpoint(_sky):
        return endpoint

    async def prepare(_sky, project, endpoint=None):
        state.prepare_calls.append((project, endpoint))
        state.started.set()
        if state.blocked:
            try:
                await state.release.wait()
            except asyncio.CancelledError:
                if not state.ignore_cancel:
                    raise
        return {"workspace": catalog.sky_workspace_name(project), "sky_api_endpoint": endpoint}

    async def verify(_sky, project, endpoint=None):
        return {"workspace": catalog.sky_workspace_name(project), "sky_api_endpoint": endpoint}

    async def readonly(executable, args, **kwargs):
        if args[:3] == ["auth", "application-default", "print-access-token"]:
            return b"unpaid-fixture-token"
        if args[:2] == ["services", "list"]:
            return json.dumps(
                [{"config": {"name": name}} for name in catalog.REQUIRED_GCP_SERVICES]
            ).encode()
        if args[:2] == ["gpus", "list"]:
            return json.dumps(
                {
                    gpu.id: [
                        {
                            "region": "us-central1",
                            "instance_type": gpu.instance_type,
                            "accelerator_count": 1,
                            "accelerator_name": gpu.id,
                            "cloud": "GCP",
                        }
                    ]
                    for gpu in catalog.GCP_GPUS
                }
            ).encode()
        return b""

    async def run(payload, stage_dir, target, on_event):
        state.runner_calls.append(dict(target))
        (stage_dir / "result.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "job_id": payload["job_id"],
                    "report": {"scope": "unpaid-fixture"},
                }
            )
        )
        return 0

    async def recover(_directory):
        return []

    async def forbidden_process(*args, **kwargs):
        pytest.fail("The one-click suite must never execute a real cloud command")

    monkeypatch.setattr(catalog, "sky_executable", lambda: "/fixture/sky")
    monkeypatch.setattr(catalog, "sky_python", lambda _: "/fixture/python")
    monkeypatch.setattr(catalog, "configured_sky_endpoint", configured_endpoint)
    monkeypatch.setattr(catalog, "prepare_sky_target", prepare)
    monkeypatch.setattr(catalog, "verify_sky_target", verify)
    monkeypatch.setattr(catalog, "run_readonly", readonly)
    monkeypatch.setattr(
        "vla_platform.compute_settings.shutil.which", lambda name: "/fixture/" + name
    )
    monkeypatch.setattr(sky_runner, "run", run)
    monkeypatch.setattr(sky_runner, "recover", recover)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden_process)

    @asynccontextmanager
    async def workspace():
        state.started, state.release = asyncio.Event(), asyncio.Event()
        storage = Storage(directory)
        await storage.initialize()
        execution = Execution(storage, Settings(data_dir=directory))
        if await execution.get("dataset") is None:
            intake = IntakeRequest(repo_id="fixture/robot", revision="a" * 40)
            metadata = json.dumps(
                {
                    "codebase_version": "v3.0",
                    "total_episodes": 2,
                    "total_frames": 20,
                    "fps": 10,
                    "features": {
                        "action": {"dtype": "float32", "shape": [6]},
                        "observation.state": {"dtype": "float32", "shape": [6]},
                    },
                }
            ).encode()
            dataset = Job(
                id="dataset",
                project_id="project",
                status="succeeded",
                request=intake,
                created_at=now(),
                updated_at=now(),
                result=profile(metadata, intake, "a" * 40),
            )
            async with storage.engine.begin() as db:
                await db.execute(
                    insert(jobs).values(
                        id=dataset.id,
                        project_id=dataset.project_id,
                        status=dataset.status,
                        record=dataset.model_dump(),
                    )
                )
        try:
            yield execution
        finally:
            await execution.close()
            await storage.close()

    return state, workspace, connection_path


def test_start_returns_while_real_compute_preparation_is_blocked(one_click):
    state, workspace, _ = one_click
    state.blocked = True

    async def exercise():
        async with workspace() as execution:
            assert execution.lifecycle.compute.gcp_status().status == "unchecked"
            job = await asyncio.wait_for(execution.submit("project", recipe()), 2)
            await asyncio.wait_for(state.started.wait(), 2)
            assert state.runner_calls == []
            assert (await execution.get(job.id)).status == "running"
            assert job.compute_target.project_id == "billing-project"
            state.release.set()
            await asyncio.wait_for(execution.tasks[job.id], 2)
            finished = await execution.get(job.id)
            assert finished.status == "succeeded", finished.error
            assert state.runner_calls[0]["project_id"] == "billing-project"

    asyncio.run(exercise())


def test_combined_admission_checks_all_project_owned_sources_before_preparation(one_click):
    state, workspace, _ = one_click
    async def exercise():
        async with workspace() as execution:
            first = await execution.get("dataset")
            first.result.features["observation.images.front"] = {"dtype": "video", "shape": [3, 32, 32]}
            await execution.save(first)
            second = first.model_copy(deep=True, update={"id": "second"})
            second.result.repo_id = "fixture/second"
            async with execution.storage.engine.begin() as db:
                await db.execute(insert(jobs).values(id=second.id, project_id=second.project_id, status=second.status, record=second.model_dump()))
            request = recipe(dataset_job_ids=["dataset", "second"], dataset_camera_mappings={ident: {"observation.images.front": "observation.images.front"} for ident in ("dataset", "second")})
            request.training.update(camera_keys=["observation.images.front"], camera_key="observation.images.front")
            await execution.lifecycle.validate("project", request)
            assert not state.prepare_calls and not state.runner_calls
            second.result.features["action"]["shape"] = [7]
            await execution.save(second)
            with pytest.raises(ValueError, match="matching action"):
                await execution.lifecycle.validate("project", request)
            assert not state.prepare_calls and not state.runner_calls
    asyncio.run(exercise())


@pytest.mark.parametrize("ignore_cancel", [False, True])
def test_cancel_during_preparation_never_launches_gpu_even_after_late_reply(
    one_click, ignore_cancel
):
    state, workspace, _ = one_click
    state.blocked, state.ignore_cancel = True, ignore_cancel

    async def exercise():
        async with workspace() as execution:
            job = await execution.submit("project", recipe())
            await asyncio.wait_for(state.started.wait(), 2)
            await asyncio.wait_for(execution.cancel(job.id), 2)
            assert (await execution.get(job.id)).status == "cancelled"
            assert state.runner_calls == []

    asyncio.run(exercise())


def test_changed_queued_project_is_rejected_without_preparing_another_project(one_click):
    state, workspace, connection_path = one_click

    async def exercise():
        async with workspace() as execution:
            await execution.native_slots.acquire()
            job = await execution.submit("project", recipe())
            connection_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "providers": {
                            "gcp": {"project_id": "other-project", "region": "us-central1"},
                        },
                    }
                )
            )
            execution.native_slots.release()
            await asyncio.wait_for(execution.tasks[job.id], 2)
            finished = await execution.get(job.id)
            assert finished.status == "failed"
            assert finished.compute_target.project_id == "billing-project"
            assert state.prepare_calls == state.runner_calls == []

    asyncio.run(exercise())


def test_restart_without_verified_flag_does_not_require_manual_setup(one_click):
    state, workspace, _ = one_click

    async def exercise():
        for _ in range(2):
            async with workspace() as execution:
                assert execution.lifecycle.compute.gcp_status().status == "unchecked"
                job = await execution.submit("project", recipe())
                await asyncio.wait_for(execution.tasks[job.id], 2)
                finished = await execution.get(job.id)
                assert finished.status == "succeeded", finished.error
        assert len(state.runner_calls) == 2

    asyncio.run(exercise())


@pytest.mark.parametrize("field", ["steps", "batch_size"])
def test_invalid_recipe_fails_before_preparation(one_click, field):
    state, workspace, _ = one_click

    async def exercise():
        async with workspace() as execution:
            with pytest.raises(ValueError):
                await execution.submit("project", recipe(training={field: -1}))
            assert state.prepare_calls == state.runner_calls == []
            assert len(await execution.list("project")) == 1

    asyncio.run(exercise())


def test_caller_cannot_supply_worker_dataset_bindings(one_click):
    state, workspace, _ = one_click

    async def exercise():
        async with workspace() as execution:
            with pytest.raises(ValueError, match="application manages combined dataset"):
                await execution.submit("project", recipe(training={"dataset_sources": [{"repo_id": "foreign/data"}]}))
            assert state.prepare_calls == state.runner_calls == []
            assert len(await execution.list("project")) == 1

    asyncio.run(exercise())
