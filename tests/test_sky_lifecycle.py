"""Cloud lifecycle dispatch with an isolated runner fixture; no cloud resources."""

import asyncio
import hashlib
import json
import sys
import types
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import insert
from vla_platform.api import create_app
from vla_platform.compute_settings import ComputeSettings
from vla_platform.contracts import IntakeRequest, Job, now
from vla_platform.datasets.inspect import profile
from vla_platform.execution import Execution
from vla_platform.lifecycle import service
from vla_platform.lifecycle.contracts import CloudExecutionTarget, PolicyRequest
from vla_platform.lifecycle.runtime import Runtime
from vla_platform.settings import Settings
from vla_platform.storage import Storage, jobs

RUNTIME_ID = "skypilot-gcp-A100"
PROJECT = "project"
DATASET = "dataset-ready"
SHA = "a" * 40
TARGET = {
    "project_id": "billing-project",
    "workspace": "firebird-gcp-fixture",
    "sky_api_endpoint": "http://127.0.0.1:46580",
    "region": "us-central1",
    "accelerator": "A100",
    "gpu_count": 1,
    "instance_type": "a2-highgpu-1g",
    "disk_size_gb": 200,
    "idle_minutes": 10,
}


def request(**updates):
    return PolicyRequest.model_validate(
        {
            "operation": "policy.finetune",
            "runtime_id": RUNTIME_ID,
            "dataset_job_id": DATASET,
            "training": {"steps": 1},
            **updates,
        }
    )


def write_result(payload, stage_dir, *, error=None, corrupt=False, wrong_identity=False):
    result = {
        "schema_version": 1,
        "job_id": "different-job" if wrong_identity else payload["job_id"],
        "report": {"scope": "cloud_protocol_fixture"},
    }
    if error:
        result["error"] = error
    else:
        bundle = stage_dir / "bundle"
        bundle.mkdir()
        raw = b"fixture checkpoint, not model weights"
        (bundle / "checkpoint.bin").write_bytes(raw)
        (bundle / "manifest.json").write_text(
            json.dumps(
                {
                    "files": {
                        "checkpoint.bin": "0" * 64 if corrupt else hashlib.sha256(raw).hexdigest()
                    },
                    "metadata": {"fixture_only": True, "method": "lora"},
                }
            )
        )
        result["artifact"] = {
            "path": str(bundle),
            "label": "Cloud fixture checkpoint",
            "format": "training_checkpoint",
        }
    (stage_dir / "result.json").write_text(json.dumps(result))


@pytest.fixture
def cloud(tmp_path, monkeypatch):
    runtime = Runtime(
        id=RUNTIME_ID,
        label="A100 cloud fixture",
        execution="skypilot",
        provider="gcp",
        region="us-central1",
        accelerator="A100",
        device="cuda",
        worker_root="workers/vla_cpp",
        training_root="workers/smolvla_qlora",
        training_python="python",
        vendor="skypilot",
        build="skypilot",
    )
    state = {"runtime": runtime, "target": dict(TARGET), "enabled": True, "native_calls": 0}

    def require_enabled(self, selected):
        if not state["enabled"]:
            raise ValueError("Google Cloud training is disabled or needs setup")

    def no_native(*args, **kwargs):
        state["native_calls"] += 1
        raise AssertionError("Cloud training must never launch a native subprocess")

    async def unconfigured_runner(*args, **kwargs):
        raise AssertionError("Test must provide a fake cloud runner")

    runner = types.ModuleType("vla_platform.lifecycle.sky_runner")
    runner.run = unconfigured_runner

    async def recovered(_data_dir):
        return []

    runner.recover = recovered
    monkeypatch.setitem(sys.modules, runner.__name__, runner)
    monkeypatch.setattr("vla_platform.lifecycle.sky_runner", runner, raising=False)
    monkeypatch.setattr(
        ComputeSettings,
        "runtime",
        lambda self, key: state["runtime"] if key == RUNTIME_ID else None,
    )
    monkeypatch.setattr(ComputeSettings, "require_enabled", require_enabled)
    monkeypatch.setattr(ComputeSettings, "cloud_target", lambda self, key: dict(state["target"]))

    async def planned(self, key):
        return dict(state["target"])

    async def ensured(self, key, target):
        require_enabled(self, state["runtime"])
        return target

    monkeypatch.setattr(ComputeSettings, "plan_cloud_target", planned)
    monkeypatch.setattr(ComputeSettings, "ensure_cloud_ready", ensured)
    monkeypatch.setattr(service, "command", no_native)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", no_native)

    @asynccontextmanager
    async def workspace():
        directory = tmp_path / "workspace"
        directory.mkdir(exist_ok=True)
        settings = Settings(data_dir=directory)
        storage = Storage(directory)
        await storage.initialize()
        execution = Execution(storage, settings)
        intake = IntakeRequest(repo_id="fixture/data", revision=SHA)
        metadata = json.dumps(
            {
                "codebase_version": "v3.0",
                "total_episodes": 1,
                "total_frames": 10,
                "fps": 10,
                "features": {
                    "action": {"dtype": "float32", "shape": [6]},
                    "observation.state": {"dtype": "float32", "shape": [6]},
                },
            }
        ).encode()
        dataset = Job(
            id=DATASET,
            project_id=PROJECT,
            status="succeeded",
            request=intake,
            created_at=now(),
            updated_at=now(),
            result=profile(metadata, intake, SHA),
        )
        async with storage.engine.begin() as connection:
            await connection.execute(
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

    return state, runner, workspace


async def finished(execution, job):
    task = execution.tasks[job.id]
    await asyncio.wait_for(task, 5)
    return await execution.get(job.id)


def test_cloud_training_uses_sky_and_imports_verified_bundle(cloud):
    state, runner, workspace = cloud
    calls = []

    async def run(payload, stage_dir, target, on_event):
        calls.append((payload, target))
        await on_event("Cloud fixture: training step 1")
        write_result(payload, stage_dir)
        return 0

    runner.run = run

    async def exercise():
        async with workspace() as execution:
            submitted = await execution.submit(PROJECT, request())
            assert submitted.compute_target == CloudExecutionTarget.model_validate(TARGET)
            job = await finished(execution, submitted)
            assert job.status == "succeeded", job.error
            assert len(calls) == 1
            payload, target = calls[0]
            assert target == TARGET
            assert payload["operation"] == "policy.finetune"
            assert payload["dataset"]["revision"] == SHA
            assert payload["parameters"]["training"]["model_revision"]
            assert payload["runtime"]["execution"] == "skypilot"
            assert state["native_calls"] == 0
            artifacts = await execution.lifecycle.artifacts(PROJECT)
            assert len(artifacts) == 1
            artifact = artifacts[0]
            assert artifact.job_id == job.id
            assert artifact.format == "training_checkpoint"
            assert artifact.metadata["fixture_only"] is True
            assert artifact.file_bytes == len(b"fixture checkpoint, not model weights")
            assert artifact.path.startswith(f"jobs/{job.id}/operation/")
            assert len(artifact.manifest_sha256) == 64
            assert (await execution.lifecycle.download(PROJECT, artifact.id)).is_file()
            messages = [event.message for event in execution.lifecycle.events(job.id)]
            assert "Cloud fixture: training step 1" in messages
            assert any("billing-project, us-central1" in message for message in messages)

    asyncio.run(exercise())


def test_queued_cloud_run_keeps_submission_billing_target(cloud):
    state, runner, workspace = cloud
    launched = []

    async def run(payload, stage_dir, target, on_event):
        launched.append(target)
        write_result(payload, stage_dir)
        return 0

    runner.run = run

    async def exercise():
        async with workspace() as execution:
            execution.native_slots = asyncio.Semaphore(0)
            submitted = await execution.submit(PROJECT, request())
            # Mutating settings while the job waits must not change its billing
            # account, location, accelerator, storage size, or idle policy.
            state["target"].update(
                project_id="different-project",
                region="europe-west4",
                accelerator="L4",
                instance_type="g2-standard-4",
                disk_size_gb=500,
                idle_minutes=30,
            )
            state["runtime"] = state["runtime"].model_copy(update={"region": "europe-west4"})
            execution.native_slots.release()
            job = await finished(execution, submitted)
            assert job.status == "succeeded", job.error
            assert launched == [TARGET]
            assert job.compute_target.model_dump() == TARGET
            assert state["native_calls"] == 0

    asyncio.run(exercise())


def test_disabled_queued_cloud_run_fails_without_launch(cloud):
    state, runner, workspace = cloud

    async def exercise():
        async with workspace() as execution:
            execution.native_slots = asyncio.Semaphore(0)
            submitted = await execution.submit(PROJECT, request())
            state["enabled"] = False
            execution.native_slots.release()
            job = await finished(execution, submitted)
            assert job.status == "failed"
            assert "disabled or needs setup" in job.error
            assert job.result is None
            assert state["native_calls"] == 0

    asyncio.run(exercise())


def test_cloud_selection_cannot_reroute_to_native_while_queued(cloud):
    state, runner, workspace = cloud

    async def exercise():
        async with workspace() as execution:
            execution.native_slots = asyncio.Semaphore(0)
            submitted = await execution.submit(PROJECT, request())
            state["runtime"] = state["runtime"].model_copy(
                update={"execution": "native", "provider": "local"}
            )
            execution.native_slots.release()
            job = await finished(execution, submitted)
            assert job.status == "failed"
            assert job.result is None
            assert state["native_calls"] == 0

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "operation",
    [
        "policy.import",
        "policy.export",
        "policy.evaluate",
        "policy.run",
        "policy.workflow",
    ],
)
def test_cloud_rejects_unsupported_operations(cloud, operation):
    _, _, workspace = cloud

    async def exercise():
        async with workspace() as execution:
            fields = {"operation": operation, "artifact_id": "unused", "training": None}
            if operation == "policy.export":
                fields["dataset_job_id"] = None
            with pytest.raises(ValueError, match="support fine-tuning and quantization"):
                await execution.submit(PROJECT, request(**fields))
            assert [job.id for job in await execution.list(PROJECT)] == [DATASET]
            assert not execution.tasks

    asyncio.run(exercise())


@pytest.mark.parametrize("selected_precision", [None, {"language": "Q4_0", "vision": "Q8_0"}])
def test_cloud_quantization_dispatches_selected_training_artifact_once(cloud, selected_precision):
    state, runner, workspace = cloud
    calls = []

    async def run(payload, stage_dir, target, on_event):
        calls.append(payload)
        write_result(payload, stage_dir)
        if payload["operation"] == "policy.quantize":
            await on_event('_quantization:{"phase":"exporting","message":"Merging checkpoint"}')
            response = json.loads((stage_dir / "result.json").read_text())
            response["artifact"]["format"] = "gguf"
            (stage_dir / "result.json").write_text(json.dumps(response))
        return 0

    runner.run = run

    async def exercise():
        async with workspace() as execution:
            training = await finished(execution, await execution.submit(PROJECT, request()))
            artifact = training.result.artifacts[0]
            quantization = await finished(
                execution,
                await execution.submit(
                    PROJECT,
                    request(
                        operation="policy.quantize",
                        artifact_id=artifact.id,
                        training=None,
                        precision=selected_precision,
                    ),
                ),
            )
            assert quantization.status == "succeeded", quantization.error
            assert len(calls) == 2
            assert calls[1]["artifact"]["id"] == artifact.id
            assert calls[1]["parameters"]["precision"] == (
                selected_precision or {"language": "Q8_0", "vision": None}
            )
            assert quantization.result.artifacts[0].parent_ids == [artifact.id]
            assert state["native_calls"] == 0
            assert any(
                event.stage == "exporting" and event.message == "Merging checkpoint"
                for event in execution.lifecycle.events(quantization.id)
            )

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "action",
    [
        "quantize-earlier",
        "resume-latest",
        "resume-torn-pointer",
        "resume-corrupt-latest",
        "resume-outside-marker",
        "resume-corrupt-descriptor",
        "resume-descriptor-list",
        "resume-descriptor-null",
        "resume-torn-index",
        "resume-invalid-index-list",
        "resume-invalid-index-entry",
        "resume-no-label",
    ],
)
def test_cloud_checkpoint_projection_and_resume_transfer_only_descriptors(cloud, action):
    from vla_platform.lifecycle.cloud_storage import encoded, install_descriptor, write_json

    _, runner, workspace = cloud
    calls = []

    async def run(payload, stage_dir, target, on_event):
        calls.append(payload)
        write_result(payload, stage_dir)
        return 0

    runner.run = run

    async def exercise():
        async with workspace() as execution:
            original = await finished(
                execution, await execution.submit(PROJECT, request(training={"steps": 4}))
            )
            stage = execution.settings.data_dir / "jobs" / original.id / "operation"
            entries = []
            for step in (1, 2):
                name = f"checkpoint-{step:06d}"
                uri = f"gs://test-bucket/jobs/{original.id}/operation/checkpoints/{name}"
                checkpoint_files = {
                    name: "a" * 64
                    for name in (
                        "stats.json",
                        "splits.json",
                        "training.pt",
                        "probe.safetensors",
                        "adapter/adapter_config.json",
                        "adapter/adapter_model.safetensors",
                        "policy/config.json",
                    )
                }
                checkpoint_files["recipe.json"] = hashlib.sha256(
                    encoded(original.request.training)
                ).hexdigest()
                checkpoint_manifest = {"schema_version": 1, "step": step, "files": checkpoint_files}
                manifest = {
                    "files": {
                        **{"checkpoint/" + name: sha for name, sha in checkpoint_files.items()},
                        "checkpoint/manifest.json": hashlib.sha256(
                            encoded(checkpoint_manifest)
                        ).hexdigest(),
                    },
                    "metadata": {
                        "architecture": "smolvla",
                        "step": step,
                        "storage": "gcs",
                        "remote_uri": uri,
                    },
                }
                item = {
                    "name": name,
                    "step": step,
                    "uri": uri,
                    "manifest": manifest,
                    "manifest_sha256": hashlib.sha256(encoded(manifest)).hexdigest(),
                    "file_bytes": 2 * 1024**3,
                    "label": f"Step {step}",
                }
                entries.append(item)
                descriptor = stage / "cloud-checkpoints" / name
                install_descriptor(descriptor, item)
                local_checkpoint = stage / "training" / name
                write_json(local_checkpoint / "manifest.json", checkpoint_manifest)
                write_json(local_checkpoint / "recipe.json", original.request.training)
                write_json(local_checkpoint / "remote-checkpoint.json", {"bundle": str(descriptor)})
            write_json(stage / "cloud-checkpoints.json", {"checkpoints": entries})
            write_json(
                stage / "training/latest.json", {"checkpoint": "checkpoint-000002", "step": 2}
            )
            artifacts = await execution.lifecycle.artifacts(PROJECT)
            remote = [
                artifact for artifact in artifacts if artifact.metadata.get("storage") == "gcs"
            ]
            assert len(remote) == 2
            assert {artifact.metadata["step"] for artifact in remote} == {1, 2}
            assert all(artifact.file_bytes == 2 * 1024**3 for artifact in remote)
            if action == "quantize-earlier":
                followup = request(
                    operation="policy.quantize",
                    training=None,
                    artifact_id=f"{original.id}:checkpoint-000001",
                )
                selected_step = 1
            else:
                original.status = "interrupted"
                await execution.save(original)
                followup = request(resume_job_id=original.id, training=None)
                selected_step = 2
                if action == "resume-torn-pointer":
                    (stage / "training/latest.json").write_text('{"checkpoint":')
                elif action == "resume-corrupt-latest":
                    (stage / "training/checkpoint-000002/recipe.json").write_text("{}")
                    selected_step = 1
                elif action in {
                    "resume-corrupt-descriptor",
                    "resume-descriptor-list",
                    "resume-descriptor-null",
                }:
                    damage = {
                        "resume-corrupt-descriptor": "{",
                        "resume-descriptor-list": "[]",
                        "resume-descriptor-null": "null",
                    }[action]
                    (stage / "cloud-checkpoints/checkpoint-000002/manifest.json").write_text(damage)
                    selected_step = 1
                    available = await execution.lifecycle.artifacts(PROJECT)
                    assert [
                        a.metadata["step"] for a in available if a.metadata.get("storage") == "gcs"
                    ] == [1]
                    with pytest.raises(ValueError, match="unavailable or corrupt"):
                        await execution.lifecycle.artifact(
                            PROJECT, f"{original.id}:checkpoint-000002"
                        )
                elif action == "resume-torn-index":
                    (stage / "cloud-checkpoints.json").write_text('{"checkpoints":')
                elif action == "resume-invalid-index-list":
                    write_json(stage / "cloud-checkpoints.json", {"checkpoints": {"wrong": "type"}})
                elif action == "resume-invalid-index-entry":
                    write_json(
                        stage / "cloud-checkpoints.json", {"checkpoints": [None, {}, *entries]}
                    )
                elif action == "resume-no-label":
                    entries[-1].pop("label")
                    write_json(stage / "cloud-checkpoints.json", {"checkpoints": entries})
                elif action == "resume-outside-marker":
                    write_json(
                        stage / "training/checkpoint-000002/remote-checkpoint.json",
                        {"bundle": str(execution.settings.data_dir / "another-run")},
                    )
            available = await execution.lifecycle.artifacts(PROJECT)
            assert original.result.artifacts[0].id in {artifact.id for artifact in available}
            completed = await finished(execution, await execution.submit(PROJECT, followup))
            assert completed.status == "succeeded", completed.error
            available = await execution.lifecycle.artifacts(PROJECT)
            assert {original.result.artifacts[0].id, completed.result.artifacts[0].id}.issubset(
                {artifact.id for artifact in available}
            )
            payload = calls[-1]
            assert payload["artifact"]["id"] == f"{original.id}:checkpoint-{selected_step:06d}"
            assert completed.result.artifacts[0].parent_ids == [payload["artifact"]["id"]]
            assert "resume_checkpoint" not in payload
            transfer = Path(payload["artifact"]["path"])
            assert {path.name for path in transfer.iterdir()} == {"manifest.json", "remote.json"}
            assert not list(stage.rglob("weights.bin"))

    from pathlib import Path

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "failure,expected",
    [
        ("worker_error", "Cloud training failed"),
        ("exit_code", "Native worker failed"),
        ("exception", "Cloud transport unavailable"),
        ("identity", "identity mismatch"),
        ("artifact_hash", "hash mismatch"),
    ],
)
def test_cloud_failure_and_untrusted_results_never_publish_artifact(cloud, failure, expected):
    state, runner, workspace = cloud

    async def run(payload, stage_dir, target, on_event):
        if failure == "exception":
            raise RuntimeError("Cloud transport unavailable")
        write_result(
            payload,
            stage_dir,
            error="Cloud training failed" if failure == "worker_error" else None,
            corrupt=failure == "artifact_hash",
            wrong_identity=failure == "identity",
        )
        return 2 if failure == "exit_code" else 0

    runner.run = run

    async def exercise():
        async with workspace() as execution:
            submitted = await execution.submit(PROJECT, request())
            job = await finished(execution, submitted)
            assert job.status == "failed"
            assert expected in job.error
            assert job.result is None
            assert await execution.lifecycle.artifacts(PROJECT) == []
            assert state["native_calls"] == 0

    asyncio.run(exercise())


@pytest.mark.parametrize("late_result", [False, True])
def test_cancellation_reaches_cloud_runner_and_fences_late_results(cloud, late_result):
    state, runner, workspace = cloud

    async def exercise():
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def run(payload, stage_dir, target, on_event):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                if late_result:
                    write_result(payload, stage_dir)
                    return 0
                raise

        runner.run = run
        async with workspace() as execution:
            submitted = await execution.submit(PROJECT, request())
            await asyncio.wait_for(started.wait(), 5)
            response = await execution.cancel(submitted.id)
            assert response.status == "cancelled"
            assert cancelled.is_set()
            job = await execution.get(submitted.id)
            assert job.status == "cancelled"
            assert job.result is None
            assert await execution.lifecycle.artifacts(PROJECT) == []
            assert state["native_calls"] == 0

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "field", ["compute_target", "cloud_target", "project_id", "instance_type", "execution"]
)
def test_policy_api_cannot_inject_server_owned_compute_target(tmp_path, field):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        project = client.post("/api/v1/projects", json={"name": "Cloud request boundary"}).json()
        response = client.post(
            f"/api/v1/projects/{project['id']}/policy-jobs",
            json={
                **request().model_dump(),
                field: TARGET if field.endswith("target") else "injected",
            },
        )
        assert response.status_code == 422
        assert any(
            error["type"] == "extra_forbidden" and error["loc"][-1] == field
            for error in response.json()["detail"]
        )
        assert client.get(f"/api/v1/projects/{project['id']}/jobs").json() == []


def test_recovery_exposes_failed_cloud_cleanup_without_restarting_training(cloud):
    state, runner, workspace = cloud

    async def run(payload, stage_dir, target, on_event):
        write_result(payload, stage_dir)
        return 0

    runner.run = run

    async def scenario():
        async with workspace() as execution:
            job = await execution.submit(PROJECT, request())
            await execution.tasks[job.id]

            async def recover(_data_dir):
                return [
                    {
                        "job_id": job.id,
                        "cluster": "fixture-owned-cluster",
                        "error": "Cloud cleanup needs attention",
                    }
                ]

            runner.recover = recover
            await execution.reconcile()
            current = await execution.get(job.id)
            assert current.status == "succeeded"
            assert current.result.artifacts
            assert "Cloud cleanup needs attention" in current.error
            assert any(
                event.stage == "cloud_cleanup" for event in execution.lifecycle.events(job.id)
            )
            assert state["native_calls"] == 0

    asyncio.run(scenario())


def test_cloud_jobs_use_two_independent_gpu_slots(cloud):
    _, runner, workspace = cloud

    async def exercise():
        both_started = asyncio.Event()
        release = asyncio.Event()
        active = set()
        peak = 0

        async def run(payload, stage_dir, target, on_event):
            nonlocal peak
            active.add(payload["job_id"])
            peak = max(peak, len(active))
            if len(active) == 2:
                both_started.set()
            await release.wait()
            write_result(payload, stage_dir)
            active.remove(payload["job_id"])
            return 0

        runner.run = run
        async with workspace() as execution:
            first = await execution.submit(PROJECT, request())
            second = await execution.submit(PROJECT, request())
            third = await execution.submit(PROJECT, request())
            await asyncio.wait_for(both_started.wait(), 5)
            assert (await execution.get(first.id)).status == "running"
            assert (await execution.get(second.id)).status == "running"
            assert (await execution.get(third.id)).status == "queued"
            release.set()
            results = await asyncio.gather(
                *(finished(execution, job) for job in (first, second, third))
            )
            assert all(job.status == "succeeded" for job in results)
            assert peak == 2

    asyncio.run(exercise())
