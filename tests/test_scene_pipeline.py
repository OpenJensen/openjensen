"""Exercise the scene Job pipeline with provider, builder and renderer boundaries mocked."""

import asyncio
import json
import zipfile
from contextlib import asynccontextmanager

import pytest
from test_scene_generation import body, setup, source
from vla_platform.contracts import now
from vla_platform.execution import Execution
from vla_platform.scenes import dataset, runtime, service
from vla_platform.scenes.agent import SceneError
from vla_platform.scenes.contracts import Proposal, SceneRun
from vla_platform.settings import Settings
from vla_platform.storage import Storage

CORE_GATES = ("source_evidence", "agent_review", "usd_structure", "isaac_runtime")


@asynccontextmanager
async def pipeline(tmp_path, monkeypatch):
    scenes, roles = setup(tmp_path, monkeypatch)
    storage = Storage(tmp_path)
    await storage.initialize()
    execution = Execution(storage, Settings(data_dir=tmp_path))
    execution.scenes = scenes
    execution._schedule = lambda job_id: None
    execution.scene_explorer = object()
    render_calls = []

    async def investigate(owner, run, dataset_job, explorer):
        frames, _ = await owner.evidence(run, dataset_job, explorer)
        folder = owner.folder(run.id)
        catalog = {
            "scope": "entire_dataset",
            "total_episodes": 30,
            "episodes": [{"episode_index": index} for index in range(30)],
        }
        trajectory = {
            "episode_index": 17,
            "fps": 30,
            "joint_names": [name + ".pos" for name in dataset.JOINTS],
            "mapping_status": "uncalibrated_dataset_normalized",
            "frames": [
                {"timestamp": index / 30, "positions": [0, -104, 90, 75, 4, 1]}
                for index in range(150)
            ],
        }
        service.write(folder / "dataset-catalog.json", catalog)
        service.write(folder / "trajectory.json", trajectory)
        run.artifacts += ["dataset-catalog.json", "trajectory.json"]
        run.source.update(
            scope="entire_dataset",
            total_episodes=30,
            inspected_episodes=[17],
            replay_episode=17,
            replay_frames=150,
        )
        evidence = {
            "source": run.source,
            "dataset_catalog": catalog,
            "replay": {"episode_index": 17},
        }
        service.write(folder / "evidence.json", evidence)
        return frames, evidence

    async def rendered(owner, run, bundle, trajectory):
        render_calls.append((run.id, bundle, trajectory))
        assert (bundle / "scene.usda").is_file()
        assert json.loads(trajectory.read_text())["episode_index"] == 17
        assert len(json.loads(trajectory.read_text())["frames"]) == 150
        folder = owner.folder(run.id) / "simulation"
        folder.mkdir()
        media = []
        for name, role in (
            ("environment-front.png", "environment_image"),
            ("environment-overview.png", "environment_image"),
            ("environment-side.png", "environment_image"),
            ("replay.mp4", "episode_replay"),
        ):
            (folder / name).write_bytes(b"fixture-verified-media")
            media.append(
                {"role": role, "path": "simulation/" + name, "label": name, "episode_index": 17}
            )
        result = {
            "status": "succeeded",
            "isaac_runtime": True,
            "calibration_verified": False,
            "calibrated_replay": False,
            "mapping_status": "uncalibrated_dataset_normalized",
            "source_frames": 150,
            "rendered_frames": 150,
            "media": media,
        }
        service.write(folder / "result.json", result)
        return result

    async def preflight(owner, run):
        assert run.stage == "preparing"
        assert (await execution.get(run.job_id)).stage == "preparing"

    monkeypatch.setattr(dataset, "investigate", investigate)
    monkeypatch.setattr(runtime, "preflight", preflight)
    monkeypatch.setattr(runtime, "run", rendered)
    await execution._accept(source(), None)
    try:
        yield execution, scenes, roles, render_calls
    finally:
        await execution.close()
        await storage.close()


def test_scene_job_runs_the_whole_pipeline_before_publishing_media_success(tmp_path, monkeypatch):
    async def exercise():
        async with pipeline(tmp_path, monkeypatch) as (execution, scenes, roles, renders):
            accepted = await execution.submit("project", body())
            await execution.run(accepted.id)
            job = await execution.get(accepted.id)
            scene = scenes.get(accepted.request.request_id)
            assert job.status == "succeeded" and job.result.validation_status == "needs_review"
            assert scene.job_id == job.id and scene.status == "needs_review"
            assert all(scene.gates[name] for name in CORE_GATES)
            assert not scene.gates["calibrated_replay"] and not scene.gates["visual_match"]
            assert roles == ["proposal", "review"] and len(renders) == 1
            assert len([item for item in scene.media if item.role == "environment_image"]) == 3
            assert len([item for item in scene.media if item.role == "episode_replay"]) == 1
            assert "simulation/replay.mp4" in job.result.scene_artifacts
            assert scenes.artifact(scene.id, "simulation/replay.mp4").is_file()
            with zipfile.ZipFile(scenes.artifact(scene.id, "scene-bundle.zip")) as bundle:
                assert "scene.usda" in bundle.namelist()
                assert len(json.loads(bundle.read("dataset-catalog.json"))["episodes"]) == 30
                assert len(json.loads(bundle.read("trajectory.json"))["frames"]) == 150

    asyncio.run(exercise())


@pytest.mark.parametrize("reason", ["Scene simulation is not configured.", "Renderer failed."])
def test_renderer_failure_keeps_usd_and_evidence_without_job_success(tmp_path, monkeypatch, reason):
    async def exercise():
        async with pipeline(tmp_path, monkeypatch) as (execution, scenes, roles, renders):

            async def unavailable(*args):
                raise SceneError(reason)

            monkeypatch.setattr(runtime, "run", unavailable)
            job = await execution.submit("project", body())
            await execution.run(job.id)
            current = await execution.get(job.id)
            scene = scenes.get(job.request.request_id)
            assert current.status == "failed" and scene.status == "failed"
            assert current.result.validation_status == "failed"
            assert scene.gates["usd_structure"] and not scene.gates["isaac_runtime"]
            assert reason in current.error
            assert scenes.artifact(scene.id, "scene-bundle.zip").is_file()
            assert scenes.artifact(scene.id, "dataset-catalog.json").is_file()
            assert "scene-bundle.zip" in current.result.scene_artifacts
            assert not renders and roles == ["proposal", "review"]

    asyncio.run(exercise())


def test_unsupported_scene_proposal_never_reaches_renderer(tmp_path, monkeypatch):
    async def exercise():
        async with pipeline(tmp_path, monkeypatch) as (execution, scenes, roles, renders):
            original = service.run_role

            async def unsupported(work, role, schema, *args, **kwargs):
                result, usage = await original(work, role, schema, *args, **kwargs)
                if schema is Proposal:
                    result = result.model_copy(update={"matches_template": False})
                    service.write(work / (role + ".json"), result.model_dump())
                return result, usage

            monkeypatch.setattr(service, "run_role", unsupported)
            job = await execution.submit("project", body())
            await execution.run(job.id)
            current = await execution.get(job.id)
            scene = scenes.get(job.request.request_id)
            assert current.status == "failed" and scene.status == "needs_review"
            assert not scene.gates["agent_review"] and not scene.gates["isaac_runtime"]
            assert not renders and all(item.role == "source_image" for item in scene.media)
            assert scenes.artifact(scene.id, "attempt-1/proposal.json").is_file()

    asyncio.run(exercise())


def test_accepted_model_and_power_survive_catalog_changes_before_dispatch(tmp_path, monkeypatch):
    async def exercise():
        async with pipeline(tmp_path, monkeypatch) as (execution, scenes, roles, renders):
            requested = body(model="gpt-6-luna", reasoning_effort="high")
            job = await execution.submit("project", requested)
            original = service.run_role
            selected = []
            stages = []

            async def observed(*args, **kwargs):
                selected.append((kwargs["model"], kwargs["reasoning_effort"]))
                stages.append((await execution.get(job.id)).stage)
                return await original(*args, **kwargs)

            monkeypatch.setattr(service, "run_role", observed)
            monkeypatch.setattr(scenes, "options", lambda: {"configured": False, "models": []})
            await execution.run(job.id)
            assert (await execution.get(job.id)).status == "succeeded"
            assert selected == [("gpt-6-luna", "high"), ("gpt-6-luna", "high")]
            assert stages == ["planning", "review"]
            assert scenes.get(job.request.request_id).agent.model == "gpt-6-luna"
            assert len(renders) == 1

    asyncio.run(exercise())


def test_shutdown_interrupts_scene_and_job_without_repeating_work(tmp_path, monkeypatch):
    async def exercise():
        async with pipeline(tmp_path, monkeypatch) as (execution, scenes, roles, renders):
            entered = asyncio.Event()
            stopped = asyncio.Event()

            async def pending(*args):
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    stopped.set()

            monkeypatch.setattr(runtime, "run", pending)
            accepted = await execution.submit("project", body())
            execution._schedule = Execution._schedule.__get__(execution)
            execution._schedule(accepted.id)
            await entered.wait()
            await execution.close()
            assert stopped.is_set()
            job = await execution.get(accepted.id)
            scene = scenes.get(accepted.request.request_id)
            assert job.status == "interrupted" and scene.status == "interrupted"
            assert not scene.gates["isaac_runtime"]
            assert scenes.artifact(scene.id, "scene-bundle.zip").is_file()
            assert (await execution.submit("project", body())).id == job.id
            assert roles == ["proposal", "review"] and not renders

    asyncio.run(exercise())


def test_reconcile_removes_investigator_and_reconstruction_auth_without_retry(tmp_path):
    scenes = service.Scenes(tmp_path)
    request = body()
    folder = scenes.folder(request.request_id)
    folder.mkdir(parents=True)
    run = SceneRun(
        id=request.request_id,
        project_id="project",
        request=request,
        status="running",
        stage="investigating",
        message="Fixture",
        created_at=now(),
        updated_at=now(),
    )
    scenes.save(run)
    auth_files = []
    for directory in ("investigation-1", "investigation-2", "attempt-1"):
        auth = folder / directory / ".codex-runtime" / "auth.json"
        auth.parent.mkdir(parents=True)
        auth.write_text("private-fixture")
        auth_files.append(auth)
    evidence = folder / "evidence.json"
    evidence.write_text('{"preserved":true}')
    other = tmp_path / "unrelated-auth.json"
    other.write_text("preserved-fixture")
    asyncio.run(scenes.reconcile())
    assert scenes.get(run.id).status == "interrupted"
    assert all(not auth.exists() for auth in auth_files)
    assert json.loads(evidence.read_text()) == {"preserved": True}
    assert other.read_text() == "preserved-fixture" and not scenes.tasks


@pytest.mark.parametrize("gate", ["source_evidence", "agent_review", "usd_structure"])
def test_media_does_not_override_a_missing_core_acceptance_gate(tmp_path, monkeypatch, gate):
    async def exercise():
        async with pipeline(tmp_path, monkeypatch) as (execution, scenes, roles, renders):
            original = runtime.run

            async def missing_gate(owner, run, *args):
                result = await original(owner, run, *args)
                run.gates[gate] = False
                return result

            monkeypatch.setattr(runtime, "run", missing_gate)
            job = await execution.submit("project", body())
            await execution.run(job.id)
            current = await execution.get(job.id)
            scene = scenes.get(job.request.request_id)
            assert current.status == "failed" and scene.gates[gate] is False
            assert len(renders) == 1
            assert "simulation/replay.mp4" in current.result.scene_artifacts

    asyncio.run(exercise())


@pytest.mark.parametrize("missing", ["environment_image", "episode_replay"])
def test_partial_render_media_cannot_publish_job_success(tmp_path, monkeypatch, missing):
    async def exercise():
        async with pipeline(tmp_path, monkeypatch) as (execution, scenes, roles, renders):
            original = runtime.run

            async def incomplete(owner, run, *args):
                result = await original(owner, run, *args)
                index = next(i for i, item in enumerate(result["media"]) if item["role"] == missing)
                result["media"].pop(index)
                return result

            monkeypatch.setattr(runtime, "run", incomplete)
            job = await execution.submit("project", body())
            await execution.run(job.id)
            current = await execution.get(job.id)
            scene = scenes.get(job.request.request_id)
            assert current.status == "failed" and len(renders) == 1
            assert scenes.artifact(scene.id, "scene-bundle.zip").is_file()
            assert all(scene.gates[name] for name in CORE_GATES)

    asyncio.run(exercise())


def test_runner_preflight_failure_stops_before_dataset_or_provider_work(tmp_path, monkeypatch):
    async def exercise():
        async with pipeline(tmp_path, monkeypatch) as (execution, scenes, roles, renders):
            preflight_stages = []
            data_calls = []

            async def unavailable(owner, run):
                preflight_stages.append((run.stage, (await execution.get(run.job_id)).stage))
                raise SceneError("Scene runner preflight is unavailable.")

            async def forbidden(*args):
                data_calls.append(True)
                pytest.fail("Preflight failure must stop before dataset investigation")

            monkeypatch.setattr(runtime, "preflight", unavailable)
            monkeypatch.setattr(dataset, "investigate", forbidden)
            job = await execution.submit("project", body())
            await execution.run(job.id)
            current = await execution.get(job.id)
            scene = scenes.get(job.request.request_id)
            assert preflight_stages == [("preparing", "preparing")]
            assert current.status == "failed" and scene.status == "failed"
            assert "Stopped during preparing" in current.error
            assert all(scene.gates[name] is False for name in CORE_GATES)
            assert not data_calls and not roles and not renders
            assert not scene.artifacts and not scene.media
            assert [event["stage"] for event in scene.events] == ["preparing", "failed"]
            assert not (scenes.folder(scene.id) / "dataset-catalog.json").exists()

    asyncio.run(exercise())
