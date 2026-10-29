"""Exercise the scene Job pipeline with provider, builder and renderer boundaries mocked."""

import asyncio
import json
import threading
import zipfile
from contextlib import asynccontextmanager

import pytest
from test_scene_generation import body, setup, source
from vla_platform.contracts import now
from vla_platform.execution import Execution
from vla_platform.scenes import dataset, runtime, saved, service
from vla_platform.scenes.agent import SceneError
from vla_platform.scenes.contracts import Proposal, SceneRequest, SceneRun
from vla_platform.settings import Settings
from vla_platform.storage import Storage

CORE_GATES = ("source_evidence", "agent_review", "usd_structure", "isaac_runtime")


@asynccontextmanager
async def pipeline(tmp_path, monkeypatch):
    scenes, roles = setup(tmp_path, monkeypatch)
    scene_options = scenes.options
    scenes.options = lambda: {**scene_options(), "simulation_available": True}
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
            "source": dict(run.source),
            "episodes": [
                {
                    "episode_index": index,
                    "frame_count": 150,
                    "cameras": [
                        {
                            "key": "observation.images.front",
                            "url": (
                                "https://huggingface.co/datasets/"
                                f"{dataset_job.result.repo_id}/resolve/{dataset_job.result.revision}/"
                                f"videos/{index}.mp4"
                            ),
                        }
                    ],
                }
                for index in range(30)
            ],
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
            catalog_sha256=service.digest(folder / "dataset-catalog.json"),
            trajectory_sha256=service.digest(folder / "trajectory.json"),
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


def retry_request(parent, **updates):
    values = parent.request.model_dump(mode="json")
    values.update(request_id="d" * 32, source_scene_id=parent.id)
    values.update(updates)
    return SceneRequest.model_validate(values)


async def failed_reconstruction(execution, scenes, monkeypatch):
    original = runtime.run

    async def failed(*args):
        raise SceneError("Fixture simulation handoff failed.")

    monkeypatch.setattr(runtime, "run", failed)
    accepted = await execution.submit("project", body())
    await execution.run(accepted.id)
    parent = scenes.get(accepted.request.request_id)
    assert parent.status == "failed"
    assert all(parent.gates[name] for name in CORE_GATES[:3])
    assert not parent.gates["isaac_runtime"]
    return parent, original


def inventory(folder):
    return {
        str(path.relative_to(folder)): service.digest(path)
        for path in folder.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("parent_status", ["failed", "interrupted"])
def test_saved_scene_retry_creates_new_job_without_repeating_provider_or_builder(
    tmp_path,
    monkeypatch,
    parent_status,
):
    async def exercise():
        async with pipeline(tmp_path, monkeypatch) as (execution, scenes, roles, renders):
            investigated, built = [], []
            original_investigate = dataset.investigate
            original_build = service.command

            async def investigate(*args):
                investigated.append(True)
                return await original_investigate(*args)

            async def build(*args, **kwargs):
                built.append(True)
                return await original_build(*args, **kwargs)

            monkeypatch.setattr(dataset, "investigate", investigate)
            monkeypatch.setattr(service, "command", build)
            parent, original_render = await failed_reconstruction(execution, scenes, monkeypatch)
            parent.status = parent_status
            parent.source["resource_cleanup"] = {"status": "verified"}
            scenes.save(parent)
            original_job = await execution.get(parent.job_id)
            original_job.status = parent_status
            await execution.save(original_job)
            parent_files = inventory(scenes.folder(parent.id))
            assert roles == ["proposal", "review"] and investigated == [True] and built == [True]
            monkeypatch.setattr(runtime, "run", original_render)
            monkeypatch.setattr(
                scenes,
                "options",
                lambda: {
                    "configured": False,
                    "simulation_available": True,
                    "models": [],
                },
            )
            child = await execution.submit("project", retry_request(parent))
            await execution.run(child.id)
            current = await execution.get(child.id)
            resumed = scenes.get(child.request.request_id)
            assert child.id != parent.job_id and resumed.id != parent.id
            assert current.status == "succeeded"
            assert resumed.source["resumed_from_scene_id"] == parent.id
            assert resumed.agent == parent.agent and resumed.usage == {}
            assert all(resumed.gates[key] for key in CORE_GATES)
            assert not resumed.gates["calibrated_replay"] and not resumed.gates["visual_match"]
            assert len(renders) == 1 and roles == ["proposal", "review"]
            assert investigated == [True] and built == [True]
            assert inventory(scenes.folder(parent.id)) == parent_files
            assert scenes.get(parent.id).status == parent_status
            assert (await execution.get(parent.job_id)).status == parent_status
            assert resumed.source["resource_cleanup"] == "unverified"
            assert service.digest(
                scenes.folder(resumed.id) / "attempt-1/bundle/scene.usda"
            ) == service.digest(scenes.folder(parent.id) / "attempt-1/bundle/scene.usda")

    asyncio.run(exercise())


@pytest.mark.parametrize("tamper", ["scene.usda", "trajectory.json", "missing_asset"])
def test_saved_scene_retry_rejects_changed_or_missing_inputs_before_admission(
    tmp_path,
    monkeypatch,
    tamper,
):
    async def exercise():
        async with pipeline(tmp_path, monkeypatch) as (execution, scenes, roles, renders):
            parent, _ = await failed_reconstruction(execution, scenes, monkeypatch)
            folder = scenes.folder(parent.id)
            if tamper == "scene.usda":
                (folder / "attempt-1/bundle/scene.usda").write_text("changed-fixture")
            elif tamper == "trajectory.json":
                (folder / "trajectory.json").write_text("{}")
            else:
                (folder / "attempt-1/bundle/frame-0.jpg").unlink()
            previous_count = len(await execution.list("project"))
            with pytest.raises(SceneError, match="saved|Saved|scene"):
                await execution.submit("project", retry_request(parent))
            assert len(await execution.list("project")) == previous_count
            assert roles == ["proposal", "review"] and not renders
            assert not scenes.folder("d" * 32).exists()

    asyncio.run(exercise())


def test_saved_scene_retry_is_project_scoped(tmp_path, monkeypatch):
    async def exercise():
        async with pipeline(tmp_path, monkeypatch) as (execution, scenes, roles, renders):
            parent, _ = await failed_reconstruction(execution, scenes, monkeypatch)
            other = source("other").model_copy(update={"id": "other-source"})
            await execution._accept(other, None)
            previous = inventory(scenes.folder(parent.id))
            with pytest.raises(SceneError, match="project"):
                await execution.submit("other", retry_request(parent, source_job_id="other-source"))
            assert inventory(scenes.folder(parent.id)) == previous
            assert roles == ["proposal", "review"] and not renders

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "updates",
    [
        {"model": "gpt-6-luna"},
        {"reasoning_effort": "high"},
    ],
)
def test_saved_scene_retry_keeps_the_parent_agent_settings(tmp_path, monkeypatch, updates):
    async def exercise():
        async with pipeline(tmp_path, monkeypatch) as (execution, scenes, roles, renders):
            parent, _ = await failed_reconstruction(execution, scenes, monkeypatch)
            with pytest.raises(SceneError, match="matching model"):
                await execution.submit("project", retry_request(parent, **updates))
            assert roles == ["proposal", "review"] and not renders

    asyncio.run(exercise())


def test_cancelling_saved_scene_retry_preserves_its_failed_parent(tmp_path, monkeypatch):
    async def exercise():
        async with pipeline(tmp_path, monkeypatch) as (execution, scenes, roles, renders):
            parent, _ = await failed_reconstruction(execution, scenes, monkeypatch)
            previous = inventory(scenes.folder(parent.id))
            entered = asyncio.Event()

            async def waiting(*args):
                entered.set()
                await asyncio.Event().wait()

            monkeypatch.setattr(runtime, "run", waiting)
            child = await execution.submit("project", retry_request(parent))
            execution._schedule = Execution._schedule.__get__(execution)
            execution._schedule(child.id)
            await entered.wait()
            await execution.cancel(child.id)
            assert (await execution.get(child.id)).status == "cancelled"
            assert scenes.get(child.request.request_id).status == "cancelled"
            assert scenes.get(parent.id).status == "failed"
            assert inventory(scenes.folder(parent.id)) == previous
            assert roles == ["proposal", "review"] and not renders

    asyncio.run(exercise())


def test_terminal_pipeline_removes_only_generated_codex_homes_and_retains_role_results(
    tmp_path,
    monkeypatch,
):
    async def exercise():
        async with pipeline(tmp_path, monkeypatch) as (execution, scenes, roles, renders):
            original = service.run_role

            async def role(work, *args, **kwargs):
                home = work / ".codex-runtime"
                home.mkdir(exist_ok=True)
                (home / "bootstrap-cache").write_text("fixture")
                return await original(work, *args, **kwargs)

            monkeypatch.setattr(service, "run_role", role)
            parent, _ = await failed_reconstruction(execution, scenes, monkeypatch)
            work = scenes.folder(parent.id) / "attempt-1"
            assert not (work / ".codex-runtime").exists()
            assert (work / "proposal.json").is_file() and (work / "review.json").is_file()
            assert (work / "bundle/scene.usda").is_file()

    asyncio.run(exercise())


def test_cancellation_drains_owned_saved_scene_copy_before_returning(tmp_path, monkeypatch):
    async def exercise():
        async with pipeline(tmp_path, monkeypatch) as (execution, scenes, roles, renders):
            parent, _ = await failed_reconstruction(execution, scenes, monkeypatch)
            previous = inventory(scenes.folder(parent.id))
            entered, release = threading.Event(), threading.Event()
            original = saved.restore

            def blocked_copy(*args):
                entered.set()
                if not release.wait(3):
                    raise SceneError("Fixture copy wait expired.")
                return original(*args)

            monkeypatch.setattr(saved, "restore", blocked_copy)
            child = await execution.submit("project", retry_request(parent))
            execution._schedule = Execution._schedule.__get__(execution)
            execution._schedule(child.id)
            assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 2), 3)
            cancellation = asyncio.create_task(execution.cancel(child.id))

            async def cancelled():
                while (await execution.get(child.id)).status != "cancelled":
                    await asyncio.sleep(0)

            try:
                await asyncio.wait_for(cancelled(), 2)
                assert not cancellation.done()
            finally:
                release.set()
            await asyncio.wait_for(cancellation, 3)
            assert scenes.get(child.request.request_id).status == "cancelled"
            assert inventory(scenes.folder(parent.id)) == previous
            assert roles == ["proposal", "review"] and not renders

    asyncio.run(exercise())


@pytest.mark.parametrize("reported", ["cluster_removed", "unverified", None])
def test_render_completion_preserves_resource_cleanup_observation(tmp_path, monkeypatch, reported):
    async def exercise():
        async with pipeline(tmp_path, monkeypatch) as (execution, scenes, roles, renders):
            original = runtime.run

            async def rendered(owner, run, *args):
                result = await original(owner, run, *args)
                raw_path = owner.folder(run.id) / "simulation/result.json"
                raw = json.loads(raw_path.read_text())
                raw["resource_cleanup"] = "unverified"
                service.write(raw_path, raw)
                if reported is not None:
                    result["resource_cleanup"] = reported
                return result

            monkeypatch.setattr(runtime, "run", rendered)
            accepted = await execution.submit("project", body())
            await execution.run(accepted.id)
            job = await execution.get(accepted.id)
            scene = scenes.get(accepted.request.request_id)
            assert job.status == "succeeded" and job.result.validation_status == "needs_review"
            assert scene.stage == "complete"
            assert scene.source["resource_cleanup"] == (reported or "unverified")
            assert "simulation/result.json" in job.result.scene_artifacts
            original_receipt = json.loads(
                scenes.artifact(scene.id, "simulation/result.json").read_text()
            )
            assert original_receipt["resource_cleanup"] == "unverified"

    asyncio.run(exercise())
