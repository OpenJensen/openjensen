"""Scene requests share durable job admission, cancellation and artifact boundaries."""

import asyncio
import json
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient
from vla_platform.api import create_app
from vla_platform.contracts import IntakeRequest, Job, now
from vla_platform.datasets.inspect import profile
from vla_platform.execution import Execution
from vla_platform.scenes.contracts import SceneJobResult, SceneRequest, SceneRun
from vla_platform.scenes.service import Scenes
from vla_platform.settings import Settings
from vla_platform.storage import Storage
from vla_platform.submissions import SubmissionConflict

MODEL = "fixture-scene"


def options():
    return {
        "configured": True,
        "models": [
            {
                "id": MODEL,
                "label": "Fixture",
                "reasoning_efforts": ["medium", "high"],
                "default_reasoning_effort": "medium",
            }
        ],
        "default_model": MODEL,
    }


def source(project="project", ident="source"):
    request = IntakeRequest(repo_id="fixture/scene-dataset")
    raw = json.dumps(
        {
            "codebase_version": "v3.0",
            "total_episodes": 30,
            "total_frames": 4500,
            "fps": 30,
            "features": {
                "observation.images.front": {"dtype": "video", "shape": [1080, 1920, 3]},
                "observation.state": {"dtype": "float32", "shape": [6]},
                "action": {"dtype": "float32", "shape": [6]},
            },
        }
    ).encode()
    return Job(
        id=ident,
        project_id=project,
        status="succeeded",
        request=request,
        result=profile(raw, request, "a" * 40),
        created_at=now(),
        updated_at=now(),
    )


def request(**updates):
    return SceneRequest(request_id="b" * 32, source_job_id="source", **updates)


async def validate(_project, _request, _source):
    return None


@asynccontextmanager
async def owner(tmp_path):
    storage = Storage(tmp_path)
    await storage.initialize()
    execution = Execution(storage, Settings(data_dir=tmp_path))
    dispatched = []
    execution._schedule = dispatched.append
    execution.scenes.validate = validate
    execution.scenes.options = options
    try:
        yield execution, dispatched
    finally:
        await execution.close()
        await storage.close()


async def add_source(execution, project="project", ident="source"):
    await execution._accept(source(project, ident), None)
    return ident


def test_concurrent_scene_clicks_accept_one_job_and_freeze_default_model(tmp_path):
    async def exercise():
        async with owner(tmp_path) as (execution, dispatched):
            await add_source(execution)
            dispatched.clear()
            accepted = await asyncio.gather(
                *[execution.submit("project", request()) for _ in range(12)]
            )
            assert len({job.id for job in accepted}) == 1
            job = accepted[0]
            assert job.kind == "scene.generate" and job.request.model == MODEL
            assert dispatched == [job.id]
            assert [item.kind for item in await execution.list("project")].count(
                "scene.generate"
            ) == 1

            async def unavailable(*args):
                pytest.fail("An accepted retry must not revalidate source or provider setup")

            execution.scenes.validate = unavailable
            execution.scenes.options = lambda: {"configured": False}
            again = await execution.submit("project", request())
            assert again.id == job.id and again.request.model == MODEL
            with pytest.raises(SubmissionConflict):
                await execution.submit("project", request(reasoning_effort="high"))

    asyncio.run(exercise())


def test_scene_request_identity_cannot_be_reused_through_another_key_or_project(tmp_path):
    async def exercise():
        async with owner(tmp_path) as (execution, dispatched):
            await add_source(execution)
            await add_source(execution, "other", "other-source")
            dispatched.clear()
            first = await execution.submit("project", request(), idempotency_key="first-key")
            for project, payload in (
                ("project", request()),
                (
                    "other",
                    SceneRequest(request_id="b" * 32, source_job_id="other-source"),
                ),
            ):
                with pytest.raises(SubmissionConflict):
                    await execution.submit(project, payload, idempotency_key="another-key")
            assert dispatched == [first.id]

    asyncio.run(exercise())


def test_scene_source_project_is_checked_before_admission(tmp_path):
    async def exercise():
        async with owner(tmp_path) as (execution, dispatched):
            await add_source(execution)
            dispatched.clear()
            with pytest.raises(ValueError, match="this project"):
                await execution.submit("another", request())
            assert not dispatched and await execution.list("another") == []

    asyncio.run(exercise())


def test_scene_restart_returns_interrupted_job_without_dispatch(tmp_path):
    async def exercise():
        async with owner(tmp_path) as (execution, _):
            await add_source(execution)
            first = await execution.submit("project", request())
        async with owner(tmp_path) as (execution, dispatched):
            await execution.reconcile()
            again = await execution.submit("project", request())
            assert again.id == first.id and again.status == "interrupted"
            assert not dispatched

    asyncio.run(exercise())


def test_cancelled_scene_never_dispatches_a_later_worker(tmp_path):
    async def exercise():
        async with owner(tmp_path) as (execution, dispatched):
            await add_source(execution)
            job = await execution.submit("project", request())
            called = []

            async def forbidden(*args):
                called.append(True)

            execution.scenes.run_job = forbidden
            cancelled = await execution.cancel(job.id)
            assert cancelled.status == "cancelled"
            await execution.run(job.id)
            assert not called
            assert (await execution.get(job.id)).status == "cancelled"

    asyncio.run(exercise())


def test_scene_executor_uses_shared_job_and_does_not_overwrite_cancellation(tmp_path):
    async def exercise():
        async with owner(tmp_path) as (execution, _):
            await add_source(execution)
            job = await execution.submit("project", request())
            entered = asyncio.Event()
            cancelled = asyncio.Event()

            async def run_scene(current, dataset, explorer, supervisor):
                assert current.id == job.id and current.status == "running"
                assert dataset.id == "source" and supervisor is execution
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise

            execution.scenes.run_job = run_scene
            execution._schedule = Execution._schedule.__get__(execution)
            execution._schedule(job.id)
            await entered.wait()
            await execution.cancel(job.id)
            assert cancelled.is_set()
            assert (await execution.get(job.id)).status == "cancelled"

    asyncio.run(exercise())


def test_scene_executor_failures_are_terminal_and_generic_errors_are_redacted(tmp_path):
    async def exercise():
        async with owner(tmp_path) as (execution, _):
            await add_source(execution)
            job = await execution.submit("project", request())

            async def failed(*args):
                raise RuntimeError("private-fixture-credential")

            execution.scenes.run_job = failed
            await execution.run(job.id)
            current = await execution.get(job.id)
            assert current.status == "failed"
            assert "private-fixture" not in current.model_dump_json()

    asyncio.run(exercise())


def test_scene_completion_can_require_review_without_losing_job_execution_success(tmp_path):
    async def exercise():
        async with owner(tmp_path) as (execution, _):
            await add_source(execution)
            job = await execution.submit("project", request())

            async def complete(current, dataset, explorer, supervisor):
                async with supervisor.lock:
                    saved = await supervisor.get(current.id)
                    saved.result = SceneJobResult(
                        scene_id=current.request.request_id,
                        validation_status="needs_review",
                        scene_artifacts=["render/front.png", "render/replay.mp4"],
                    )
                    saved.status = "succeeded"
                    await supervisor.save(saved)

            execution.scenes.run_job = complete
            await execution.run(job.id)
            current = await execution.get(job.id)
            assert current.status == "succeeded"
            assert current.result.validation_status == "needs_review"
            assert current.result.scene_artifacts[-1] == "render/replay.mp4"

    asyncio.run(exercise())


def test_scene_job_api_uses_standard_history_and_idempotency(tmp_path, monkeypatch):
    dispatched = []
    monkeypatch.setattr(Execution, "_schedule", lambda self, job_id: dispatched.append(job_id))
    monkeypatch.setattr(Scenes, "options", lambda self: options())
    monkeypatch.setattr(
        Scenes,
        "validate",
        lambda self, project, payload, dataset: validate(project, payload, dataset),
    )
    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as client:
        project = client.post("/api/v1/projects", json={"name": "Scene job fixture"}).json()["id"]
        client.portal.call(add_source, app.state.execution, project)
        dispatched.clear()
        endpoint = f"/api/v1/projects/{project}/scene-jobs"
        body = request().model_dump(mode="json")
        first = client.post(endpoint, json=body)
        assert first.status_code == 202
        job = first.json()
        assert first.headers["Idempotency-Key"] == body["request_id"]
        assert job["kind"] == "scene.generate" and job["request"]["episode_index"] is None
        assert job["request"]["model"] == MODEL
        again = client.post(endpoint, json=body)
        assert again.status_code == 202 and again.json()["id"] == job["id"]
        recovered = client.get(
            f"/api/v1/projects/{project}/submissions/{body['request_id']}",
            params={"operation": "scene.generate"},
        )
        assert recovered.status_code == 200 and recovered.json()["id"] == job["id"]
        assert recovered.headers["Idempotency-Key"] == body["request_id"]
        assert dispatched == [job["id"]]
        history = client.get(f"/api/v1/projects/{project}/jobs").json()
        assert any(item["id"] == job["id"] for item in history)
        assert client.get(f"/api/v1/jobs/{job['id']}/scene").status_code == 404
        assert client.post(f"/api/v1/projects/{project}/scenes", json=body).status_code == 409
        changed = dict(body, reasoning_effort="high")
        assert client.post(endpoint, json=changed).status_code == 409
        assert (
            client.post(
                endpoint,
                json=body,
                headers=[("Idempotency-Key", "one"), ("Idempotency-Key", "two")],
            ).status_code
            == 422
        )


def test_bound_scene_media_is_inline_ranged_and_allowlisted(tmp_path, monkeypatch):
    monkeypatch.setattr(Execution, "_schedule", lambda *args: None)
    monkeypatch.setattr(Scenes, "options", lambda self: options())
    monkeypatch.setattr(
        Scenes,
        "validate",
        lambda self, project, payload, dataset: validate(project, payload, dataset),
    )
    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as client:
        project = client.post("/api/v1/projects", json={"name": "Media fixture"}).json()["id"]
        client.portal.call(add_source, app.state.execution, project)
        job = client.post(
            f"/api/v1/projects/{project}/scene-jobs", json=request().model_dump(mode="json")
        ).json()
        ident = job["request"]["request_id"]
        folder = app.state.scenes.folder(ident)
        folder.mkdir(parents=True)
        (folder / "environment.png").write_bytes(b"png-fixture")
        (folder / "replay.mp4").write_bytes(b"video-fixture-content")
        run = SceneRun(
            id=ident,
            job_id=job["id"],
            project_id=project,
            request=SceneRequest.model_validate(job["request"]),
            status="needs_review",
            stage="complete",
            created_at=now(),
            updated_at=now(),
            message="Rendered fixture",
            artifacts=["environment.png", "replay.mp4"],
            events=[
                {"at": now(), "stage": "rendering", "message": "Rendered fixture", "attempt": 1}
            ],
        )
        app.state.scenes.save(run)
        base = f"/api/v1/jobs/{job['id']}/scene"
        assert client.get(base).json()["job_id"] == job["id"]
        events_url = f"/api/v1/jobs/{job['id']}/events"
        events = client.get(events_url).json()
        assert events[0]["sequence"] == 1 and events[0]["stage"] == "rendering"
        assert client.get(events_url, params={"after": 1}).json() == []
        image = client.get(base + "/artifact", params={"name": "environment.png"})
        assert image.status_code == 200 and image.headers["content-type"] == "image/png"
        assert image.headers["content-disposition"].startswith("inline")
        video = client.get(
            base + "/artifact", params={"name": "replay.mp4"}, headers={"Range": "bytes=0-4"}
        )
        assert video.status_code == 206 and video.content == b"video"
        assert video.headers["content-type"] == "video/mp4"
        assert client.get(base + "/artifact", params={"name": "../run.json"}).status_code == 404
        run.job_id = "not-the-bound-job"
        app.state.scenes.save(run)
        assert client.get(base).status_code == 404
