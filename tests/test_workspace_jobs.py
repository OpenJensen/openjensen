"""Shared workload history and admission without executing workers or providers."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from vla_platform.api import create_app
from vla_platform.contracts import IntakeRequest, Job, JobCancellation, now
from vla_platform.execution import Execution
from vla_platform.scenes.contracts import SceneJobResult, SceneRequest, SceneRun
from vla_platform.settings import Settings
from vla_platform.workspace_jobs import CATALOG, OPERATIONS, core_job


@pytest.fixture
def client(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    # Admission persists records, but no fixture ever schedules worker execution.
    monkeypatch.setattr(Execution, "_schedule", lambda self, ident: None)
    app = create_app(Settings(data_dir=tmp_path / "workspace", local_root=source))
    with TestClient(app) as value:
        yield value


def project(client, name="Project"):
    return client.post("/api/v1/projects", json={"name": name}).json()["id"]


def submission(path):
    return {
        "operation": "dataset.inspect",
        "inputs": {"source": "local", "path": str(path), "snapshot_for_training": True},
    }


def accept(client, project_id, body, key="fixture-job"):
    return client.post(
        f"/api/v1/projects/{project_id}/work-jobs", json=body, headers={"Idempotency-Key": key}
    )


def test_history_and_cancel_are_project_scoped(client):
    first, second = project(client), project(client, "Other project")
    accepted = accept(client, first, submission(client.app.state.execution.settings.local_root))
    assert accepted.status_code == 202, accepted.text
    row = accepted.json()
    assert row["id"] == "core:" + row["source_id"]
    assert row["operation"] == "dataset.inspect" and row["execution"]["infrastructure"] == "local"
    assert client.get(f"/api/v1/projects/{second}/work-jobs").json() == []
    for suffix in ("", "/cancel"):
        endpoint = f"/api/v1/projects/{second}/work-jobs/{row['id']}{suffix}"
        response = client.post(endpoint) if suffix else client.get(endpoint)
        assert response.status_code == 404
    owner = f"/api/v1/projects/{first}/work-jobs/{row['id']}"
    assert client.get(owner).json()["status"] == "queued"
    cancelled = client.post(owner + "/cancel")
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["status"] == "cancelled"
    assert not cancelled.json()["can_cancel"]
    assert client.post(owner + "/cancel").status_code == 409


def test_shared_submission_is_idempotent_and_rejects_changed_inputs(client):
    ident = project(client)
    body = submission(client.app.state.execution.settings.local_root)
    first = accept(client, ident, body)
    second = accept(client, ident, body)
    assert first.status_code == second.status_code == 202
    assert first.json()["id"] == second.json()["id"]
    changed = json.loads(json.dumps(body))
    changed["inputs"]["revision"] = "different"
    conflict = accept(client, ident, changed)
    assert conflict.status_code == 409, conflict.text
    assert len(client.get(f"/api/v1/projects/{ident}/work-jobs").json()) == 1
    assert len(client.get(f"/api/v1/projects/{ident}/jobs").json()) == 1


def test_idempotency_key_is_required_and_duplicate_headers_are_rejected(client):
    ident = project(client)
    endpoint = f"/api/v1/projects/{ident}/work-jobs"
    body = submission(client.app.state.execution.settings.local_root)
    assert client.post(endpoint, json=body).status_code == 422
    assert (
        client.post(
            endpoint, json=body, headers=[("Idempotency-Key", "one"), ("Idempotency-Key", "two")]
        ).status_code
        == 422
    )
    assert client.get(f"/api/v1/projects/{ident}/jobs").json() == []


@pytest.mark.parametrize(
    "change",
    [
        {"operation": "module.install"},
        {"operation": "application.update"},
        {"target_id": "remote-arbitrary"},
        {"inputs": {"operation": "policy.run"}},
        {"inputs": {"source": "local", "path": "/example", "command": "untrusted"}},
    ],
)
def test_invalid_or_maintenance_work_never_enters_job_history(client, change):
    ident = project(client)
    body = {**submission(client.app.state.execution.settings.local_root), **change}
    response = accept(client, ident, body)
    assert response.status_code == 422, response.text
    assert client.get(f"/api/v1/projects/{ident}/jobs").json() == []


def test_operation_catalog_excludes_maintenance_and_has_unique_operations(client):
    response = client.get("/api/v1/job-operations")
    assert response.status_code == 200
    operations = response.json()
    ids = {row["id"] for row in operations}
    assert len(ids) == len(operations)
    assert {
        "dataset.inspect",
        "dataset.augment",
        "scene.generate",
        "decision.score",
        "agent.control",
        "policy.finetune",
    }.issubset(ids)
    assert not any("install" in ident or "update" in ident for ident in ids)
    assert ids == set(CATALOG) and len(OPERATIONS) == len(CATALOG)


def test_history_includes_existing_core_scene_agent_and_dataset_records(client, monkeypatch):
    ident = project(client)
    core = accept(client, ident, submission(client.app.state.execution.settings.local_root)).json()
    scenes = client.app.state.scenes
    run = SceneRun(
        id="a" * 32,
        project_id=ident,
        request=SceneRequest(request_id="a" * 32, source_job_id=core["source_id"]),
        status="needs_simulation",
        stage="structure_validated",
        created_at=now(),
        updated_at=now(),
        message="Scene awaits simulator validation",
        gates={"usd_structure": True, "isaac_runtime": False},
    )
    scenes.folder(run.id).mkdir(parents=True)
    scenes.save(run)
    client.app.state.dataset_library.begin(ident, "Unfinished import")
    session = SimpleNamespace(
        id="session-1",
        project_id=ident,
        provider_id="provider",
        task="Move an object",
        target_kind="replay",
        status="waiting_for_acceptance",
        created_at=now(),
        updated_at=now(),
        message="Review proposed movement",
        mode="supervised",
        step_count=0,
        calls_used=1,
        task_success_verified=False,
        teaching_job_id=core["source_id"],
    )

    async def sessions(project_id):
        return [session] if project_id == ident else []

    monkeypatch.setattr(client.app.state.agent_control, "list", sessions)
    rows = client.get(f"/api/v1/projects/{ident}/work-jobs").json()
    assert {row["id"] for row in rows} == {
        core["id"],
        "scene:" + run.id,
        "agent:session-1",
        "dataset:" + client.app.state.dataset_library.operation_records(ident)[0]["id"],
    }
    scene = next(row for row in rows if row["operation"] == "scene.generate")
    assert scene["status"] == "waiting" and scene["source_status"] == "needs_simulation"
    assert not scene["can_cancel"]
    scene_detail = client.get(f"/api/v1/projects/{ident}/work-jobs/{scene['id']}").json()
    assert scene_detail["result"]["gates"]["isaac_runtime"] is False
    agent = next(row for row in rows if row["operation"] == "agent.control")
    assert agent["status"] == "waiting" and agent["can_cancel"]
    assert agent["parent_ids"] == [core["id"]]
    assert rows == sorted(rows, key=lambda row: (row["created_at"], row["id"]), reverse=True)


def test_legacy_job_without_execution_binding_remains_readable():
    record = Job(
        id="existing",
        project_id="project",
        request=IntakeRequest(repo_id="fixture/data"),
        created_at=now(),
        updated_at=now(),
        status="failed",
        error="A saved worker failure",
    )
    assert "execution" not in record.model_dump()
    projected = core_job(record)
    assert projected.id == "core:existing" and projected.execution is None
    assert projected.message == record.error and not projected.can_cancel


def test_unknown_project_and_unknown_job_are_explicit(client):
    ident = project(client)
    assert client.get("/api/v1/projects/missing/work-jobs").status_code == 404
    assert client.get(f"/api/v1/projects/{ident}/work-jobs/core:missing").status_code == 404
    assert client.post(f"/api/v1/projects/{ident}/work-jobs/core:missing/cancel").status_code == 404


def test_scene_repeated_submission_retains_owned_identity(client, monkeypatch):
    from test_scene_generation import source as scene_source

    ident = project(client)
    dataset = scene_source(ident)
    client.portal.call(client.app.state.execution._accept, dataset, None)
    default = {"model": "fixture-model"}
    models = [
        {"id": name, "reasoning_efforts": ["medium"]}
        for name in ("fixture-model", "another-fixture-model")
    ]
    monkeypatch.setattr(
        client.app.state.scenes,
        "options",
        lambda: {
            "configured": True,
            "models": models,
            "default_model": default["model"],
        },
    )
    validated = []
    validate = client.app.state.scenes.validate

    async def checked(project_id, body, source):
        validated.append(body.request_id)
        return await validate(project_id, body, source)

    monkeypatch.setattr(client.app.state.scenes, "validate", checked)
    body = {"operation": "scene.generate", "inputs": {"source_job_id": dataset.id}}
    one = accept(client, ident, body, "scene-request")
    assert one.status_code == 202, one.text
    default["model"] = "another-fixture-model"
    two = accept(client, ident, body, "scene-request")
    assert two.status_code == 202, two.text
    assert one.json()["id"] == two.json()["id"]
    assert one.json()["id"].startswith("core:")
    assert one.json()["inputs"]["model"] == two.json()["inputs"]["model"] == "fixture-model"
    assert len(validated) == 1
    wrong = accept(client, ident, {**body, "target_id": "unregistered"}, "other")
    assert wrong.status_code == 422
    changed = accept(
        client, ident, {**body, "inputs": {**body["inputs"], "max_attempts": 3}}, "scene-request"
    )
    assert changed.status_code == 409
    history = client.get(f"/api/v1/projects/{ident}/work-jobs").json()
    assert len(history) == 2
    assert sum(row["operation"] == "scene.generate" for row in history) == 1


def test_history_pagination_filters_and_summary_payloads_are_bounded(client):
    ident = project(client)
    accepted = []
    for index in range(3):
        response = accept(
            client,
            ident,
            submission(client.app.state.execution.settings.local_root),
            f"history-{index}",
        )
        assert response.status_code == 202
        accepted.append(response.json())
    run = SceneRun(
        id="b" * 32,
        project_id=ident,
        request=SceneRequest(request_id="b" * 32, source_job_id=accepted[0]["source_id"]),
        status="needs_review",
        stage="review",
        created_at=now(),
        updated_at=now(),
        message="Review result",
        gates={"visual_match": False},
        events=[{"at": now(), "stage": "review", "message": "Scene proposal ready"}],
    )
    client.app.state.scenes.folder(run.id).mkdir(parents=True)
    client.app.state.scenes.save(run)
    endpoint = f"/api/v1/projects/{ident}/work-jobs"
    first = client.get(endpoint, params={"limit": 2})
    second = client.get(endpoint, params={"limit": 2, "offset": 2})
    assert first.headers["x-total-count"] == second.headers["x-total-count"] == "4"
    assert first.headers["x-next-offset"] == "2" and "x-next-offset" not in second.headers
    first_ids, second_ids = (
        {row["id"] for row in first.json()},
        {row["id"] for row in second.json()},
    )
    assert len(first_ids) == len(second_ids) == 2 and not first_ids.intersection(second_ids)
    assert len(first_ids.union(second_ids)) == 4
    for row in first.json() + second.json():
        assert row["inputs"] == {} and row["result"] is None
        assert row["outputs"] == [] and row["events"] == []
    scene = client.get(endpoint, params={"module": "real-to-sim", "q": "REAL TO SIM"})
    assert scene.headers["x-total-count"] == "1" and scene.json()[0]["id"] == "scene:" + run.id
    assert client.get(endpoint, params={"module": "augmentation"}).json() == []
    assert len(client.get(endpoint, params={"status": "active"}).json()) == 3
    assert len(client.get(endpoint, params={"status": "queued"}).json()) == 3
    assert len(client.get(endpoint, params={"status": "waiting"}).json()) == 1
    detail = client.get(endpoint + "/scene:" + run.id).json()
    assert detail["inputs"]["source_job_id"] == accepted[0]["source_id"]
    assert detail["events"] and detail["result"]["gates"]["visual_match"] is False
    client.post(endpoint + "/" + accepted[0]["id"] + "/cancel")
    assert len(client.get(endpoint, params={"status": "cancelled"}).json()) == 1
    assert len(client.get(endpoint, params={"status": "active"}).json()) == 2


@pytest.mark.parametrize(
    "query",
    [
        {"limit": 201},
        {"limit": 0},
        {"offset": -1},
        {"offset": 1000001},
        {"q": "x" * 201},
        {"status": "unknown"},
    ],
)
def test_job_history_rejects_unbounded_or_invalid_queries(client, query):
    ident = project(client)
    assert client.get(f"/api/v1/projects/{ident}/work-jobs", params=query).status_code == 422


@pytest.mark.parametrize("validation_status", ["needs_review", "needs_simulation"])
@pytest.mark.parametrize("source_status", ["failed", "succeeded"])
def test_partial_core_scene_results_need_attention_and_keep_artifacts(
    validation_status, source_status
):
    job = Job(
        id="scene-job",
        project_id="project",
        kind="scene.generate",
        request=SceneRequest(request_id="a" * 32, source_job_id="dataset-job"),
        status=source_status,
        stage="validation",
        created_at=now(),
        updated_at=now(),
        result=SceneJobResult(
            scene_id="a" * 32,
            validation_status=validation_status,
            scene_artifacts=["scene.usda", "evidence.json"],
        ),
    )
    row = core_job(job)
    assert row.status == "waiting" and row.source_status == source_status
    assert row.result["validation_status"] == validation_status
    assert [output.label for output in row.outputs] == ["scene.usda", "evidence.json"]
    assert all("/jobs/scene-job/scene/artifact?name=" in output.url for output in row.outputs)
    assert row.parent_ids == ["core:dataset-job"] and not row.can_cancel


def test_fully_validated_completed_core_scene_remains_completed():
    job = Job(
        id="scene-job",
        project_id="project",
        kind="scene.generate",
        request=SceneRequest(request_id="a" * 32, source_job_id="dataset-job"),
        status="succeeded",
        stage="complete",
        created_at=now(),
        updated_at=now(),
        result=SceneJobResult(
            scene_id="a" * 32, validation_status="needs_review", scene_artifacts=["scene.usda"]
        ),
    )
    assert core_job(job).status == "succeeded"
    assert core_job(job).outputs


@pytest.mark.parametrize(
    "cleanup,expected", [(None, "cancelled"), ("pending", "cancelling"), ("failed", "waiting")]
)
def test_scene_attention_projection_preserves_cancellation_priority(cleanup, expected):
    cancellation = JobCancellation(requested_at=now(), cleanup=cleanup) if cleanup else None
    job = Job(
        id="scene-job",
        project_id="project",
        kind="scene.generate",
        request=SceneRequest(request_id="a" * 32, source_job_id="dataset-job"),
        status="cancelled",
        stage="validation",
        cancellation=cancellation,
        created_at=now(),
        updated_at=now(),
        result=SceneJobResult(
            scene_id="a" * 32,
            validation_status="needs_simulation",
            scene_artifacts=["evidence.json"],
        ),
    )
    assert core_job(job).status == expected
    assert core_job(job).outputs


def test_scene_still_needing_simulation_never_appears_fully_validated():
    job = Job(
        id="scene-job",
        project_id="project",
        kind="scene.generate",
        request=SceneRequest(request_id="a" * 32, source_job_id="dataset-job"),
        status="succeeded",
        stage="complete",
        created_at=now(),
        updated_at=now(),
        result=SceneJobResult(
            scene_id="a" * 32, validation_status="needs_simulation", scene_artifacts=["scene.usda"]
        ),
    )
    assert core_job(job).status == "waiting"
    assert core_job(job).source_status == "succeeded"
    assert core_job(job).outputs


def saved_scene_retry_fixture(client):
    ident = project(client)
    dataset = accept(
        client, ident, submission(client.app.state.execution.settings.local_root)
    ).json()
    parent_request = SceneRequest(request_id="e" * 32, source_job_id=dataset["source_id"])
    parent = Job(
        id="parent-scene-job",
        project_id=ident,
        kind="scene.generate",
        request=parent_request,
        status="failed",
        stage="simulation",
        created_at=now(),
        updated_at=now(),
        result=SceneJobResult(
            scene_id=parent_request.request_id,
            validation_status="failed",
            scene_artifacts=["scene.usda"],
        ),
    )
    client.portal.call(client.app.state.execution._accept, parent, None)
    run = SceneRun(
        id=parent_request.request_id,
        job_id=parent.id,
        project_id=ident,
        request=parent_request,
        status="failed",
        stage="simulation",
        created_at=now(),
        updated_at=now(),
        message="Previous simulation failed",
        artifacts=["scene.usda"],
    )
    folder = client.app.state.scenes.folder(run.id)
    folder.mkdir(parents=True)
    (folder / "scene.usda").write_text("#usda 1.0\n")
    client.app.state.scenes.save(run)
    child_request = SceneRequest(
        request_id="f" * 32, source_job_id=dataset["source_id"], source_scene_id=run.id
    )
    child = Job(
        id="child-scene-job",
        project_id=ident,
        kind="scene.generate",
        request=child_request,
        status="queued",
        created_at=now(),
        updated_at=now(),
    )
    client.portal.call(client.app.state.execution._accept, child, None)
    return ident, dataset, parent, child, run


def test_saved_scene_lineage_resolves_hidden_parent_and_preserves_artifacts(client, monkeypatch):
    ident, dataset, parent, child, run = saved_scene_retry_fixture(client)
    rows = client.get(f"/api/v1/projects/{ident}/work-jobs").json()
    assert "scene:" + run.id not in {row["id"] for row in rows}
    projected = next(row for row in rows if row["id"] == "core:" + child.id)
    assert projected["parent_ids"] == [dataset["id"], "scene:" + run.id]

    def forbidden(*args):
        raise AssertionError("A direct parent lookup must not scan job history")

    monkeypatch.setattr(client.app.state.scenes, "list", forbidden)
    endpoint = f"/api/v1/projects/{ident}/work-jobs/scene:{run.id}"
    detail = client.get(endpoint)
    assert detail.status_code == 200 and detail.json()["id"] == "scene:" + run.id
    assert detail.json()["outputs"][0]["label"] == "scene.usda"
    artifact = client.get(detail.json()["outputs"][0]["url"])
    assert artifact.status_code == 200 and artifact.text.startswith("#usda")
    other = project(client, "Other project")
    assert client.get(f"/api/v1/projects/{other}/work-jobs/scene:{run.id}").status_code == 404


def test_scene_alias_diagnostics_use_the_scoped_core_owner(client):
    import io
    import zipfile

    ident, dataset, parent, child, run = saved_scene_retry_fixture(client)
    store = client.app.state.job_observability
    store.record(parent.id, "Parent worker output", kind="log", stream="stderr")
    store.failure(
        parent.id,
        stage="simulation",
        message="Parent runtime failure",
        code="WORKER_EXIT",
        exit_code=1,
    )
    endpoint = f"/api/v1/projects/{ident}/work-jobs/scene:{run.id}"
    logs = client.get(endpoint + "/logs").json()
    assert any(row["message"] == "Parent worker output" for row in logs["records"])
    assert all(row["job_id"] == "core:" + parent.id for row in logs["records"])
    assert client.get(endpoint + "/failure").json()["exit_code"] == 1
    archive = client.get(endpoint + "/diagnostics.zip")
    assert archive.status_code == 200
    with zipfile.ZipFile(io.BytesIO(archive.content)) as exported:
        assert json.loads(exported.read("job.json"))["id"] == "core:" + parent.id
        assert b"Parent worker output" in exported.read("diagnostics.jsonl")
    # Canonical diagnostic identity never changes the alias used for navigation.
    assert client.get(endpoint).json()["id"] == "scene:" + run.id
    other = project(client, "Other project")
    for suffix in ("logs", "failure", "diagnostics.zip"):
        assert (
            client.get(f"/api/v1/projects/{other}/work-jobs/scene:{run.id}/{suffix}").status_code
            == 404
        )
    # Even a scene record pointing at a real core job in another project cannot
    # turn its valid same-project alias into cross-project diagnostic access.
    foreign = parent.model_copy(update={"id": "foreign-scene-job", "project_id": other})
    client.portal.call(client.app.state.execution._accept, foreign, None)
    run.job_id = foreign.id
    client.app.state.scenes.save(run)
    for suffix in ("logs", "failure", "diagnostics.zip"):
        assert client.get(endpoint + "/" + suffix).status_code == 404


def test_active_scene_alias_cancellation_stops_the_owning_core_job(client):
    ident, dataset, parent, child, run = saved_scene_retry_fixture(client)
    state = client.app.state
    parent.status, parent.result = "running", None
    client.portal.call(state.execution.save, parent)
    run.status = "running"
    state.scenes.save(run)
    stopped = []

    async def attach_worker():
        async def worker():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                current = state.scenes.get(run.id)
                current.status = "cancelled"
                state.scenes.event(current, "cancelled", "Owned fixture worker stopped")
                stopped.append(True)
                raise

        task = asyncio.create_task(worker())
        state.execution.tasks[parent.id] = task
        await asyncio.sleep(0)

    client.portal.call(attach_worker)
    endpoint = f"/api/v1/projects/{ident}/work-jobs/scene:{run.id}/cancel"
    response = client.post(endpoint)
    assert response.status_code == 200, response.text
    assert response.json()["id"] == "scene:" + run.id
    assert response.json()["status"] == "cancelled"
    owner = client.portal.call(state.execution.get, parent.id)
    assert owner.status == "cancelled" and owner.cancellation.cleanup == "succeeded"
    assert stopped == [True] and state.execution.tasks[parent.id].done()


@pytest.mark.parametrize("mismatch", ["missing", "project", "scene", "operation"])
def test_scene_alias_cancellation_rejects_forged_or_missing_core_ownership(client, mismatch):
    ident, dataset, parent, child, run = saved_scene_retry_fixture(client)
    state = client.app.state
    run.status = "running"
    if mismatch == "missing":
        run.job_id = "missing-scene-job"
        owner = None
    elif mismatch == "operation":
        run.job_id = dataset["source_id"]
        owner = client.portal.call(state.execution.get, run.job_id)
    else:
        owner = parent.model_copy(update={"id": "forged-scene-job", "status": "running"})
        if mismatch == "project":
            owner.project_id = project(client, "Other scene owner")
        else:
            owner.request = owner.request.model_copy(update={"request_id": "b" * 32})
        client.portal.call(state.execution._accept, owner, None)
        run.job_id = owner.id
    state.scenes.save(run)
    response = client.post(f"/api/v1/projects/{ident}/work-jobs/scene:{run.id}/cancel")
    assert response.status_code == 404
    assert state.scenes.get(run.id).status == "running"
    if owner is not None:
        current = client.portal.call(state.execution.get, owner.id)
        assert current.status == owner.status
        assert current.cancellation is None
