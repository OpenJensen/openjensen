import asyncio
import hashlib
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import insert
from vla_platform.api import create_app
from vla_platform.contracts import IntakeRequest, Job, WorkerResult, now
from vla_platform.datasets import cache, inspect
from vla_platform.datasets.inspect import profile
from vla_platform.execution import Execution
from vla_platform.settings import Settings
from vla_platform.storage import Storage, jobs

SHA = "a" * 40
NEW_SHA = "b" * 40
RAW = json.dumps(
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


@pytest.fixture(autouse=True)
def park_workers(monkeypatch):
    async def wait(self, job_id):
        await asyncio.Event().wait()

    # Cache tests control completion explicitly. Real worker persistence/reuse is
    # covered by test_intake, including restart of the application.
    monkeypatch.setattr(Execution, "run", wait)


async def saved_job(execution, *, project="p", revision=SHA, status="succeeded", repo="test/data"):
    request = IntakeRequest(repo_id=repo, revision="main")
    job = Job(
        id=f"{project}-{revision}-{status}",
        project_id=project,
        status=status,
        request=request,
        created_at=now(),
        updated_at=now(),
        result=profile(RAW, request, revision) if status == "succeeded" else None,
    )
    async with execution.storage.engine.begin() as connection:
        await connection.execute(
            insert(jobs).values(
                id=job.id, project_id=project, status=status, record=job.model_dump()
            )
        )
    return job


def test_hub_reuses_existing_inspection_and_follows_latest(tmp_path, monkeypatch):
    resolved = SHA
    refs = []

    async def resolve(request):
        refs.append(request.revision)
        return resolved

    monkeypatch.setattr(cache, "resolve_hub_revision", resolve)

    async def run():
        nonlocal resolved
        storage = Storage(tmp_path)
        await storage.initialize()
        execution = Execution(storage, Settings(data_dir=tmp_path))
        execution.slots = asyncio.Semaphore(0)
        try:
            previous = await saved_job(execution)
            # Older saved requests may hold a mutable ref; match their result SHA.
            reused = await execution.inspections.submit(
                "p", IntakeRequest(repo_id="test/data", revision="")
            )
            assert reused.id == previous.id
            assert not execution.tasks
            resolved = NEW_SHA
            new = await execution.inspections.submit("p", IntakeRequest(repo_id="test/data"))
            assert new.id != previous.id
            assert new.request.revision == NEW_SHA
            assert new.status == "queued"
            assert refs == ["main", "main"]
            assert len(await execution.list("p")) == 2
        finally:
            await execution.close()
            await storage.close()

    asyncio.run(run())


def test_concurrent_inspections_share_one_inflight_job(tmp_path, monkeypatch):
    async def resolve(request):
        await asyncio.sleep(0)
        return SHA

    monkeypatch.setattr(cache, "resolve_hub_revision", resolve)

    async def run():
        storage = Storage(tmp_path)
        await storage.initialize()
        execution = Execution(storage, Settings(data_dir=tmp_path))
        execution.slots = asyncio.Semaphore(0)
        try:
            results = await asyncio.gather(
                *[
                    execution.inspections.submit("p", IntakeRequest(repo_id="test/data"))
                    for _ in range(8)
                ]
            )
            assert len({job.id for job in results}) == 1
            assert len(await execution.list("p")) == 1
            job = await execution.get(results[0].id)
            job.status = "running"
            await execution.save(job)
            assert (await execution.inspections.submit("p", job.request)).id == job.id
            await execution.finish(job.id, WorkerResult(result=profile(RAW, job.request, SHA)))
            assert (await execution.inspections.submit("p", job.request)).status == "succeeded"
        finally:
            await execution.close()
            await storage.close()

    asyncio.run(run())


@pytest.mark.parametrize("status", ["failed", "cancelled", "interrupted"])
def test_unfinished_results_are_retryable(tmp_path, status):
    async def run():
        storage = Storage(tmp_path)
        await storage.initialize()
        execution = Execution(storage, Settings(data_dir=tmp_path))
        execution.slots = asyncio.Semaphore(0)
        try:
            old = await saved_job(execution, status=status)
            new = await execution.inspections.submit(
                "p", IntakeRequest(repo_id="test/data", revision=SHA)
            )
            assert new.id != old.id
            assert new.status == "queued"
        finally:
            await execution.close()
            await storage.close()

    asyncio.run(run())


def test_cache_is_scoped_to_project_and_repository(tmp_path):
    async def run():
        storage = Storage(tmp_path)
        await storage.initialize()
        execution = Execution(storage, Settings(data_dir=tmp_path))
        execution.slots = asyncio.Semaphore(0)
        try:
            old = await saved_job(execution)
            other_project = await execution.inspections.submit(
                "other", IntakeRequest(repo_id="test/data", revision=SHA)
            )
            other_repo = await execution.inspections.submit(
                "p", IntakeRequest(repo_id="test/other", revision=SHA)
            )
            assert len({old.id, other_project.id, other_repo.id}) == 3
        finally:
            await execution.close()
            await storage.close()

    asyncio.run(run())


def test_local_metadata_changes_invalidate_cache(tmp_path):
    dataset = tmp_path / "datasets" / "robot"
    (dataset / "meta").mkdir(parents=True)
    metadata = dataset / "meta/info.json"
    metadata.write_bytes(RAW)

    async def run():
        storage = Storage(tmp_path)
        await storage.initialize()
        execution = Execution(storage, Settings(data_dir=tmp_path, local_root=dataset.parent))
        execution.slots = asyncio.Semaphore(0)
        try:
            request = IntakeRequest(source="local", path="robot")
            first = await execution.inspections.submit("p", request)
            assert first.request.path == str(dataset)
            absolute = await execution.inspections.submit(
                "p", request.model_copy(update={"path": str(dataset)})
            )
            assert absolute.id == first.id
            metadata.write_bytes(RAW.replace(b'"total_frames": 20', b'"total_frames": 21'))
            changed = await execution.inspections.submit("p", request)
            assert changed.id != first.id
            assert changed.request.revision != first.request.revision
        finally:
            await execution.close()
            await storage.close()

    asyncio.run(run())


def test_local_identity_never_reuses_a_path_outside_current_root(tmp_path):
    dataset = tmp_path / "outside"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_bytes(RAW)
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "escape").symlink_to(dataset, target_is_directory=True)
    assert cache.local_identity(IntakeRequest(source="local", path="escape"), allowed) is None


def test_hub_resolution_reads_current_ref_and_skips_pinned_network(monkeypatch):
    paths = []
    client_type = httpx.AsyncClient

    def reply(request):
        paths.append(request.url.path)
        return httpx.Response(200, json={"sha": NEW_SHA})

    monkeypatch.setattr(
        inspect.httpx,
        "AsyncClient",
        lambda **kwargs: client_type(transport=httpx.MockTransport(reply), **kwargs),
    )

    async def run():
        assert (
            await inspect.resolve_hub_revision(IntakeRequest(repo_id="test/data", revision=SHA))
            == SHA
        )
        assert not paths
        assert (
            await inspect.resolve_hub_revision(IntakeRequest(repo_id="test/data", revision=""))
            == NEW_SHA
        )
        assert (
            await inspect.resolve_hub_revision(
                IntakeRequest(repo_id="test/data", revision="refs/pr/1")
            )
            == NEW_SHA
        )
        assert paths == [
            "/api/datasets/test/data/revision/main",
            "/api/datasets/test/data/revision/refs/pr/1",
        ]

    asyncio.run(run())


@pytest.mark.parametrize(
    "error,status",
    [
        (ValueError("Dataset, revision or LeRobot metadata was not found"), 422),
        (httpx.ReadTimeout("timeout"), 504),
        (httpx.ConnectError("offline"), 502),
    ],
)
def test_resolution_error_does_not_serve_stale_latest(tmp_path, monkeypatch, error, status):
    async def fail(request):
        raise error

    monkeypatch.setattr(cache, "resolve_hub_revision", fail)
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        project = client.post("/api/v1/projects", json={"name": "Cached"}).json()
        response = client.post(
            f"/api/v1/projects/{project['id']}/intakes",
            json={"repo_id": "test/data", "revision": ""},
        )
        assert response.status_code == status
        assert "detail" in response.json()
        assert client.get(f"/api/v1/projects/{project['id']}/jobs").json() == []


@pytest.mark.parametrize("extra", [b'"extra":NaN', b'"total_frames":20', b'"extra":1e999'])
def test_strict_local_admission_prevents_reusing_legacy_success(tmp_path, extra):
    dataset = tmp_path / "datasets" / "robot"
    (dataset / "meta").mkdir(parents=True)
    # Older decoding ignored this invalid/ambiguous field but recorded these exact bytes.
    raw = RAW[:-1] + b"," + extra + b"}"
    (dataset / "meta/info.json").write_bytes(raw)
    revision = "metadata-sha256:" + hashlib.sha256(raw).hexdigest()

    async def run():
        storage = Storage(tmp_path)
        await storage.initialize()
        execution = Execution(storage, Settings(data_dir=tmp_path, local_root=dataset.parent))
        request = IntakeRequest(source="local", path=str(dataset), revision=revision)
        original_revision = "metadata-sha256:" + hashlib.sha256(RAW).hexdigest()
        legacy_result = profile(RAW, request, original_revision).model_copy(
            update={"revision": revision, "metadata_sha256": hashlib.sha256(raw).hexdigest()}
        )
        old = Job(
            id="legacy-local-success",
            project_id="p",
            status="succeeded",
            request=request,
            created_at=now(),
            updated_at=now(),
            result=legacy_result,
        )
        try:
            async with storage.engine.begin() as connection:
                await connection.execute(
                    insert(jobs).values(
                        id=old.id, project_id="p", status=old.status, record=old.model_dump()
                    )
                )
            assert cache.local_identity(request, dataset.parent) is None
            current = await execution.inspections.submit("p", request)
            assert current.id != old.id and current.status == "queued"
            assert current.result is None
            assert await execution.get(old.id) == old
        finally:
            await execution.close()
            await storage.close()

    asyncio.run(run())


def test_valid_in_root_symlink_keeps_local_cache_identity(tmp_path):
    dataset = tmp_path / "robot"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_bytes(RAW)
    (tmp_path / "alias").symlink_to(dataset, target_is_directory=True)
    request = IntakeRequest(source="local", path="alias")
    resolved = cache.local_identity(request, tmp_path)
    assert resolved is not None
    assert resolved.path == str(dataset)
    assert resolved.revision == "metadata-sha256:" + hashlib.sha256(RAW).hexdigest()


def test_invalid_local_profile_cannot_reuse_a_metadata_hash(tmp_path):
    dataset = tmp_path / "robot"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text('{"codebase_version":"v3.0"}')
    assert cache.local_identity(IntakeRequest(source="local", path="robot"), tmp_path) is None
