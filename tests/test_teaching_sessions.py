"""Managed local teaching contracts and ownership; generated fixtures only."""

import asyncio
import json
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from vla_platform.api import create_app
from vla_platform.contracts import IntakeRequest
from vla_platform.execution import Execution
from vla_platform.settings import Settings
from vla_platform.storage import Storage
from vla_platform.submissions import SubmissionConflict
from vla_platform.teaching_sessions import config, publication
from vla_platform.teaching_sessions.contracts import TeachingCaptureRequest
from vla_platform.teaching_sessions.service import TeachingSessions, acquire_lease

REPO = Path(__file__).resolve().parents[1]


def recipe(**changes):
    return TeachingCaptureRequest(profile_id="local", profile_sha256="a" * 64, **changes)


@pytest.mark.parametrize(
    "field,value",
    [
        ("timeout_seconds", True),
        ("timeout_seconds", "300"),
        ("timeout_seconds", 0),
        ("timeout_seconds", 3601),
        ("profile_id", "../escape"),
        ("profile_sha256", "bad"),
        ("executable", "/bin/sh"),
        ("url", "https://elsewhere.invalid"),
        ("output", "/tmp/other"),
    ],
)
def test_request_cannot_relax_boundaries(field, value):
    data = recipe().model_dump()
    data[field] = value
    with pytest.raises(ValueError):
        TeachingCaptureRequest.model_validate(data)


def test_legacy_intake_serialization_unchanged():
    assert IntakeRequest(source="local", path="dataset").model_dump() == {
        "source": "local",
        "repo_id": None,
        "revision": "main",
        "path": "dataset",
        "snapshot_for_training": False,
    }


def test_durable_teaching_key_concurrency_and_restart(tmp_path):
    async def exercise():
        storage = Storage(tmp_path)
        await storage.initialize()
        execution = Execution(storage, Settings(data_dir=tmp_path))
        dispatched = []

        async def validate(_project, _request):
            return None

        async def record(job_id):
            dispatched.append(job_id)

        execution.teaching.validate = validate
        execution.run = record
        request = recipe()
        first = await asyncio.gather(
            *[execution.submit("p", request, idempotency_key="durable-teaching") for _ in range(8)]
        )
        await asyncio.sleep(0)
        assert len({item.id for item in first}) == 1
        assert dispatched == [first[0].id]
        assert first[0].kind == "teaching.capture"
        with pytest.raises(SubmissionConflict):
            await execution.submit(
                "p", recipe(timeout_seconds=299), idempotency_key="durable-teaching"
            )
        await execution.close()
        await storage.close()
        storage = Storage(tmp_path)
        await storage.initialize()
        restarted = Execution(storage, Settings(data_dir=tmp_path))

        async def forbidden(*_):
            pytest.fail("An accepted key must not relaunch or resolve current profiles")

        restarted.teaching.validate = forbidden
        restarted.run = forbidden
        try:
            await restarted.reconcile()
            recovered = await restarted.submit("p", request, idempotency_key="durable-teaching")
            assert recovered.id == first[0].id and recovered.status == "interrupted"
            assert not restarted.teaching.live
        finally:
            await restarted.close()
            await storage.close()

    asyncio.run(exercise())


def test_api_missing_profile_and_project_fail_closed(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        project = client.post("/api/v1/projects", json={"name": "Generated test"}).json()["id"]
        response = client.get(f"/api/v1/projects/{project}/teaching/profiles")
        assert response.status_code == 200
        assert response.json() == {
            "configured": False,
            "available": False,
            "profiles": [],
            "message": "Managed local Isaac teaching is not configured.",
        }
        started = client.post(
            f"/api/v1/projects/{project}/teaching/sessions", json=recipe().model_dump()
        )
        assert started.status_code == 503
        assert client.get(f"/api/v1/projects/{project}/jobs").json() == []
        assert client.get("/api/v1/projects/missing/teaching/profiles").status_code == 404


def test_api_uses_same_durable_key_and_no_cross_project_controls(tmp_path, monkeypatch):
    calls = []

    async def validate(*_):
        return None

    async def record(_self, job_id):
        calls.append(job_id)

    monkeypatch.setattr(TeachingSessions, "validate", validate)
    monkeypatch.setattr(Execution, "run", record)
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        project = client.post("/api/v1/projects", json={"name": "A"}).json()["id"]
        other = client.post("/api/v1/projects", json={"name": "B"}).json()["id"]
        url = f"/api/v1/projects/{project}/teaching/sessions"
        first = client.post(
            url, json=recipe().model_dump(), headers={"Idempotency-Key": "saved-key"}
        )
        assert first.status_code == 202 and first.headers["Idempotency-Key"] == "saved-key"
        again = client.post(
            url, json=recipe().model_dump(), headers={"Idempotency-Key": "saved-key"}
        )
        job_id = first.json()["id"]
        assert again.json()["id"] == job_id
        assert calls == [job_id]
        lookup = client.get(
            f"/api/v1/projects/{project}/submissions/saved-key?operation=teaching.capture"
        )
        assert lookup.json()["id"] == job_id
        assert (
            client.post(f"/api/v1/projects/{other}/teaching/sessions/{job_id}/stop").status_code
            == 404
        )
        for _ in range(2):
            stop = client.post(url + f"/{job_id}/stop")
            assert stop.status_code == 200 and stop.json()["stop_requested"] is True
        assert client.get(url + f"/{job_id}/state").status_code == 409


def operator_files(tmp_path):
    capture = tmp_path / "published"
    capture.mkdir()
    scene = tmp_path / "scene.usda"
    scene.write_text("#usda 1.0\n")
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "scene": str(scene),
                "camera": "/World/Camera",
                "articulation": "/World/Robot",
                "joints": ["joint_0"],
                "width": 2,
                "height": 2,
                "fps": 30,
                "physics_hz": 60,
                "lineage_group": "generated-control-fixture",
                "max_steps": 10,
                "max_episode_bytes": 4096,
            }
        )
    )
    recording = tmp_path / "recordings.json"
    recording.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_python": sys.executable,
                "worker_root": str(REPO / "workers/teaching"),
                "ffmpeg": sys.executable,
                "ffprobe": sys.executable,
                "projects": [{"project_id": "p", "capture_root": str(capture)}],
            }
        )
    )
    profile = {
        "id": "local",
        "label": "Generated test profile",
        "project_id": "p",
        "isaac_python": sys.executable,
        "worker_root": str(REPO / "workers/teaching"),
        "settings_path": str(settings),
        "lease_path": str(tmp_path / "gpu.lock"),
        "control_port": 45678,
        "accept_eula": True,
    }
    config_path = tmp_path / "managed.json"
    config_path.write_text(json.dumps({"schema_version": 1, "profiles": [profile]}))
    return config_path, recording, settings, scene


def test_profile_identity_binds_settings_scene_and_recording_mapping(tmp_path):
    path, recording, settings, scene = operator_files(tmp_path)
    first = config.load(path, recording, "local", "p")
    scene.write_text("#usda 1.0\n# changed root bytes\n")
    assert config.load(path, recording, "local", "p").identity != first.identity
    with pytest.raises(config.TeachingError, match="belongs"):
        config.load(path, recording, "local", "other")
    value = config.options(path, recording, "p")
    assert value["profiles"][0]["runtime_verified"] is False
    if not sys.platform.startswith("linux"):
        assert value["available"] is False


def test_kernel_lease_excludes_other_owner_and_does_not_replace_file(tmp_path):
    path = tmp_path / "lease"
    fd = acquire_lease(path)
    try:
        with pytest.raises(BlockingIOError):
            acquire_lease(path)
    finally:
        os.close(fd)
    later = acquire_lease(path)
    os.close(later)
    link = tmp_path / "linked"
    link.symlink_to(path)
    with pytest.raises(OSError):
        acquire_lease(link)


def test_capture_copy_is_bounded_and_publication_never_replaces(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    (source / "session.json").write_bytes(b"example raw fixture")
    before = publication.inventory(source, 1024)
    stage = tmp_path / "stage"
    publication.stage_capture(source, stage, {"inventory": before}, 1024)
    target = tmp_path / "catalog"
    target.mkdir()
    destination = publication.publish(stage, target, "a" * 32)
    assert publication.inventory(destination, 1024) == before
    assert publication.inventory(source, 1024) == before
    stage.mkdir()
    (stage / "other").write_text("retain")
    with pytest.raises(FileExistsError):
        publication.publish(stage, target, "a" * 32)
    assert (stage / "other").read_text() == "retain"
    with pytest.raises(config.TeachingError, match="byte"):
        publication.inventory(source, 1)


def test_publication_fsync_failure_rolls_back_only_owned_directory(tmp_path, monkeypatch):
    source, target = tmp_path / "stage", tmp_path / "catalog"
    source.mkdir()
    target.mkdir()
    original = publication._publish
    calls = []

    def failure(stage, destination):
        calls.append((stage, destination))
        original(stage, destination)
        if len(calls) == 1:
            raise OSError("generated fsync failure after rename")

    monkeypatch.setattr(publication, "_publish", failure)
    with pytest.raises(OSError):
        publication.publish(source, target, "a" * 32)
    assert source.is_dir() and not (target / ("a" * 32)).exists()


def test_inventory_refuses_symlinks_without_reading_target(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    secret = tmp_path / "unrelated"
    secret.write_text("must not read")
    (root / "linked").symlink_to(secret)
    with pytest.raises(config.TeachingError, match="linked"):
        publication.inventory(root, 1024)


def test_ready_requires_matching_authenticated_state_and_source_receipt(tmp_path):
    from vla_platform.datasets import recordings
    from vla_platform.teaching_sessions.service import Live

    async def exercise():
        config_path, recording_path, _settings, _scene = operator_files(tmp_path)
        profile = config.load(config_path, recording_path, "local", "p")
        directory = tmp_path / "job"
        capture = directory / "capture"
        capture.mkdir(parents=True)
        meta = {
            "schema_version": 1,
            "session_id": "a" * 32,
            "controller": "joint_position_targets",
            "state_units": "radians",
            "action_units": "radians",
            "timebase": "simulation_seconds",
            "camera_key": "observation.images.front",
            "camera_prim": "/World/Camera",
            "joint_names": ["joint_0"],
            "origin": "recorded",
            "lineage_group": "group",
            "scene_sha256": profile.scene_sha256,
            "width": 2,
            "height": 2,
            "fps": 30,
            "physics_hz": 60,
        }
        publication.new_json(capture / "session.json", meta)
        publication.new_json(
            directory / "preflight.json",
            {
                "schema_version": 1,
                "metadata": {
                    key: value
                    for key, value in meta.items()
                    if key not in {"schema_version", "session_id"}
                },
                "settings_sha256": profile.settings_sha256,
            },
        )
        service = TeachingSessions(SimpleNamespace())
        live = Live(profile, SimpleNamespace(returncode=None), "private", directory)
        response = {
            "mode": "idle",
            "episode_id": None,
            "revision": 0,
            "session_id": "b" * 32,
            "joints": ["joint_0"],
        }

        async def actual_endpoint(*_):
            return response

        service.upstream = actual_endpoint
        with pytest.raises(config.TeachingError, match="identity"):
            await service.ready(live)
        assert live.session_id is None
        response["session_id"] = "a" * 32
        assert await service.ready(live)
        assert live.session_id == "a" * 32
        value = recordings.decode((directory / "preflight.json").read_bytes())
        value["settings_sha256"] = "0" * 64
        (directory / "preflight.json").write_bytes(recordings.canonical(value))
        with pytest.raises(config.TeachingError, match="settings"):
            await service.ready(live)

    asyncio.run(exercise())


def test_cleanup_drain_survives_repeated_parent_cancellation():
    from vla_platform.teaching_sessions.service import drain

    async def exercise():
        entered, release = asyncio.Event(), asyncio.Event()
        completed = []

        async def owned_cleanup():
            entered.set()
            await release.wait()
            completed.append(True)

        child = asyncio.create_task(owned_cleanup())
        owner = asyncio.create_task(drain(child))
        await entered.wait()
        owner.cancel()
        await asyncio.sleep(0)
        owner.cancel()
        await asyncio.sleep(0)
        assert not child.cancelled() and not owner.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert completed == [True]

    asyncio.run(exercise())


def test_relay_refuses_another_session_before_network():
    from vla_platform.teaching_api import TeachingCommand
    from vla_platform.teaching_sessions.service import Live

    async def exercise():
        service = TeachingSessions(SimpleNamespace())

        async def owned(*_):
            return SimpleNamespace(id="job", status="running")

        async def forbidden(*_):
            pytest.fail("A foreign session must never reach the worker")

        service.owned_job = owned
        service.upstream = forbidden
        service.live["job"] = Live(
            None, SimpleNamespace(returncode=None), "private", Path("/unused"), session_id="a" * 32
        )
        command = TeachingCommand(
            session_id="b" * 32, command_id="command", expected_revision=0, operation="pause"
        )
        with pytest.raises(config.TeachingError, match="different"):
            await service.relay("project", "job", "/commands", command)

    asyncio.run(exercise())


def test_publication_refuses_changed_source_and_keeps_private_stage(tmp_path):
    root, stage = tmp_path / "raw", tmp_path / "stage"
    root.mkdir()
    (root / "session.json").write_bytes(b"original")
    expected = publication.inventory(root, 1024)
    (root / "session.json").write_bytes(b"mutated")
    with pytest.raises(config.TeachingError, match="changed"):
        publication.stage_capture(root, stage, {"inventory": expected}, 1024)
    assert (root / "session.json").read_bytes() == b"mutated"
    assert stage.is_dir()


def test_post_rename_failure_never_removes_concurrent_replacement(tmp_path, monkeypatch):
    stage, catalog = tmp_path / "stage", tmp_path / "catalog"
    stage.mkdir()
    catalog.mkdir()
    destination = catalog / ("a" * 32)
    original = publication._publish
    displaced = tmp_path / "owned-retained"

    def changed(source, target):
        original(source, target)
        target.rename(displaced)
        target.mkdir()
        (target / "other-owner").write_text("preserve")
        raise OSError("generated concurrent replacement")

    monkeypatch.setattr(publication, "_publish", changed)
    with pytest.raises(publication.PublicationUncertain, match="may remain"):
        publication.publish(stage, catalog, "a" * 32)
    assert (destination / "other-owner").read_text() == "preserve"
    assert displaced.is_dir()


@pytest.mark.parametrize(
    "field,value", [("schema_version", True), ("accept_eula", 1), ("accept_eula", "true")]
)
def test_config_literal_types_do_not_accept_boolean_numeric_aliases(tmp_path, field, value):
    path, recording, _settings, _scene = operator_files(tmp_path)
    data = json.loads(path.read_text())
    if field == "schema_version":
        data[field] = value
    else:
        data["profiles"][0][field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        config.load(path, recording, "local", "p")


def test_real_journal_bytes_publish_into_existing_recording_catalog(tmp_path):
    """Use the real journal/inspector with explicitly synthetic simulator observations."""
    import importlib.util

    from vla_platform.datasets import recordings

    source = REPO / "workers/teaching/tests/conftest.py"
    spec = importlib.util.spec_from_file_location("managed_teaching_fixture", source)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    from firebird_teaching.dataset import inspect_capture
    from firebird_teaching.managed import verify_capture

    config_path, recording_path, _settings, _scene = operator_files(tmp_path)
    profile = config.load(config_path, recording_path, "local", "p")
    raw = tmp_path / "generated-raw"
    episodes = fixture.record_fixture(raw, episodes=2, frames=3)
    before = publication.inventory(raw, 1024**2)
    verified = tmp_path / "verified.json"
    verify_capture(raw, verified, 1024**2)
    value = publication.proof(raw, verified, 1024**2)
    stage = tmp_path / "private-stage"
    publication.stage_capture(raw, stage, value, 1024**2)
    published = publication.publish(stage, profile.recording.root("p"), value["session_id"])
    catalog = recordings.catalog(profile.recording, "p")
    assert len(catalog["captures"]) == 1
    capture = catalog["captures"][0]
    assert capture["session_id"] == value["session_id"]
    assert capture["session_sha256"] == value["session_sha256"]
    assert capture["origin"] == "synthetic"
    assert {item["episode_id"] for item in capture["episodes"]} == set(episodes)
    assert capture["content_verified"] is False  # Read-only catalog does not re-read frames.
    metadata, admitted, inventory = inspect_capture(published, episodes)
    assert metadata["state_units"] == metadata["action_units"] == "radians"
    assert metadata["controller"] == "joint_position_targets"
    assert len({item["lineage_group"] for item in admitted}) == 1
    assert any(
        row["requested_target_rad"] != row["applied_target_rad"]
        for item in admitted
        for row in item["rows"]
    )
    assert inventory == before == publication.inventory(raw, 1024**2)


@pytest.mark.parametrize(
    "status,expected",
    [
        ("running", True),
        ("queued", False),
        ("cancelled", False),
        ("failed", False),
        ("interrupted", False),
        ("succeeded", False),
    ],
)
def test_ready_status_requires_running_saved_job(status, expected):
    from vla_platform.teaching_sessions.service import Live

    async def exercise():
        job = SimpleNamespace(id="job", status=status, result=None)
        service = TeachingSessions(SimpleNamespace())

        async def owned(*_):
            return job

        service.owned_job = owned
        service.live[job.id] = Live(
            None,
            SimpleNamespace(returncode=None),
            "private",
            Path("/unused"),
            session_id="a" * 32,
        )
        response = await service.status("p", job.id)
        assert response["ready"] is expected
        assert response["session_id"] == "a" * 32
        assert response["stop_requested"] is False

    asyncio.run(exercise())


def controlled_run(tmp_path, monkeypatch):
    """Run the real adapter control flow with inert process handles, never an SDK."""
    from vla_platform.teaching_sessions import service as module

    catalog = tmp_path / "catalog"
    catalog.mkdir()
    job = SimpleNamespace(
        id="owned-job",
        project_id="p",
        request=recipe(),
        status="queued",
        result=None,
        error=None,
    )
    saved = []

    async def get(_job_id):
        return job

    async def save(current):
        saved.append((current.status, current.error))

    execution = SimpleNamespace(
        settings=SimpleNamespace(
            data_dir=tmp_path,
            teaching_session_config=None,
            recording_config=None,
        ),
        lock=asyncio.Lock(),
        get=get,
        save=save,
    )
    adapter = TeachingSessions(execution)
    profile = SimpleNamespace(
        identity="a" * 64,
        recording=SimpleNamespace(identity="b" * 64, root=lambda _: catalog),
        value=SimpleNamespace(
            lease_path=str(tmp_path / "lease"),
            isaac_python=sys.executable,
            settings_path=str(tmp_path / "unused-settings.json"),
            worker_root=str(tmp_path),
            control_port=45678,
            max_capture_bytes=1024**2,
        ),
        environment=lambda _: {},
    )
    owner = SimpleNamespace(pid=123456789, returncode=None, stdin=None)
    verifier = SimpleNamespace(pid=123456790, returncode=0, stdin=None)

    async def wait_owner():
        return owner.returncode

    async def wait_verifier():
        return verifier.returncode

    owner.wait, verifier.wait = wait_owner, wait_verifier
    directory = tmp_path / "jobs" / job.id / "teaching"
    handles, cleaned = [], []

    async def spawn(*_args, **_kwargs):
        handle = owner if not handles else verifier
        handles.append(handle)
        if handle is owner:
            raw = directory / "capture"
            raw.mkdir()
            (raw / "session.json").write_bytes(b"retained generated raw evidence")
            publication.new_json(
                directory / "terminal.json",
                {
                    "schema_version": 1,
                    "reason": "requested_stop",
                    "worker_exit_code": 0,
                    "supervisor_pid": owner.pid,
                    "group_id": owner.pid,
                    "group_kill_required": True,
                },
            )
        return handle

    async def cleanup(handle):
        cleaned.append(handle)

    async def validate(*_):
        return profile

    async def ready(live):
        live.session_id = "c" * 32
        owner.returncode = -signal.SIGKILL
        return True

    value = {
        "session_id": "c" * 32,
        "session_sha256": "d" * 64,
        "inventory_sha256": "e" * 64,
        "origin": "synthetic",
        "lineage_group": "fixture",
        "episodes": [
            {
                "episode_id": "f" * 32,
                "receipt_sha256": "0" * 64,
                "frames": 1,
                "termination": "finish",
                "outcome": "unknown",
            }
        ],
    }

    def stage(raw, target, _value, _maximum):
        target.mkdir()
        (target / "session.json").write_bytes((raw / "session.json").read_bytes())

    adapter.validate, adapter.ready = validate, ready
    monkeypatch.setattr(module, "spawn", spawn)
    monkeypatch.setattr(module, "cleanup", cleanup)
    monkeypatch.setattr(module, "load", lambda *_: profile)
    monkeypatch.setattr(publication, "proof", lambda *_: value)
    monkeypatch.setattr(publication, "stage_capture", stage)
    return SimpleNamespace(
        adapter=adapter,
        job=job,
        owner=owner,
        verifier=verifier,
        cleaned=cleaned,
        handles=handles,
        directory=directory,
        catalog=catalog,
        saved=saved,
        module=module,
    )


def test_successful_cleanup_is_not_repeated_after_parent_cancellation(tmp_path, monkeypatch):
    async def exercise():
        fixture = controlled_run(tmp_path, monkeypatch)
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def cleanup(handle):
            calls.append(handle)
            entered.set()
            await release.wait()

        monkeypatch.setattr(fixture.module, "cleanup", cleanup)
        task = asyncio.create_task(fixture.adapter.run(fixture.job))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
            assert calls == [fixture.owner]
            assert fixture.handles == [fixture.owner]  # No content verifier or publication.
            assert list(fixture.catalog.iterdir()) == []
            assert not fixture.adapter.live and not fixture.adapter.stops
            fd = acquire_lease(tmp_path / "lease")
            os.close(fd)
        finally:
            release.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())


def test_failed_rollback_retains_capture_and_honest_job_guidance(tmp_path, monkeypatch):
    async def exercise():
        fixture = controlled_run(tmp_path, monkeypatch)
        original = publication._publish
        calls = []

        def failed_rollback(source, destination):
            calls.append((source, destination))
            if len(calls) == 1:
                original(source, destination)
                raise OSError("generated fsync failure after successful publication")
            raise OSError("generated rollback rename failure")

        monkeypatch.setattr(publication, "_publish", failed_rollback)
        await fixture.adapter.run(fixture.job)
        assert len(calls) == 2
        assert fixture.job.status == "failed" and fixture.job.result is None
        assert "may remain in the catalog" in fixture.job.error
        assert "No capture was published" not in fixture.job.error
        assert fixture.saved[-1] == ("failed", fixture.job.error)
        assert fixture.cleaned == [fixture.owner, fixture.verifier]
        assert len(fixture.handles) == 2  # No automatic retry or replacement.
        destination = fixture.catalog / ("c" * 32) / "session.json"
        raw = fixture.directory / "capture" / "session.json"
        assert destination.read_bytes() == raw.read_bytes() == b"retained generated raw evidence"
        assert not (tmp_path / (".teaching-" + fixture.job.id)).exists()
        assert not fixture.adapter.live and not fixture.adapter.stops

    asyncio.run(exercise())
