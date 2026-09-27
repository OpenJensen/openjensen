"""Persistent augmentation orchestration with isolated, unpaid media/provider doubles."""

import asyncio
import hashlib
import io
import json
import threading
import time
import zipfile
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import insert
from vla_platform.api import create_app
from vla_platform.augmentation import service
from vla_platform.augmentation.contracts import (
    MODEL,
    AugmentationRequest,
    AugmentationResult,
)
from vla_platform.contracts import IntakeRequest, Job, now
from vla_platform.datasets.inspect import profile
from vla_platform.lifecycle.contracts import LifecycleResult, PolicyRequest
from vla_platform.settings import Settings
from vla_platform.storage import jobs

SHA = "a" * 40
CAMERA = "observation.images.wrist"


def inspected_job(project_id, **overrides):
    intake = IntakeRequest(repo_id="fixture/robot")
    metadata = json.dumps(
        {
            "codebase_version": "v3.0",
            "total_episodes": 3,
            "total_frames": 900,
            "fps": 30,
            "features": {
                CAMERA: {"dtype": "video", "shape": [3, 240, 320]},
                "observation.state": {"dtype": "float32", "shape": [2]},
                "action": {"dtype": "float32", "shape": [2]},
            },
        }
    ).encode()
    return Job(
        **(
            {
                "id": "inspected",
                "project_id": project_id,
                "status": "succeeded",
                "request": intake,
                "result": profile(metadata, intake, SHA),
                "created_at": now(),
                "updated_at": now(),
            }
            | overrides
        )
    )


def seed(client, app, record):
    async def insert_record():
        async with app.state.execution.storage.engine.begin() as connection:
            await connection.execute(
                insert(jobs).values(
                    id=record.id,
                    project_id=record.project_id,
                    status=record.status,
                    record=record.model_dump(),
                )
            )

    client.portal.call(insert_record)


def project(client):
    return client.post("/api/v1/projects", json={"name": "Augmentation fixture"}).json()["id"]


def request_body(**overrides):
    return {
        "source_job_id": "inspected",
        "episode_indices": [0, 2],
        "camera_key": CAMERA,
        "preset": "texture",
        "prompt": "Use blue fabric.",
        "start_seconds": 1,
        "duration_seconds": 5,
    } | overrides


def submit(client, project_id, **overrides):
    return client.post(
        f"/api/v1/projects/{project_id}/augmentations", json=request_body(**overrides)
    )


def wait(client, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        record = response.json()
        if record["status"] not in {"queued", "running"}:
            return record
        time.sleep(0.01)
    pytest.fail("Augmentation fixture did not finish")


@pytest.fixture
def doubles(monkeypatch):
    state = SimpleNamespace(
        events=[],
        missing_camera=None,
        short_episode=None,
        trim_failure=None,
        verification_failure=False,
        blocked=False,
        late_success=False,
        started=threading.Event(),
        provider_closed=False,
        explorer_closed=False,
        source_bytes=b"source video bytes remain unchanged",
    )
    monkeypatch.setenv("GEMINI_API_KEY", "unpaid-fixture-key")
    monkeypatch.setattr(service.shutil, "which", lambda binary: f"/fixture/{binary}")

    class Explorer:
        def __init__(self):
            self.reader = self

        async def preview(self, source, episode):
            assert source.result.revision == SHA
            state.events.append(("preview", episode))
            start = episode * 10
            return SimpleNamespace(
                duration_seconds=3 if episode == state.short_episode else 10,
                cameras=[]
                if episode == state.missing_camera
                else [
                    SimpleNamespace(
                        key=CAMERA,
                        start_seconds=start,
                        end_seconds=start + 10,
                        url=f"https://huggingface.co/datasets/fixture/robot/resolve/{SHA}/video.mp4",
                    )
                ],
            )

        async def read(self, url, max_bytes, budget):
            assert SHA in url
            assert max_bytes == 48 * 1024 * 1024
            return state.source_bytes

        async def close(self):
            state.explorer_closed = True

    class Provider:
        def __init__(self, key):
            assert key == "unpaid-fixture-key"

        async def edit(self, content, prompt):
            state.events.append(("paid_edit", content, prompt))
            state.started.set()
            if state.blocked:
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    if not state.late_success:
                        raise
            return b"augmented:" + content, "interaction-fixture"

        async def close(self):
            state.provider_closed = True

    async def trim(source, target, start, seconds):
        assert source.read_bytes() == state.source_bytes
        state.events.append(("trim", start, seconds))
        if state.trim_failure == start:
            raise ValueError("Camera media does not cover the requested clip interval")
        target.write_bytes(f"trimmed:{start}:{seconds}".encode())

    async def verify_output(path, seconds):
        assert path.read_bytes().startswith(b"augmented:trimmed:")
        assert seconds == 5
        if state.verification_failure:
            raise ValueError("Gemini changed the clip duration")

    monkeypatch.setattr(service, "DatasetExplorer", Explorer)
    monkeypatch.setattr(service, "GeminiOmniProvider", Provider)
    monkeypatch.setattr(service, "trim", trim)
    monkeypatch.setattr(service, "verify_output", verify_output)
    return state


def test_augmentation_persists_pinned_provenance_and_downloads_review_bundle(tmp_path, doubles):
    settings = Settings(data_dir=tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        pid = project(client)
        source = inspected_job(pid)
        seed(client, app, source)
        original = client.get("/api/v1/jobs/inspected").json()
        options = client.get("/api/v1/augmentation-options").json()
        assert options["configured"] is True and options["model"] == MODEL
        assert "unpaid-fixture-key" not in json.dumps(options)
        response = submit(client, pid)
        assert response.status_code == 202, response.text
        jid = response.json()["id"]
        completed = wait(client, jid)
        assert completed["status"] == "succeeded", completed
        assert completed["kind"] == completed["result"]["operation"] == "dataset.augment"
        result = completed["result"]
        assert result["review_required"] is True
        assert result["source_job_id"] == "inspected"
        assert result["repo_id"] == "fixture/robot"
        assert result["revision"] == SHA
        assert result["metadata_sha256"] == source.result.metadata_sha256
        assert "Preserve the exact robot" in result["prompt"]
        assert "matte light oak" in result["prompt"] and "Use blue fabric." in result["prompt"]
        assert len(result["warnings"]) >= 1
        # A v3 shared-video offset must be added to the user-selected episode-relative offset.
        assert [(c["source_start_seconds"], c["source_end_seconds"]) for c in result["clips"]] == [
            (1, 6),
            (21, 26),
        ]
        assert [event[0] for event in doubles.events] == [
            "preview",
            "trim",
            "preview",
            "trim",
            "paid_edit",
            "paid_edit",
        ]
        download = client.get(f"/api/v1/jobs/{jid}/augmentation/download")
        assert download.status_code == 200
        with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
            assert set(archive.namelist()) == {
                "manifest.json",
                "original-0.mp4",
                "augmented-0.mp4",
                "original-1.mp4",
                "augmented-1.mp4",
            }
            manifest = json.loads(archive.read("manifest.json"))
            assert manifest["result"] == result
            assert manifest["request"] == completed["request"]
            for clip in result["clips"]:
                index = clip["index"]
                raw = archive.read(f"original-{index}.mp4")
                edited = archive.read(f"augmented-{index}.mp4")
                assert hashlib.sha256(raw).hexdigest() == clip["input_sha256"]
                assert hashlib.sha256(edited).hexdigest() == clip["output_sha256"]
                assert clip["interaction_id"] == "interaction-fixture"
                assert (
                    client.get(f"/api/v1/jobs/{jid}/augmentation/clips/{index}").content == edited
                )
                assert (
                    client.get(
                        f"/api/v1/jobs/{jid}/augmentation/clips/{index}?original=true"
                    ).content
                    == raw
                )
        assert client.get(f"/api/v1/jobs/{jid}/augmentation/clips/2").status_code == 422
        assert client.get("/api/v1/jobs/inspected").json() == original
        assert doubles.source_bytes == b"source video bytes remain unchanged"
        assert doubles.provider_closed and doubles.explorer_closed
    with TestClient(create_app(settings)) as client:
        assert client.get(f"/api/v1/jobs/{jid}").json() == completed
        assert client.get(f"/api/v1/jobs/{jid}/augmentation/download").content == download.content
    assert sum(event[0] == "paid_edit" for event in doubles.events) == 2


@pytest.mark.parametrize("missing", ["key", "ffmpeg", "ffprobe"])
def test_missing_server_configuration_rejects_before_scheduling(
    tmp_path, monkeypatch, doubles, missing
):
    if missing == "key":
        monkeypatch.delenv("GEMINI_API_KEY")
    else:
        monkeypatch.setattr(
            service.shutil, "which", lambda binary: None if binary == missing else "/fixture/tool"
        )
    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as client:
        pid = project(client)
        seed(client, app, inspected_job(pid))
        assert client.get("/api/v1/augmentation-options").json()["configured"] is False
        response = submit(client, pid)
        assert response.status_code == 422
        assert "Configure" in response.json()["detail"]
        assert len(client.get(f"/api/v1/projects/{pid}/jobs").json()) == 1
    assert doubles.events == []


@pytest.mark.parametrize("api_key_present", [False, True])
def test_saved_cloud_login_is_used_without_gemini_key(
    tmp_path, monkeypatch, doubles, api_key_present
):
    if not api_key_present:
        monkeypatch.delenv("GEMINI_API_KEY")
    (tmp_path / "cloud-connections.json").write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {"gcp": {"project_id": "robotics-demo", "region": "us-central1"}},
            }
        )
    )

    class CloudProvider(service.GeminiOmniProvider):
        def __init__(self, *, google_cloud_project):
            assert google_cloud_project == "robotics-demo"

    monkeypatch.setattr(service, "GeminiOmniProvider", CloudProvider)
    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as client:
        pid = project(client)
        seed(client, app, inspected_job(pid))
        options = client.get("/api/v1/augmentation-options").json()
        assert options["configured"] is True
        assert options["auth_mode"] == "google_cloud"
        assert options["google_cloud_project"] == "robotics-demo"
        assert options["model"] == "gemini-omni-1.1-flash-preview"
        response = submit(client, pid)
        assert response.status_code == 202, response.text
        completed = wait(client, response.json()["id"])
        assert completed["status"] == "succeeded", completed
        assert completed["result"]["model"] == options["model"]
        assert completed["result"]["auth_mode"] == "google_cloud"
        assert completed["result"]["google_cloud_project"] == "robotics-demo"
        (tmp_path / "cloud-connections.json").unlink()
        options = client.get("/api/v1/augmentation-options").json()
        assert options["google_cloud_project"] is None
        assert options["auth_mode"] == ("gemini_api_key" if api_key_present else "unconfigured")


def test_saved_cloud_config_does_not_fall_back_to_key_when_cli_missing(
    tmp_path, monkeypatch, doubles
):
    (tmp_path / "cloud-connections.json").write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {"gcp": {"project_id": "robotics-demo", "region": "us-central1"}},
            }
        )
    )
    monkeypatch.setattr(
        service.shutil, "which", lambda name: None if name == "gcloud" else "/fixture/tool"
    )
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        options = client.get("/api/v1/augmentation-options").json()
        assert options["configured"] is False
        assert options["auth_mode"] == "google_cloud"
        assert "Google Cloud CLI" in options["setup_message"]


@pytest.mark.parametrize(
    "case,expected",
    [
        ("missing_source", "inspected dataset"),
        ("other_project", "in this project"),
        ("unfinished", "successful dataset inspection"),
        ("local", "public Hugging Face"),
        ("wrong_kind", "dataset inspection"),
        ("camera_missing", "video camera"),
        ("camera_not_video", "video camera"),
        ("episode_outside", "outside the inspected dataset"),
    ],
)
def test_invalid_source_and_selection_never_schedule_paid_work(tmp_path, doubles, case, expected):
    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as client:
        pid = project(client)
        record = inspected_job(project(client) if case == "other_project" else pid)
        changes = {}
        if case == "unfinished":
            record.status, record.result = "failed", None
        elif case == "local":
            record.request = IntakeRequest(source="local", path="dataset")
            record.result.source = "local"
            record.result.repo_id = None
            record.result.revision = f"metadata-sha256:{record.result.metadata_sha256}"
        elif case == "wrong_kind":
            record.kind = "policy.import"
            record.request = PolicyRequest(
                operation="policy.import", runtime_id="fixture", source_id="fixture"
            )
            record.result = None
        elif case == "camera_missing":
            changes["camera_key"] = "missing"
        elif case == "camera_not_video":
            changes["camera_key"] = "observation.state"
        elif case == "episode_outside":
            changes["episode_indices"] = [3]
        if case != "missing_source":
            seed(client, app, record)
        response = submit(client, pid, **changes)
        assert response.status_code == 422, response.text
        assert expected in response.json()["detail"]
        assert not app.state.execution.tasks
    assert doubles.events == []


@pytest.mark.parametrize(
    "changes",
    [
        {"episode_indices": []},
        {"episode_indices": [0, 0]},
        {"episode_indices": [0, 1, 2, 3, 4]},
        {"episode_indices": [-1]},
        {"episode_indices": [True]},
        {"episode_indices": [0.5]},
        {"start_seconds": -1},
        {"duration_seconds": 0},
        {"duration_seconds": 11},
        {"preset": "custom", "prompt": "   "},
        {"camera_key": " "},
        {"model": "unreviewed-provider"},
    ],
)
def test_request_bounds_rejected_at_api_boundary(tmp_path, doubles, changes):
    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as client:
        pid = project(client)
        response = submit(client, pid, **changes)
        assert response.status_code == 422
        assert client.get(f"/api/v1/projects/{pid}/jobs").json() == []
    assert doubles.events == []


@pytest.mark.parametrize("failure", ["missing_camera", "short_episode", "trim_failure"])
def test_entire_selection_is_preflighted_before_first_paid_edit(tmp_path, doubles, failure):
    setattr(doubles, failure, 21 if failure == "trim_failure" else 2)
    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as client:
        pid = project(client)
        seed(client, app, inspected_job(pid))
        response = submit(client, pid)
        assert response.status_code == 202
        jid = response.json()["id"]
        failed = wait(client, jid)
        assert failed["status"] == "failed"
        assert failed["result"] is None
        assert client.get(f"/api/v1/jobs/{jid}/augmentation/download").status_code == 422
        assert (
            client.get(f"/api/v1/jobs/{jid}/augmentation/clips/0?original=true").status_code == 422
        )
    assert any(event[0] == "trim" for event in doubles.events)
    assert not any(event[0] == "paid_edit" for event in doubles.events)
    assert doubles.provider_closed and doubles.explorer_closed


def test_invalid_generated_timing_cannot_publish_a_bundle(tmp_path, doubles):
    doubles.verification_failure = True
    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as client:
        pid = project(client)
        seed(client, app, inspected_job(pid))
        jid = submit(client, pid).json()["id"]
        failed = wait(client, jid)
        assert failed["status"] == "failed"
        assert "duration" in failed["error"]
        assert failed["result"] is None
        assert client.get(f"/api/v1/jobs/{jid}/augmentation/download").status_code == 422
    assert sum(event[0] == "paid_edit" for event in doubles.events) == 1


@pytest.mark.parametrize("late_success", [False, True])
@pytest.mark.parametrize("episode_indices", [[0], [0, 1]])
def test_cancelled_augmentation_never_publishes_even_a_late_result(
    tmp_path, doubles, late_success, episode_indices
):
    doubles.blocked, doubles.late_success = True, late_success
    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as client:
        pid = project(client)
        seed(client, app, inspected_job(pid))
        jid = submit(client, pid, episode_indices=episode_indices).json()["id"]
        assert doubles.started.wait(5)
        response = client.post(f"/api/v1/jobs/{jid}/cancel")
        assert response.json()["status"] == "cancelled"
        assert "may still run and incur charges" in response.json()["error"]
        cancelled = wait(client, jid)
        assert cancelled["result"] is None
        assert client.get(f"/api/v1/jobs/{jid}/augmentation/download").status_code == 422
        assert client.get(f"/api/v1/jobs/{jid}/augmentation/clips/0").status_code == 422
        assert client.post(f"/api/v1/jobs/{jid}/cancel").json() == cancelled
    assert doubles.provider_closed and doubles.explorer_closed
    assert sum(event[0] == "paid_edit" for event in doubles.events) == 1


def test_restart_interrupts_without_retrying_or_adopting_augmentation_outputs(tmp_path, doubles):
    doubles.blocked = True
    settings = Settings(data_dir=tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        pid = project(client)
        seed(client, app, inspected_job(pid))
        jid = submit(client, pid, episode_indices=[0]).json()["id"]
        assert doubles.started.wait(5)
    # Even a plausible late file on disk must never become a published result after restart.
    (tmp_path / "jobs" / jid / "augmentation.zip").write_bytes(b"late remote output")
    with TestClient(create_app(settings)) as client:
        record = client.get(f"/api/v1/jobs/{jid}").json()
        assert record["status"] == "interrupted"
        assert record["result"] is None
        assert "paid requests retried" in record["error"]
        assert client.get(f"/api/v1/jobs/{jid}/augmentation/download").status_code == 422
    assert sum(event[0] == "paid_edit" for event in doubles.events) == 1


def augmentation_result():
    return AugmentationResult(
        prompt="Change lighting",
        source_job_id="inspected",
        repo_id="fixture/robot",
        revision=SHA,
        metadata_sha256="b" * 64,
        clips=[
            {
                "index": 0,
                "episode_index": 0,
                "camera_key": CAMERA,
                "source_start_seconds": 0,
                "source_end_seconds": 5,
                "input_sha256": "c" * 64,
                "output_sha256": "d" * 64,
            }
        ],
        warnings=["Review required"],
    )


@pytest.mark.parametrize("family", ["inspection", "policy", "augmentation"])
def test_persisted_jobs_reject_a_result_from_another_operation_family(family):
    record = inspected_job("p")
    values = record.model_dump()
    if family == "inspection":
        values["result"] = augmentation_result()
    elif family == "policy":
        values.update(
            kind="policy.import",
            request=PolicyRequest(
                operation="policy.import", runtime_id="fixture", source_id="fixture"
            ),
            result=augmentation_result(),
        )
    else:
        values.update(
            kind="dataset.augment",
            request=AugmentationRequest(**request_body()),
            result=LifecycleResult(),
        )
    with pytest.raises(ValidationError, match="result must match its operation family"):
        Job.model_validate(values)


def test_persisted_jobs_reject_mismatched_augmentation_operation():
    values = inspected_job("p").model_dump()
    values.update(request=AugmentationRequest(**request_body()), result=None)
    with pytest.raises(ValidationError, match="kind must match"):
        Job.model_validate(values)
