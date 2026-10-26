"""Full-dataset coverage and camera selection tested without any real model calls."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from test_augmentation import (
    CAMERA,
    SHA,
    inspected_job,
    project,
    seed,
    submit,
    wait,
)
from test_augmentation import (
    doubles as augmentation_doubles,
)
from vla_platform.api import create_app
from vla_platform.augmentation import service
from vla_platform.augmentation.contracts import AugmentationRequest
from vla_platform.settings import Settings

FRONT = "observation.images.front"


@pytest.fixture
def doubles(monkeypatch):
    return augmentation_doubles.__wrapped__(monkeypatch)


def make_recipe(client, app, pid):
    source = inspected_job(pid)
    source.result.features[FRONT] = {"dtype": "video", "shape": [3, 240, 320]}
    seed(client, app, source)
    parent = wait(client, submit(client, pid).json()["id"])
    assert parent["status"] == "succeeded", parent
    return parent


def full_request(recipe, cameras=None):
    return {
        **recipe["request"],
        "episode_indices": [],
        "camera_key": (cameras or [CAMERA])[0],
        "start_seconds": 0,
        "duration_seconds": 10,
        "dataset_scope": {"recipe_job_id": recipe["id"], "camera_keys": cameras or [CAMERA]},
    }


def full_doubles(monkeypatch, state, lengths=(21, 5, 11)):
    original_explorer = service.DatasetExplorer
    original_download = service.download_video

    class Explorer(original_explorer):
        async def preview(self, source, episode):
            state.events.append(("full_preview", episode))
            return SimpleNamespace(
                duration_seconds=lengths[episode],
                cameras=[
                    SimpleNamespace(
                        key=key,
                        start_seconds=episode * 100,
                        end_seconds=episode * 100 + lengths[episode],
                        url=f"https://huggingface.co/datasets/fixture/robot/resolve/{SHA}/{key}.mp4",
                    )
                    for key in [CAMERA, FRONT]
                    if key != getattr(state, "absent", None)
                ],
            )

    async def download(client, url, target):
        state.events.append(("full_download", url))
        await original_download(client, url, target)

    async def verify(path, seconds):
        assert path.read_bytes().startswith(b"augmented:")
        assert 1 <= seconds <= 10
        state.events.append(("verified", seconds))

    monkeypatch.setattr(service, "DatasetExplorer", Explorer)
    monkeypatch.setattr(service, "download_video", download)
    monkeypatch.setattr(service, "verify_output", verify)


def test_plan_is_unpaid_and_full_generation_covers_only_selected_cameras(
    tmp_path, doubles, monkeypatch
):
    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as client:
        pid = project(client)
        recipe = make_recipe(client, app, pid)
        full_doubles(monkeypatch, doubles)
        body = full_request(recipe, [FRONT, CAMERA])
        before = len([e for e in doubles.events if e[0] == "paid_edit"])
        planned = client.post(f"/api/v1/projects/{pid}/augmentations/plan", json=body)
        assert planned.status_code == 200, planned.text
        plan = planned.json()
        assert plan["total_episodes"] == 3
        assert plan["camera_keys"] == [FRONT, CAMERA]
        assert plan["total_clips"] == plan["provider_requests"] == 12
        assert plan["total_source_seconds"] == 74
        assert len([e for e in doubles.events if e[0] == "paid_edit"]) == before
        assert not [e for e in doubles.events if e[0] == "full_download"]
        assert client.post(f"/api/v1/projects/{pid}/augmentations", json=body).status_code == 422
        body["dataset_scope"]["plan_sha256"] = plan["plan_sha256"]
        response = client.post(f"/api/v1/projects/{pid}/augmentations", json=body)
        assert response.status_code == 202, response.text
        job = wait(client, response.json()["id"])
        assert job["status"] == "succeeded", job
        clips = job["result"]["clips"]
        assert len(clips) == 12
        for episode, length in enumerate([21, 5, 11]):
            for key in [FRONT, CAMERA]:
                parts = [
                    c for c in clips if c["episode_index"] == episode and c["camera_key"] == key
                ]
                assert parts[0]["source_start_seconds"] == episode * 100
                assert parts[-1]["source_end_seconds"] == episode * 100 + length
                assert all(
                    a["source_end_seconds"] == b["source_start_seconds"]
                    for a, b in zip(parts, parts[1:])
                )
        assert len([e for e in doubles.events if e[0] == "paid_edit"]) == before + 12
        assert len([e for e in doubles.events if e[0] == "full_download"]) == 2
        # Every source is trimmed before the first batch provider call.
        batch_events = doubles.events[
            doubles.events.index(next(e for e in doubles.events if e[0] == "full_download")) :
        ]
        first_edit = next(i for i, e in enumerate(batch_events) if e[0] == "paid_edit")
        assert sum(e[0] == "trim" for e in batch_events[:first_edit]) == 12
        saved = json.loads((tmp_path / "jobs" / job["id"] / "manifest.json").read_text())
        assert saved["result"]["dataset_scope"] == body["dataset_scope"]
        assert client.get(f"/api/v1/jobs/{job['id']}/augmentation/download").status_code == 200


@pytest.mark.parametrize(
    "case",
    [
        "missing_camera",
        "stale_plan",
        "wrong_project",
        "changed_prompt",
        "no_cameras",
        "duplicate_camera",
    ],
)
def test_invalid_dataset_plans_make_no_generation_calls(tmp_path, doubles, monkeypatch, case):
    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as client:
        pid = project(client)
        recipe = make_recipe(client, app, pid)
        full_doubles(monkeypatch, doubles)
        body = full_request(recipe)
        before = len([e for e in doubles.events if e[0] == "paid_edit"])
        if case == "missing_camera":
            doubles.absent = CAMERA
        elif case == "wrong_project":
            pid = project(client)
        elif case == "changed_prompt":
            body["prompt"] = "A different edit"
        elif case == "no_cameras":
            body["dataset_scope"]["camera_keys"] = []
        elif case == "duplicate_camera":
            body["dataset_scope"]["camera_keys"] = [CAMERA, CAMERA]
        if case == "stale_plan":
            body["dataset_scope"]["plan_sha256"] = "0" * 64
            response = client.post(f"/api/v1/projects/{pid}/augmentations", json=body)
            assert response.status_code == 202
            job = wait(client, response.json()["id"])
            assert job["status"] == "failed"
            assert "plan changed" in job["error"]
        else:
            assert (
                client.post(f"/api/v1/projects/{pid}/augmentations/plan", json=body).status_code
                == 422
            )
        assert len([e for e in doubles.events if e[0] == "paid_edit"]) == before


def test_single_camera_and_refinement_plan_replay_the_saved_recipe_chain(
    tmp_path, doubles, monkeypatch
):
    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as client:
        pid = project(client)
        parent = make_recipe(client, app, pid)
        refined = wait(
            client,
            submit(
                client,
                pid,
                episode_indices=[0],
                source_clip={"job_id": parent["id"], "clip_index": 0, "variant": "augmented"},
                prompt="Make the cup red.",
            ).json()["id"],
        )
        full_doubles(monkeypatch, doubles)
        body = full_request(refined, [FRONT])
        body.pop("source_clip", None)
        plan = client.post(f"/api/v1/projects/{pid}/augmentations/plan", json=body).json()
        assert plan["total_clips"] == 6 and plan["provider_requests"] == 12
        assert plan["recipe_prompts"] == [parent["result"]["prompt"], refined["result"]["prompt"]]
        body["dataset_scope"]["plan_sha256"] = plan["plan_sha256"]
        result = wait(
            client, client.post(f"/api/v1/projects/{pid}/augmentations", json=body).json()["id"]
        )
        assert result["status"] == "succeeded", result
        assert {c["camera_key"] for c in result["result"]["clips"]} == {FRONT}


def test_batch_cancellation_prevents_further_paid_requests(tmp_path, doubles, monkeypatch):
    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as client:
        pid = project(client)
        recipe = make_recipe(client, app, pid)
        full_doubles(monkeypatch, doubles)
        body = full_request(recipe)
        plan = client.post(f"/api/v1/projects/{pid}/augmentations/plan", json=body).json()
        body["dataset_scope"]["plan_sha256"] = plan["plan_sha256"]
        before = len([e for e in doubles.events if e[0] == "paid_edit"])
        doubles.blocked = True
        doubles.started.clear()
        job = client.post(f"/api/v1/projects/{pid}/augmentations", json=body).json()
        assert doubles.started.wait(3)
        client.post(f"/api/v1/jobs/{job['id']}/cancel")
        assert wait(client, job["id"])["status"] == "cancelled"
        assert len([e for e in doubles.events if e[0] == "paid_edit"]) == before + 1


def test_legacy_preview_fingerprint_omits_batch_scope():
    request = AugmentationRequest(source_job_id="source", episode_indices=[0], camera_key=CAMERA)
    assert "dataset_scope" not in request.model_dump(mode="json")
