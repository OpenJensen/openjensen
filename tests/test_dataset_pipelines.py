"""Cleaning a dataset: check every episode, review the verdicts, save the rest.

The steps run on real LeRobot files (Parquet and MP4) and the saved dataset must pass
the complete training validator, augmented copies included.
"""

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from test_dataset_combine import save, table
from test_dataset_snapshots import build_dataset
from vla_platform.api import create_app
from vla_platform.datasets.local_preview import reader_python
from vla_platform.datasets.snapshots import create_snapshot
from vla_platform.pipelines import vision, worker
from vla_platform.settings import Settings

pytestmark = pytest.mark.skipif(os.name == "nt", reason="Secure snapshots require POSIX")


def trace(*segments):
    """Commanded and measured gripper positions from (frames, commanded, stall) pieces.

    ``stall`` is how far short of the command the gripper stops: about one unit when it
    closes on nothing, several when an object is in the way.
    """
    commanded, measured = [], []
    for frames, value, stall in segments:
        commanded += [value] * frames
        measured += [value + stall] * frames
    return commanded, measured


def test_gripper_trace_finds_missed_grasps_drops_and_unfinished_episodes():
    clean = trace((10, 0.5, 1), (20, 40, 1), (30, 0.5, 9), (20, 40, 1), (20, 0.5, 1))
    assert worker.grasp_events(*clean)["verdict"] == "clean"
    retried = trace((10, 0.5, 1), (20, 40, 1), (25, 0.5, 1), (20, 40, 1), (30, 0.5, 9), (20, 40, 1))
    label = worker.grasp_events(*retried)
    assert (label["verdict"], label["missed_grasps"], label["closes"]) == ("regrasp", 1, 2)
    never = trace((10, 0.5, 1), (20, 40, 1), (30, 0.5, 1))
    assert worker.grasp_events(*never)["verdict"] == "no_grasp"
    still_holding = trace((10, 0.5, 1), (20, 40, 1), (40, 0.5, 9))
    assert worker.grasp_events(*still_holding)["verdict"] == "not_released"
    # Opening only partly still lets the object go.
    partly = trace((10, 0.5, 1), (20, 40, 1), (30, 0.5, 9), (10, 15, 0), (20, 0.5, 1))
    assert worker.grasp_events(*partly)["verdict"] == "clean"


def frame_files(tmp_path, entry):
    frames = tmp_path / "frames" / entry
    frames.mkdir(parents=True)
    for number in range(2):
        (frames / f"front-{number:02d}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    return [
        {"path": str(frames / "front-00.jpg"), "seconds": 0.5, "event": None},
        {"path": str(frames / "front-01.jpg"), "seconds": 1.2, "event": "after gripper close 1"},
    ]


def test_vision_review_sends_every_frame_and_reads_the_verdict(tmp_path, monkeypatch):
    sent = []

    def answer(request):
        body = json.loads(request.content)
        sent.append(body)
        verdict = {
            "task_completed": True,
            "grasp_attempts": 1,
            "first_attempt_succeeded": True,
            "object_dropped": False,
            "outcome": "success",
            "confidence": 0.9,
            "notes": "The object ends in the container.",
        }
        return httpx.Response(
            200,
            json={
                "output": [{"content": [{"type": "output_text", "text": json.dumps(verdict)}]}],
                "usage": {"input_tokens": 1000, "output_tokens": 50},
            },
        )

    real = httpx.AsyncClient
    monkeypatch.setattr(
        vision.httpx, "AsyncClient", lambda: real(transport=httpx.MockTransport(answer))
    )
    entries = [
        {
            "id": "e0000",
            "task": "Put the cube in the box",
            "labels": {
                "frames": {"stills": {"observation.images.front": frame_files(tmp_path, "e0000")}}
            },
        }
    ]
    summary = asyncio.run(
        vision.review(entries, "test-key", {"model": "gpt-5.5", "camera_notes": {"front": "x"}})
    )
    assert summary["outcomes"] == {"success": 1}
    assert summary["cost_usd"] == pytest.approx(1000 / 1e6 * 5 + 50 / 1e6 * 30)
    assert entries[0]["labels"]["vision"]["outcome"] == "success"
    content = sent[0]["input"][0]["content"]
    assert sum(part["type"] == "input_image" for part in content) == 2
    assert any("after gripper close 1" in part.get("text", "") for part in content)
    assert "Put the cube in the box" in content[0]["text"]
    assert sent[0]["store"] is False and sent[0]["text"]["format"]["strict"] is True


def test_decisions_api_answers_fixed_questions_with_probabilities(tmp_path, monkeypatch):
    sent = []

    def answer(request):
        sent.append((str(request.url), json.loads(request.content)))
        return httpx.Response(
            200,
            json={
                "answers": [
                    {
                        "type": "choice",
                        "name": "outcome",
                        "choice": "failed",
                        "probabilities": [
                            {"value": "success", "probability": 0.05},
                            {"value": "success_after_retry", "probability": 0.05},
                            {"value": "failed", "probability": 0.9},
                        ],
                        "confidence": 0.88,
                    },
                    {"type": "predicate", "name": "task_completed", "probability": 0.1},
                    {"type": "predicate", "name": "first_attempt_succeeded", "probability": 0.2},
                    {"type": "predicate", "name": "object_dropped", "probability": 0.7},
                    {
                        "type": "choice",
                        "name": "grasp_attempts",
                        "choice": "3+",
                        "probabilities": [],
                        "confidence": 0.6,
                    },
                ],
                "usage": {"input_tokens": 20000},
            },
        )

    real = httpx.AsyncClient
    monkeypatch.setattr(
        vision.httpx, "AsyncClient", lambda: real(transport=httpx.MockTransport(answer))
    )
    entries = [
        {
            "id": "e0000",
            "task": "Put the cube in the box",
            "labels": {
                "frames": {"stills": {"observation.images.front": frame_files(tmp_path, "e0000")}}
            },
        }
    ]
    summary = asyncio.run(vision.decide(entries, "test-key", {"camera_notes": {"front": "x"}}))
    verdict = entries[0]["labels"]["vision"]
    assert (verdict["outcome"], verdict["grasp_attempts"]) == ("failed", 3)
    assert verdict["object_dropped"] and not verdict["task_completed"]
    assert summary["cost_usd"] == pytest.approx(20000 / 1e6 * 0.10)
    url, body = sent[0]
    assert url.endswith("/v1/decisions") and body["model"] == "gpt-6-luna"
    assert {q["name"] for q in body["questions"]} >= {"outcome", "grasp_attempts"}
    images = [p for p in body["input"][0]["content"] if p["type"] == "input_image"]
    assert len(images) == 2 and images[0]["image_url"].startswith("data:image/jpeg;base64,")


def wait(client, project, ident):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        run = client.get(f"/api/v1/projects/{project}/dataset-pipelines/{ident}").json()
        if run["status"] not in {"queued", "running"}:
            return run
        time.sleep(0.1)
    raise AssertionError("Pipeline did not finish")


def test_check_review_augment_and_save_a_dataset_the_trainer_accepts(tmp_path, monkeypatch):
    """The journey through the same requests the review page sends."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    source = build_dataset(tmp_path / "recorded", episodes=4)
    settings = Settings(data_dir=tmp_path / "workspace", local_root=tmp_path)
    with TestClient(create_app(settings)) as client:
        project = client.post("/api/v1/projects", json={"name": "Cube"}).json()["id"]
        dataset = save(client, project, "cube", source)
        root = f"/api/v1/projects/{project}/dataset-pipelines"
        created = client.post(root, json={"dataset_id": dataset["id"], "recipe": "check"})
        assert created.status_code == 201, created.text
        run = wait(client, project, created.json()["id"])
        assert run["status"] == "succeeded", run["message"]
        processors = [step["processor"] for step in run["steps"]]
        assert processors == [
            "read_dataset",
            "gripper_events",
            "sample_frames",
            "vision_review",
            "keep_where",
        ]
        # Without an OpenAI key the vision review says so instead of failing the run.
        assert "OpenAI API key" in run["results"][3]["summary"]["skipped"]
        assert "no gripper" in run["results"][1]["summary"]["skipped"]
        # Every step wrote its own manifest.
        manifest = client.get(f"{root}/{run['id']}/steps/0/manifest").json()["entries"]
        assert [entry["id"] for entry in manifest] == ["e0000", "e0001", "e0002", "e0003"]

        view = client.get(f"{root}/{run['id']}/episodes").json()
        assert len(view["episodes"]) == 4 and all(row["keep"] for row in view["episodes"])
        media = f"{root}/{run['id']}/media"
        strip = client.get(media, params={"path": view["episodes"][0]["strip"]})
        assert strip.status_code == 200 and strip.headers["content-type"] == "image/jpeg"
        assert client.get(media, params={"path": "run.json"}).status_code == 404
        assert client.get(media, params={"path": "../../x.jpg"}).status_code == 404

        reviewed = client.put(f"{root}/{run['id']}/review", json={"decisions": {"e0001": False}})
        assert reviewed.json()["review"] == {"e0001": False}
        saving = client.post(
            f"{root}/{run['id']}/save", json={"name": "cube-cleaned", "augment_copies": 1}
        )
        assert saving.status_code == 202, saving.text
        run = wait(client, project, run["id"])
        assert run["status"] == "succeeded", run["message"]
        saved = run["saved_datasets"][0]
        assert (saved["episodes"], saved["augmented"]) == (6, 3)
        # Steps before the review were not repeated.
        assert run["results"][0]["finished_at"] < run["results"][5]["started_at"]

        library = client.get(f"/api/v1/datasets/{saved['dataset_id']}").json()
        assert library["status"] == "ready" and library["name"] == "cube-cleaned"
        folder = settings.data_dir / "dataset-library" / saved["dataset_id"] / "source"
        snapshot = create_snapshot(tmp_path, folder, tmp_path / "snapshots")
        assert snapshot["total_episodes"] == 6 and snapshot["total_frames"] == 18
        episodes = table(folder / "meta/episodes/chunk-000/file-000.parquet")
        assert [e["dataset_from_index"] for e in episodes] == [0, 3, 6, 9, 12, 15]
        lineage = json.loads((folder / "meta/firebird-lineage.json").read_text())["episodes"]
        assert [item["origin"] for item in lineage] == ["imported"] * 3 + ["augmented"] * 3
        groups = [item["lineage_group"].split(":")[1] for item in lineage]
        assert groups == ["0", "2", "3", "0", "2", "3"]
        assert snapshot["lineage_validated"] is True
        # The dropped episode's frames are gone; kept frames are copied unchanged.
        third = table(folder / "data/chunk-000/file-001.parquet")
        assert [row["index"] for row in third] == [3, 4, 5]
        assert len(list((folder / "videos").rglob("*.mp4"))) == 6


def test_generative_edits_are_kept_only_when_the_pose_check_passes(tmp_path, monkeypatch):
    """A repainted clip that redraws the arm must never reach a training set."""
    from vla_platform.pipelines import generative

    source = build_dataset(tmp_path / "recorded", episodes=2)
    manifest = tmp_path / "read.jsonl"
    code = subprocess.run(
        [str(reader_python()), "-I", "-B", str(Path(worker.__file__))],
        input=json.dumps({"operation": "read", "root": str(source), "output": str(manifest)}),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert code.returncode == 0, code.stdout
    entries = [json.loads(line) for line in manifest.read_text().splitlines()]

    class Same:
        async def edit(self, video, prompt):
            assert "Change only" in prompt or "lighting" in prompt
            return video, None

    verdicts = iter([[0.9] * 5, [0.9, 0.9, 0.1, 0.9, 0.9]])

    def answer(request):
        body = json.loads(request.content)
        assert all(len(q["instructions"]) == 5 for q in body["questions"])
        probabilities = next(verdicts)
        return httpx.Response(
            200,
            json={
                "answers": [
                    {"type": "predicate", "name": q["name"], "probability": p}
                    for q, p in zip(body["questions"], probabilities, strict=True)
                ],
                "usage": {"input_tokens": 1000},
            },
        )

    real = httpx.AsyncClient
    monkeypatch.setattr(
        generative.httpx, "AsyncClient", lambda: real(transport=httpx.MockTransport(answer))
    )
    summary = asyncio.run(
        generative.edit_entries(
            entries,
            {"count": 2, "size": [16, 16], "concurrency": 1},
            tmp_path / "edits",
            provider=Same(),
            decisions_key="test-key",
        )
    )
    assert (summary["edited"], summary["accepted"]) == (2, 1), summary["errors"]
    added = [e for e in entries if e["origin"] == "augmented"]
    assert len(added) == 1 and added[0]["augmentation"]["method"] == "gemini_omni"
    edited = Path(added[0]["videos"]["observation.images.front"]["path"])
    assert worker.frame_count(edited) == added[0]["length"]
    assert summary["rejected"][0]["gate"][2] == 0.1
