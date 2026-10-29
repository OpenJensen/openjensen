"""Dataset-wide scene evidence stays pinned, agent-selected and complete for replay."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from vla_platform.contracts import CameraPreview, IntakeRequest, Job, now
from vla_platform.datasets.inspect import profile
from vla_platform.scenes import dataset
from vla_platform.scenes.agent import SceneError
from vla_platform.scenes.contracts import (
    AgentSettings,
    ClipRequest,
    Investigation,
    SceneRequest,
    SceneRun,
)
from vla_platform.scenes.service import Scenes


def source():
    request = IntakeRequest(repo_id="fixture/scene-dataset")
    raw = json.dumps(
        {
            "codebase_version": "v3.0",
            "total_episodes": 30,
            "total_frames": 4500,
            "fps": 30,
            "features": {
                "observation.images.front": {"dtype": "video", "shape": [1080, 1920, 3]},
                "observation.state": {
                    "dtype": "float32",
                    "shape": [6],
                    "names": [name + ".pos" for name in dataset.JOINTS],
                },
                "action": {"dtype": "float32", "shape": [6]},
            },
        }
    ).encode()
    return Job(
        id="source",
        project_id="project",
        status="succeeded",
        request=request,
        result=profile(raw, request, "a" * 40),
        created_at=now(),
        updated_at=now(),
    )


def clip(episode=17, camera="observation.images.front", start=0, end=1):
    return ClipRequest(
        episode_index=episode,
        camera_key=camera,
        start_fraction=start,
        end_fraction=end,
        reason="Inspect workspace and arm motion",
    )


def decision(*clips, ready=False, replay=17):
    return Investigation(
        ready=ready,
        clips=list(clips),
        replay_episode=replay,
        observations=["Fixture evidence"],
        uncertainties=["Joint calibration is unverified"],
    )


def prepared(tmp_path, monkeypatch, decisions=None):
    scenes = Scenes(tmp_path)
    request = SceneRequest(request_id="c" * 32, source_job_id="source", model="fixture-scene")
    folder = scenes.folder(request.request_id)
    folder.mkdir(parents=True)
    job_source = source()
    run = SceneRun(
        id=request.request_id,
        job_id="fixture-job",
        project_id="project",
        request=request,
        agent=AgentSettings(model="fixture-scene"),
        status="running",
        stage="evidence",
        message="Fixture",
        created_at=now(),
        updated_at=now(),
        source={
            "repo_id": job_source.result.repo_id,
            "revision": job_source.result.revision,
            "metadata_sha256": job_source.result.metadata_sha256,
        },
        gates={"source_evidence": False},
    )
    rows = [
        {"episode_index": index, "length": 150, "tasks": ["Fixture task"]} for index in range(30)
    ]
    observed = {"downloads": [], "roles": [], "reads": [], "cameras": [], "episodes": []}

    async def metadata(*args):
        return {}

    async def episodes(profile, budget, start, count):
        observed["episodes"].append((start, count))
        return rows[start : start + count]

    async def cameras(profile, info, row, item, budget, warnings, **kwargs):
        observed["cameras"].append(row["episode_index"])
        return [
            CameraPreview(
                key="observation.images.front",
                url=(
                    "https://huggingface.co/datasets/fixture/scene-dataset/resolve/"
                    f"{'a' * 40}/videos/{row['episode_index']}.mp4"
                ),
                start_seconds=10,
                end_seconds=15,
                width=1920,
                height=1080,
                fps=30,
            )
        ]

    async def read(*args):
        observed["reads"].append(args)
        return b"fixture-parquet"

    explorer = SimpleNamespace(
        metadata=metadata,
        episodes=episodes,
        cameras=cameras,
        reader=SimpleNamespace(read=read),
        data_path=lambda profile, info, row: f"data/episode-{row['episode_index']}.parquet",
        client=object(),
        reader_python="fixture-reader",
    )

    async def download(client, url, target):
        observed["downloads"].append(url)
        target.write_bytes(b"fixture-video" * 100)

    async def command(args, **kwargs):
        from pathlib import Path

        target = Path(args[-1])
        target.write_bytes(b"fixture-frame" * 100)
        return b""

    supplied = list(decisions or [decision(clip()), decision(ready=True)])

    async def role(work, name, schema, prompt, frames, data_dir, **kwargs):
        observed["roles"].append({"prompt": prompt, "frames": list(frames)})
        result = supplied.pop(0)
        (work / (name + ".json")).write_text(result.model_dump_json())
        return result, {"input_tokens": 7, "output_tokens": 3}

    trajectory = [
        {"frame_index": index, "timestamp": index / 30, "state": [-25, -104, 90, 75, 4, 1]}
        for index in range(150)
    ]

    async def parquet(raw, operation, *, episode_index, python):
        assert raw == b"fixture-parquet" and operation == "trajectory"
        assert episode_index == 17 and python == "fixture-reader"
        return trajectory

    monkeypatch.setattr(dataset, "download_video", download)
    monkeypatch.setattr(dataset, "command", command)
    monkeypatch.setattr(dataset, "run_role", role)
    monkeypatch.setattr(dataset, "read_parquet", parquet)
    return scenes, run, job_source, explorer, observed, trajectory, rows


def test_agent_sees_all_episodes_but_only_selected_video_is_downloaded(tmp_path, monkeypatch):
    scenes, run, source_job, explorer, observed, trajectory, _ = prepared(tmp_path, monkeypatch)

    async def exercise():
        frames, evidence = await dataset.investigate(scenes, run, source_job, explorer)
        assert observed["episodes"] == [(0, 30)]
        assert observed["cameras"] == list(range(30))
        assert len(observed["downloads"]) == 1 and observed["downloads"][0].endswith("/17.mp4")
        assert len(observed["roles"]) == 2
        assert observed["roles"][0]["frames"] == []
        assert len(observed["roles"][1]["frames"]) == 5
        assert len(frames) == 5 and evidence["dataset_catalog"]["total_episodes"] == 30
        assert evidence["replay"]["episode_index"] == 17
        assert run.source["inspected_episodes"] == [17]
        assert run.source["replay_frames"] == len(trajectory)
        assert run.gates["source_evidence"] is True
        assert run.usage == {"input_tokens": 14, "output_tokens": 6}
        folder = scenes.folder(run.id)
        assert not list(folder.glob(".source-*.mp4"))
        saved = json.loads((folder / "trajectory.json").read_text())
        assert (
            len(saved["frames"]) == 150
            and saved["mapping_status"] == "uncalibrated_dataset_normalized"
        )
        assert len(json.loads((folder / "dataset-catalog.json").read_text())["episodes"]) == 30

    asyncio.run(exercise())


def test_incomplete_catalog_stops_before_agent_or_video_work(tmp_path, monkeypatch):
    scenes, run, source_job, explorer, observed, _, rows = prepared(tmp_path, monkeypatch)
    rows.pop()

    async def exercise():
        with pytest.raises(SceneError, match="complete dataset"):
            await dataset.investigate(scenes, run, source_job, explorer)
        assert not observed["roles"] and not observed["downloads"]

    asyncio.run(exercise())


def test_catalog_cannot_silently_omit_an_advertised_video_camera(tmp_path, monkeypatch):
    scenes, run, source_job, explorer, observed, _, _ = prepared(tmp_path, monkeypatch)
    source_job.result.features["observation.images.side"] = {
        "dtype": "video",
        "shape": [1080, 1920, 3],
    }

    async def exercise():
        with pytest.raises(SceneError, match="camera"):
            await dataset.investigate(scenes, run, source_job, explorer)
        assert not observed["roles"] and not observed["downloads"]

    asyncio.run(exercise())


@pytest.mark.parametrize("requested", [clip(31), clip(camera="not-a-camera")])
def test_agent_cannot_download_a_video_outside_the_pinned_catalog(tmp_path, monkeypatch, requested):
    scenes, run, source_job, explorer, observed, _, _ = prepared(
        tmp_path, monkeypatch, [decision(requested)]
    )

    async def exercise():
        with pytest.raises(SceneError, match="outside the dataset catalog"):
            await dataset.investigate(scenes, run, source_job, explorer)
        assert not observed["downloads"] and not observed["reads"]

    asyncio.run(exercise())


def test_replay_episode_must_have_been_inspected(tmp_path, monkeypatch):
    scenes, run, source_job, explorer, observed, _, _ = prepared(
        tmp_path, monkeypatch, [decision(clip()), decision(ready=True, replay=29)]
    )

    async def exercise():
        with pytest.raises(SceneError, match="inspected"):
            await dataset.investigate(scenes, run, source_job, explorer)
        assert not observed["reads"]
        assert not list(scenes.folder(run.id).glob(".source-*.mp4"))

    asyncio.run(exercise())


def test_final_round_cannot_accept_new_video_evidence_the_agent_never_viewed(tmp_path, monkeypatch):
    rounds = dataset.investigation_rounds(30)
    selections = [
        decision(clip(17, start=index / rounds, end=(index + 1) / rounds))
        for index in range(rounds - 1)
    ]
    selections.append(decision(clip(17, start=(rounds - 1) / rounds), ready=True))
    scenes, run, source_job, explorer, observed, _, _ = prepared(
        tmp_path,
        monkeypatch,
        selections,
    )

    async def exercise():
        with pytest.raises(SceneError, match="budget|evidence|view"):
            await dataset.investigate(scenes, run, source_job, explorer)
        assert len(observed["roles"]) == rounds
        assert len(observed["roles"][-1]["frames"]) == 5 * (rounds - 1)
        assert not observed["reads"]
        assert not list(scenes.folder(run.id).glob(".source-*.mp4"))

    asyncio.run(exercise())


def test_agent_can_inspect_every_video_in_the_complete_thirty_episode_dataset(
    tmp_path, monkeypatch
):
    selections = [
        decision(*[clip(episode) for episode in range(start, min(start + 4, 30))])
        for start in range(0, 30, 4)
    ]
    selections.append(decision(ready=True))
    scenes, run, source_job, explorer, observed, _, _ = prepared(
        tmp_path,
        monkeypatch,
        selections,
    )

    async def exercise():
        frames, evidence = await dataset.investigate(scenes, run, source_job, explorer)
        assert len(observed["roles"]) == dataset.investigation_rounds(30) == 9
        assert len(observed["downloads"]) == 30
        assert len(set(observed["downloads"])) == 30
        assert len(observed["roles"][-1]["frames"]) == len(frames) == 150
        assert run.source["inspected_episodes"] == list(range(30))
        assert len(evidence["inspected_clips"]) == 30
        assert evidence["replay"]["episode_index"] == 17
        assert run.gates["source_evidence"] is True
        assert not list(scenes.folder(run.id).glob(".source-*.mp4"))

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "videos,rounds", [(1, 3), (4, 3), (5, 3), (8, 3), (9, 4), (30, 9), (96, 25)]
)
def test_round_budget_can_cover_all_videos_and_a_final_ready_decision(videos, rounds):
    assert dataset.investigation_rounds(videos) == rounds
    assert (rounds - 1) * 4 >= videos


def test_catalog_includes_every_advertised_camera_without_eager_video_downloads(
    tmp_path, monkeypatch
):
    scenes, run, source_job, explorer, observed, _, _ = prepared(tmp_path, monkeypatch)
    side = "observation.images.side"
    source_job.result.features[side] = {"dtype": "video", "shape": [1080, 1920, 3]}
    original = explorer.cameras

    async def cameras(*args, **kwargs):
        front = (await original(*args, **kwargs))[0]
        return [
            front,
            front.model_copy(update={"key": side, "url": front.url.replace("/videos/", "/side/")}),
        ]

    explorer.cameras = cameras

    async def exercise():
        catalog, _, _ = await dataset.catalog(scenes, run, source_job, explorer)
        assert len(catalog["episodes"]) == 30
        assert all(len(episode["cameras"]) == 2 for episode in catalog["episodes"])
        total = sum(len(episode["cameras"]) for episode in catalog["episodes"])
        assert dataset.investigation_rounds(total) == 16
        assert not observed["roles"] and not observed["downloads"]

    asyncio.run(exercise())


def test_excessive_dataset_video_scope_rejects_before_provider_work(tmp_path, monkeypatch):
    scenes, run, source_job, explorer, observed, _, _ = prepared(tmp_path, monkeypatch)
    sides = ["observation.images." + name for name in ("left", "side", "top")]
    for key in sides:
        source_job.result.features[key] = {"dtype": "video", "shape": [1080, 1920, 3]}
    original = explorer.cameras

    async def cameras(*args, **kwargs):
        front = (await original(*args, **kwargs))[0]
        return [front] + [
            front.model_copy(
                update={"key": key, "url": front.url.replace("/videos/", f"/{index}/")}
            )
            for index, key in enumerate(sides)
        ]

    explorer.cameras = cameras

    async def exercise():
        with pytest.raises(SceneError, match="96|limit|video"):
            await dataset.investigate(scenes, run, source_job, explorer)
        assert not observed["roles"] and not observed["downloads"]

    asyncio.run(exercise())


@pytest.mark.parametrize("invalid", ["truncated", "noncontiguous", "nonfinite", "wrong_dimensions"])
def test_replay_requires_complete_contiguous_finite_joint_states(tmp_path, monkeypatch, invalid):
    scenes, run, source_job, explorer, observed, trajectory, _ = prepared(tmp_path, monkeypatch)
    if invalid == "truncated":
        trajectory.pop()
    elif invalid == "noncontiguous":
        trajectory[70]["frame_index"] = 69
    elif invalid == "nonfinite":
        trajectory[70]["state"] = [float("nan")] * 6
    else:
        trajectory[70]["state"] = [0] * 5

    async def exercise():
        with pytest.raises(SceneError):
            await dataset.investigate(scenes, run, source_job, explorer)
        assert run.gates["source_evidence"] is False
        assert not (scenes.folder(run.id) / "trajectory.json").exists()
        assert not list(scenes.folder(run.id).glob(".source-*.mp4"))

    asyncio.run(exercise())


def test_selected_source_video_is_removed_after_frame_extraction_failure(tmp_path, monkeypatch):
    scenes, run, source_job, explorer, observed, _, _ = prepared(tmp_path, monkeypatch)

    async def failed(*args, **kwargs):
        raise SceneError("Fixture decoding stopped")

    monkeypatch.setattr(dataset, "command", failed)

    async def exercise():
        with pytest.raises(SceneError, match="decoding stopped"):
            await dataset.investigate(scenes, run, source_job, explorer)
        assert len(observed["downloads"]) == 1
        assert not list(scenes.folder(run.id).glob(".source-*.mp4"))

    asyncio.run(exercise())
