"""Cloud scene ownership and media integrity checks; no provider calls or GPU."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vla_platform.scenes import runtime
from vla_platform.scenes.agent import SceneError

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "scene_runner_test_module", ROOT / "workers/skypilot/scene_runner.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def config(tmp_path, monkeypatch):
    executable = tmp_path / "gcloud"
    executable.write_text("#!/bin/sh\nexit 1\n")
    executable.chmod(0o700)
    sky_config = tmp_path / "sky.yaml"
    sky_config.write_text("allowed_clouds: [gcp]\n")
    value = {
        "sky_python": sys.executable,
        "gcloud_binary": str(executable),
        "sky_config": str(sky_config),
        "project_id": "example-project",
        "image": "nvcr.io/nvidia/isaac-sim@sha256:" + "a" * 64,
        "machine_image": "projects/example-project/global/images/reviewed-runtime",
        "region": "us-central1",
        "results_uri": "gs://example-results/scene-jobs/",
        "accept_eula": True,
    }
    file = tmp_path / "runtime.json"
    file.write_text(json.dumps(value))
    monkeypatch.setenv("JENSEN_SCENE_SIMULATION_CONFIG", str(file))
    return value, file


def test_configuration_is_pinned_and_never_passes_provider_keys(config, monkeypatch):
    value, file = config
    assert runtime.available()
    selected = runtime.configuration()
    assert selected["results_uri"].endswith("/scene-jobs")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-secret")
    monkeypatch.setenv("CODEX_API_KEY", "synthetic-test-secret")
    environment = runtime._environment(selected)
    assert "OPENAI_API_KEY" not in environment and "CODEX_API_KEY" not in environment
    assert environment["SKYPILOT_GLOBAL_CONFIG"] == selected["sky_config"]
    assert environment["CLOUDSDK_CORE_PROJECT"] == selected["project_id"]
    file.write_text(json.dumps({**value, "sky_api_endpoint": "http://127.0.0.1:46582"}))
    assert runtime._environment(runtime.configuration())["SKYPILOT_API_SERVER_ENDPOINT"] == (
        "http://127.0.0.1:46582"
    )
    for changed in [
        {"image": "nvcr.io/nvidia/isaac-sim:latest"},
        {"accept_eula": False},
        {"results_uri": "gs://example-results/scene-jobs/../other"},
        {"sky_python": "/missing/trusted/python"},
        {"region": "us-central1;unexpected"},
        {"sky_api_endpoint": "http://127.0.0.1:99999"},
        {"sky_api_endpoint": "http://example.com:46582"},
        {"unexpected_command": "arbitrary"},
    ]:
        file.write_text(json.dumps({**value, **changed}))
        assert not runtime.available()
        with pytest.raises(SceneError):
            runtime.configuration()


@pytest.fixture
def owned(tmp_path, config):
    value, _ = config
    path = tmp_path / "ownership.json"
    ident = "a" * 32
    record = {
        "run_id": ident,
        "cluster": "scene-" + ident[:16],
        "config": value,
        "inputs": str(tmp_path / "inputs"),
        "output": str(tmp_path / "outputs"),
        "results_prefix": "gs://example-results/scene-jobs/" + ident,
        "receipt": str(path),
        "scene_sha256": "b" * 64,
        "trajectory_sha256": "c" * 64,
    }
    runner.write(
        path,
        {
            "run_id": ident,
            "cluster": record["cluster"],
            "launch_attempted": False,
            "state": "prepared",
        },
    )
    return record


def test_launch_persists_ownership_before_the_cloud_request(owned):
    sky = MagicMock()
    sky.status.return_value = "status-request"
    sky.get.side_effect = [[], (7, None)]

    def launch(*args, **kwargs):
        receipt = runner.receipt(owned)
        assert receipt["launch_attempted"] is True
        assert receipt["state"] == "submitting"
        return "launch-request"

    sky.launch.side_effect = launch
    runner.launch(owned, sky)
    saved = runner.receipt(owned)
    assert saved["job_id"] == 7 and saved["launch_request_id"] == "launch-request"
    assert sky.launch.call_args.kwargs["down"] is True
    assert sky.launch.call_args.kwargs["idle_minutes_to_autostop"] == 5
    assert sky.Resources.call_args.kwargs["accelerators"] == "L4:1"
    assert sky.Resources.call_args.kwargs["memory"] == "48+"
    script = sky.Task.call_args.kwargs["run"]
    assert "--network none" in script and "--env ACCEPT_EULA=Y" in script
    assert "OPENAI_API_KEY" not in script and "application_default_credentials" not in script
    with pytest.raises(ValueError, match="must not be repeated"):
        runner.launch(owned, sky)


def test_existing_cluster_is_never_adopted_or_deleted(owned):
    sky = MagicMock()
    sky.status.return_value = "status-request"
    sky.get.return_value = [{"name": owned["cluster"]}]
    with pytest.raises(ValueError, match="already in use"):
        runner.launch(owned, sky)
    assert runner.receipt(owned)["launch_attempted"] is False
    sky.launch.assert_not_called()
    runner.down(owned, sky)
    sky.down.assert_not_called()


def test_observation_requires_the_exact_submitted_job(owned):
    saved = runner.receipt(owned)
    saved.update(job_id=7, launch_attempted=True)
    runner.write(owned["receipt"], saved)
    sky = MagicMock()
    sky.queue.return_value = "queue-request"
    sky.get.return_value = [{"job_id": 7, "job_name": "different", "status": "SUCCEEDED"}]
    with pytest.raises(ValueError, match="identity"):
        runner.observe(owned, sky)
    sky.get.return_value[0]["job_name"] = owned["cluster"]
    runner.observe(owned, sky)
    assert runner.receipt(owned)["state"] == "SUCCEEDED"


def _outputs(folder, request):
    folder.mkdir(exist_ok=True)
    media = []
    for name in runtime.LIMITS:
        if name.endswith(".png"):
            (folder / name).write_bytes(b"\x89PNG\r\n\x1a\n" + b"synthetic PNG fixture")
        elif name.endswith(".mp4"):
            (folder / name).write_bytes(b"\x00\x00\x00\x18ftyp" + b"synthetic MP4 fixture")
        elif name.endswith(".jsonl"):
            (folder / name).write_text('{"source_frame":0}\n')
        else:
            continue
        if name.endswith((".png", ".mp4")):
            path = folder / name
            media.append(
                {
                    "role": "episode_replay" if name.endswith(".mp4") else "environment_image",
                    "path": name,
                    "label": name,
                    "bytes": path.stat().st_size,
                    "sha256": runtime._sha(path),
                }
            )
    source = json.loads(Path(request["trajectory"]).read_text())
    result = {
        "status": "succeeded",
        "isaac_runtime": True,
        "scene_sha256": request["scene_sha256"],
        "trajectory_sha256": request["trajectory_sha256"],
        "source_frames": len(source["frames"]),
        "rendered_frames": len(source["frames"]),
        "episode_index": source["episode_index"],
        "fps": source["fps"],
        "mapping_status": "uncalibrated_dataset_normalized",
        "calibration_verified": False,
        "calibrated_replay": False,
        "task_success": None,
        "media": media,
        "trace": {
            "path": "replay-trace.jsonl",
            "sha256": runtime._sha(folder / "replay-trace.jsonl"),
        },
    }
    (folder / "result.json").write_text(json.dumps(result))
    return result


def test_verification_rejects_changed_media_and_false_calibration(tmp_path):
    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text(json.dumps({"episode_index": 3, "fps": 30, "frames": [{}, {}]}))
    request = {
        "trajectory": str(trajectory),
        "scene_sha256": "b" * 64,
        "trajectory_sha256": runtime._sha(trajectory),
    }
    outputs = tmp_path / "outputs"
    result = _outputs(outputs, request)
    verified = runtime.verify(outputs, request)
    assert all(item["path"].startswith("simulation/") for item in verified["media"])
    (outputs / "replay.mp4").write_bytes(b"changed after collection")
    with pytest.raises(SceneError, match="hash or size"):
        runtime.verify(outputs, request)
    _outputs(outputs, request)
    result["calibration_verified"] = True
    (outputs / "result.json").write_text(json.dumps(result))
    with pytest.raises(SceneError, match="differ"):
        runtime.verify(outputs, request)


@pytest.mark.anyio
async def test_runtime_uses_one_submission_and_owned_teardown(tmp_path, config, monkeypatch):
    folder = tmp_path / "scene"
    folder.mkdir()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "scene.usda").write_text("synthetic scene fixture")
    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text(json.dumps({"episode_index": 3, "fps": 30, "frames": [{}, {}]}))
    events, calls = [], []

    async def progress(*args):
        events.append(args[1:])

    scenes = SimpleNamespace(root=ROOT, folder=lambda _: folder, progress=progress)
    scene = SimpleNamespace(id="d" * 32)

    async def command(args, **kwargs):
        operation = args[2]
        calls.append(operation)
        request = runner.request(args[-1])
        saved = runner.receipt(request)
        if operation == "launch":
            saved.update(launch_attempted=True, job_id=7, state="submitted")
        elif operation == "observe":
            saved["state"] = "SUCCEEDED"
        elif operation == "collect":
            _outputs(Path(request["output"]), request)
        elif operation == "down":
            saved["resource_cleanup"] = "cluster_removed"
        runner.write(request["receipt"], saved)

    monkeypatch.setattr(runtime, "command", command)
    result = await runtime.run(scenes, scene, bundle, trajectory)
    assert calls == ["launch", "observe", "collect", "down"]
    assert result["resource_cleanup"] == "cluster_removed"
    assert not (folder / "simulation-runtime/inputs").exists()
    assert (folder / "simulation-runtime/ownership.json").is_file()
    assert (folder / "simulation/replay.mp4").is_file()
    assert not result["calibration_verified"]


@pytest.mark.anyio
async def test_cancellation_still_cleans_up_owned_resources(tmp_path, config, monkeypatch):
    import asyncio

    folder = tmp_path / "scene"
    folder.mkdir()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "scene.usda").write_text("synthetic scene fixture")
    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text(json.dumps({"episode_index": 3, "fps": 30, "frames": [{}, {}]}))

    async def progress(*args):
        pass

    scenes = SimpleNamespace(root=ROOT, folder=lambda _: folder, progress=progress)
    calls = []

    async def command(args, **kwargs):
        calls.append(args[2])
        if args[2] == "launch":
            raise asyncio.CancelledError
        request = runner.request(args[-1])
        saved = runner.receipt(request)
        saved["resource_cleanup"] = "cluster_removed"
        runner.write(request["receipt"], saved)

    monkeypatch.setattr(runtime, "command", command)
    with pytest.raises(asyncio.CancelledError):
        await runtime.run(scenes, SimpleNamespace(id="e" * 32), bundle, trajectory)
    assert calls == ["launch", "down"]
    assert (
        json.loads((folder / "simulation-runtime/cleanup.json").read_text())["resource_cleanup"]
        == "cluster_removed"
    )
