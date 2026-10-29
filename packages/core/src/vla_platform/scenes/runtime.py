"""Owned SkyPilot scene rendering; cloud and simulator libraries stay isolated."""

import asyncio
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

from vla_platform.scenes.agent import SceneError, command

LIMITS = {
    "result.json": 1024**2,
    "replay.mp4": 400 * 1024**2,
    "replay-trace.jsonl": 32 * 1024**2,
    "environment-front.png": 10 * 1024**2,
    "environment-overview.png": 10 * 1024**2,
    "environment-side.png": 10 * 1024**2,
}
TERMINAL = {"SUCCEEDED", "FAILED", "FAILED_SETUP", "FAILED_DRIVER", "CANCELLED"}
CONFIG_FIELDS = {
    "sky_python",
    "sky_config",
    "gcloud_binary",
    "project_id",
    "image",
    "machine_image",
    "region",
    "results_uri",
    "accept_eula",
}


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024**2), b""):
            result.update(part)
    return result.hexdigest()


def _json(path, limit=1024**2):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise SceneError("Scene simulation record is missing or exceeds its bound.")
    return json.loads(path.read_bytes())


def configuration():
    location = os.environ.get("JENSEN_SCENE_SIMULATION_CONFIG")
    if not location:
        raise SceneError("Scene simulation is not configured.")
    value = _json(location, 64 * 1024)
    if (
        not isinstance(value, dict)
        or not CONFIG_FIELDS <= set(value)
        or set(value) - CONFIG_FIELDS - {"sky_api_endpoint"}
    ):
        raise SceneError("Scene simulation configuration has unsupported fields.")
    if value["accept_eula"] is not True:
        raise SceneError("The operator must accept the NVIDIA container license.")
    patterns = {
        "project_id": r"[a-z][a-z0-9-]{4,61}[a-z0-9]",
        "image": r"[A-Za-z0-9][A-Za-z0-9._:/-]+@sha256:[a-f0-9]{64}",
        "machine_image": r"projects/[a-z0-9-]+/global/images/[A-Za-z0-9._-]+",
        "region": r"[a-z]+-[a-z]+[0-9]",
        "results_uri": r"gs://[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]/[A-Za-z0-9/_-]+/?",
    }
    for name, pattern in patterns.items():
        if not isinstance(value[name], str) or not re.fullmatch(pattern, value[name]):
            raise SceneError("Scene simulation configuration has an invalid " + name + ".")
    if any(part in {"", ".", ".."} for part in value["results_uri"][5:].rstrip("/").split("/")):
        raise SceneError("Scene simulation result location is invalid.")
    for name in ("sky_python", "gcloud_binary", "sky_config"):
        path = Path(value[name])
        if not path.is_absolute() or not path.is_file():
            raise SceneError("Scene simulation runtime dependency is missing.")
        if name != "sky_config" and not os.access(path, os.X_OK):
            raise SceneError("Scene simulation executable is unavailable.")
    endpoint = value.get("sky_api_endpoint")
    if endpoint is not None:
        if (
            not isinstance(endpoint, str)
            or not re.fullmatch(r"http://127\.0\.0\.1:[1-9][0-9]{3,4}", endpoint)
            or not 1024 <= int(endpoint.rsplit(":", 1)[1]) <= 65535
        ):
            raise SceneError("Scene SkyPilot API endpoint must use a local nonprivileged port.")
    value["results_uri"] = value["results_uri"].rstrip("/")
    return value


def available():
    try:
        configuration()
        return True
    except SceneError, OSError, ValueError, TypeError:
        return False


def _snapshot(source, destination, *, python_only=False):
    total, count = 0, 0
    if source.is_symlink() or not source.is_dir():
        raise SceneError("Scene worker input is missing or linked.")
    destination.mkdir(parents=True, exist_ok=False)
    for current, dirs, files in os.walk(source, followlinks=False):
        dirs[:] = sorted(
            name for name in dirs if not name.startswith(".") and name != "__pycache__"
        )
        if any((Path(current) / name).is_symlink() for name in dirs):
            raise SceneError("Scene worker inputs contain a linked directory.")
        for name in sorted(files):
            if name.startswith(".") or (python_only and not name.endswith(".py")):
                continue
            origin = Path(current) / name
            if origin.is_symlink() or not origin.is_file():
                raise SceneError("Scene worker inputs contain a nonregular file.")
            total += origin.stat().st_size
            count += 1
            if total > 256 * 1024**2 or count > 2048:
                raise SceneError("Scene worker inputs exceed the snapshot bound.")
            target = destination / origin.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, target)


def _environment(config):
    env = {
        key: value
        for key, value in os.environ.items()
        if key in {"HOME", "PATH", "LANG", "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR"}
    }
    env.update(
        SKYPILOT_GLOBAL_CONFIG=config["sky_config"],
        SKYPILOT_PROJECT_CONFIG=config["sky_config"],
        CLOUDSDK_CORE_PROJECT=config["project_id"],
        CLOUDSDK_PYTHON=config["sky_python"],
        PYTHONDONTWRITEBYTECODE="1",
    )
    if config.get("sky_api_endpoint"):
        env["SKYPILOT_API_SERVER_ENDPOINT"] = config["sky_api_endpoint"]
    env["PATH"] = str(Path(config["gcloud_binary"]).parent) + os.pathsep + env.get("PATH", "")
    return env


def verify(folder, request):
    result = _json(folder / "result.json")
    source = _json(request["trajectory"], 8 * 1024**2)
    if (
        result.get("status") != "succeeded"
        or result.get("isaac_runtime") is not True
        or result.get("scene_sha256") != request["scene_sha256"]
        or result.get("trajectory_sha256") != request["trajectory_sha256"]
        or result.get("source_frames") != len(source["frames"])
        or result.get("rendered_frames") != len(source["frames"])
        or result.get("episode_index") != source["episode_index"]
        or result.get("fps") != source["fps"]
        or result.get("mapping_status") != "uncalibrated_dataset_normalized"
        or result.get("calibration_verified") is not False
        or result.get("calibrated_replay") is not False
        or result.get("task_success") is not None
    ):
        raise SceneError("Simulation results differ from the submitted scene or episode.")
    media = result.get("media")
    expected = {
        "environment-front.png": "environment_image",
        "environment-overview.png": "environment_image",
        "environment-side.png": "environment_image",
        "replay.mp4": "episode_replay",
    }
    if not isinstance(media, list) or len(media) != len(expected):
        raise SceneError("Simulation is missing its environment images or replay video.")
    found = set()
    for item in media:
        if (
            not isinstance(item, dict)
            or item.get("path") not in expected
            or item["path"] in found
            or item.get("role") != expected[item["path"]]
        ):
            raise SceneError("Simulation media has an unexpected identity.")
        name = item["path"]
        found.add(name)
        local = folder / name
        if (
            local.is_symlink()
            or not local.is_file()
            or not 0 < local.stat().st_size <= LIMITS[name]
            or type(item.get("bytes")) is not int
            or item["bytes"] != local.stat().st_size
            or item.get("sha256") != _sha(local)
        ):
            raise SceneError("Simulation media hash or size could not be verified.")
        if name.endswith(".png"):
            with local.open("rb") as stream:
                header = stream.read(24)
            if header[:8] != b"\x89PNG\r\n\x1a\n":
                raise SceneError("Simulation environment image is not PNG.")
        else:
            with local.open("rb") as stream:
                if stream.read(12)[4:8] != b"ftyp":
                    raise SceneError("Simulation replay is not MP4.")
        item["path"] = "simulation/" + name
    trace = result.get("trace")
    if (
        not isinstance(trace, dict)
        or trace.get("path") != "replay-trace.jsonl"
        or not (folder / trace["path"]).is_file()
        or (folder / trace["path"]).is_symlink()
        or not 0 < (folder / trace["path"]).stat().st_size <= LIMITS[trace["path"]]
        or trace.get("sha256") != _sha(folder / trace["path"])
    ):
        raise SceneError("Simulation is missing its recorded replay trace.")
    return result


async def _finish(task):
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


async def preflight(scenes, scene):
    """Check SDK/configuration and read-only account access before provider work."""
    config = configuration()
    folder = scenes.folder(scene.id)
    request_path = folder / ".simulation-preflight.json"
    output = folder / ".simulation-preflight-result.json"
    record = {
        "run_id": scene.id,
        "cluster": "scene-" + scene.id[:16],
        "config": config,
        "inputs": str(folder),
        "results_prefix": config["results_uri"] + "/" + scene.id,
        "preflight_output": str(output),
    }
    from vla_platform.scenes.service import write

    write(request_path, record)
    try:
        await command(
            [
                config["sky_python"],
                scenes.root / "workers/skypilot/scene_runner.py",
                "preflight",
                "--request",
                request_path,
            ],
            cwd=folder,
            env=_environment(config),
            timeout=60,
            limit=128 * 1024,
        )
        result = _json(output, 4096)
        if result.get("passed") is not True or result.get("skypilot") != "0.13.0":
            raise SceneError("Scene compute preflight did not produce a valid result.")
    except SceneError, OSError, ValueError:
        raise SceneError(
            "Scene compute preflight failed. Check the SkyPilot runtime, account "
            "and private API endpoint before starting provider work."
        ) from None


async def run(scenes, scene, bundle, trajectory):
    config = configuration()
    folder = scenes.folder(scene.id)
    runtime = folder / "simulation-runtime"
    runtime.mkdir(mode=0o700, exist_ok=False)
    inputs = runtime / "inputs"
    inputs.mkdir()
    _snapshot(Path(bundle), inputs / "bundle")
    worker = scenes.root / "workers/isaac_sim"
    (inputs / "worker").mkdir()
    _snapshot(worker / "sim_worker", inputs / "worker/sim_worker", python_only=True)
    shutil.copyfile(worker / "scene_job.py", inputs / "worker/scene_job.py")
    shutil.copyfile(trajectory, inputs / "trajectory.json")
    request = {
        "run_id": scene.id,
        "cluster": "scene-" + scene.id[:16],
        "config": config,
        "inputs": str(inputs),
        "output": str(folder / "simulation"),
        "trajectory": str(inputs / "trajectory.json"),
        "receipt": str(runtime / "ownership.json"),
        "scene_sha256": _sha(inputs / "bundle/scene.usda"),
        "trajectory_sha256": _sha(inputs / "trajectory.json"),
        "results_prefix": config["results_uri"] + "/" + scene.id,
    }
    # This receipt exists before the first possible cloud side effect.
    from vla_platform.scenes.service import write

    write(runtime / "request.json", request)
    write(
        runtime / "ownership.json",
        {
            "run_id": scene.id,
            "cluster": request["cluster"],
            "launch_attempted": False,
            "state": "prepared",
            "resource_cleanup": "not_launched",
        },
    )
    helper = scenes.root / "workers/skypilot/scene_runner.py"

    async def invoke(operation, timeout=90):
        await command(
            [config["sky_python"], helper, operation, "--request", runtime / "request.json"],
            cwd=runtime,
            env=_environment(config),
            timeout=timeout,
            limit=1024**2,
        )

    completed, result = False, None
    try:
        async with asyncio.timeout(3600):
            await scenes.progress(scene, "simulation", "Preparing the scene simulation.")
            await invoke("launch", 900)
            previous = None
            while True:
                await invoke("observe")
                owned = _json(runtime / "ownership.json")
                state = owned.get("state")
                if state != previous:
                    await scenes.progress(
                        scene, "simulation", "Simulation " + str(state).lower() + "."
                    )
                    previous = state
                if state in TERMINAL:
                    if state != "SUCCEEDED":
                        raise SceneError(
                            "Scene simulation did not complete; its evidence was retained."
                        )
                    break
                await asyncio.sleep(5)
            await scenes.progress(
                scene, "simulation", "Saving environment images and episode replay."
            )
            await invoke("collect", 600)
            result = verify(folder / "simulation", request)
            completed = True
    except TimeoutError:
        raise SceneError(
            "Scene simulation exceeded its time budget; no replacement was launched."
        ) from None
    finally:

        async def cleanup():
            try:
                await invoke("down", 180)
                return _json(runtime / "ownership.json").get("resource_cleanup", "unverified")
            except SceneError, OSError, ValueError:
                await scenes.progress(
                    scene,
                    "simulation",
                    (
                        "Compute cleanup needs operator attention. "
                        "Automatic teardown remains scheduled."
                    ),
                )
                return "unverified"

        cleanup_status = await _finish(asyncio.create_task(cleanup()))
        write(runtime / "cleanup.json", {"resource_cleanup": cleanup_status})
        if result is not None:
            result["resource_cleanup"] = cleanup_status
        if completed and cleanup_status in {"cluster_removed", "not_launched"}:
            # Completed inputs are disposable; ownership and cloud artifacts remain.
            shutil.rmtree(inputs)
    return result


async def recover(scenes, scene):
    """On restart, stop only recorded cloud ownership without repeating the job."""
    runtime = scenes.folder(scene.id) / "simulation-runtime"
    request_path = runtime / "request.json"
    if not request_path.is_file():
        return "not_launched"
    request = _json(request_path)
    if request.get("run_id") != scene.id or request.get("cluster") != "scene-" + scene.id[:16]:
        raise SceneError("Interrupted scene ownership could not be verified.")
    owned = _json(runtime / "ownership.json")
    if owned.get("run_id") != scene.id or owned.get("cluster") != request["cluster"]:
        raise SceneError("Interrupted scene ownership receipt differs from its request.")
    config = request["config"]
    try:
        await command(
            [
                config["sky_python"],
                scenes.root / "workers/skypilot/scene_runner.py",
                "down",
                "--request",
                request_path,
            ],
            cwd=runtime,
            env=_environment(config),
            timeout=180,
            limit=1024**2,
        )
        result = _json(runtime / "ownership.json").get("resource_cleanup", "unverified")
    except SceneError, OSError, ValueError:
        result = "unverified"
    from vla_platform.scenes.service import write

    write(runtime / "cleanup.json", {"resource_cleanup": result})
    return result
