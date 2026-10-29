"""Fixed SkyPilot/GCS adapter for owned scene jobs; never imports application secrets."""

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import tempfile
from importlib.metadata import version
from pathlib import Path

SDK_VERSION = "0.13.0"
MAX_JSON = 1024**2
FILES = {
    "result.json": MAX_JSON,
    "replay.mp4": 400 * 1024**2,
    "replay-trace.jsonl": 32 * 1024**2,
    "environment-front.png": 10 * 1024**2,
    "environment-overview.png": 10 * 1024**2,
    "environment-side.png": 10 * 1024**2,
}
TOTAL_BYTES = 512 * 1024**2
ACTIVE = {"INIT", "PENDING", "SETTING_UP", "RUNNING"}
TERMINAL = {"SUCCEEDED", "FAILED", "FAILED_SETUP", "FAILED_DRIVER", "CANCELLED"}


def read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON:
        raise ValueError("Scene runtime record is missing or exceeds its bound")
    return json.loads(path.read_bytes())


def write(path, value):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Linked scene runtime record")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".scene-sky-", delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write((json.dumps(value, allow_nan=False) + "\n").encode())
            file.flush()
            file.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def request(path):
    record = read(path)
    ident = record.get("run_id")
    if (
        not isinstance(ident, str)
        or not re.fullmatch(r"[a-f0-9]{32}", ident)
        or record.get("cluster") != "scene-" + ident[:16]
        or not isinstance(record.get("config"), dict)
    ):
        raise ValueError("Invalid scene runtime ownership request")
    return record


def receipt(record):
    value = read(record["receipt"])
    if value.get("run_id") != record["run_id"] or value.get("cluster") != record["cluster"]:
        raise ValueError("Scene runtime receipt belongs to different work")
    return value


def _request_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
        raise ValueError("SkyPilot request identity is unavailable")
    return value


def _sdk():
    if version("skypilot") != SDK_VERSION:
        raise ValueError("Scene runtime requires the reviewed SkyPilot version")
    import sky

    return sky


def _get(sky, pending):
    return sky.get(_request_id(pending))


def _cloud(record, args, *, timeout=60):
    config = record["config"]
    return subprocess.run(
        [config["gcloud_binary"], "--quiet", "--project", config["project_id"], *args],
        check=True,
        capture_output=True,
        timeout=timeout,
    ).stdout


def _script(record):
    config = record["config"]
    prefix = record["results_prefix"]
    quote = shlex.quote
    # Credentials stay on the VM host; only immutable inputs reach the container.
    body = "\n".join(
        [
            "set -euo pipefail",
            "mkdir -p job-output",
            "chmod 0777 job-output",
            'scene_inputs="$PWD"',
            "scene_container=" + quote(record["cluster"]),
            "upload_scene_results() {",
            '  timeout --kill-after=5s 20s docker rm -f "$scene_container" >/dev/null 2>&1 || true',
            "  for scene_file in " + " ".join(quote(name) for name in FILES) + "; do",
            '    scene_path="job-output/$scene_file"',
            '    if [ -f "$scene_path" ] && [ ! -L "$scene_path" ]; then',
            '      sudo chmod a+r -- "$scene_path" || continue',
            '      timeout --kill-after=10s 90s gcloud storage cp "$scene_path" '
            + quote(prefix + "/")
            + " || true",
            "    fi",
            "  done",
            "}",
            "trap upload_scene_results EXIT",
            "trap 'exit 143' TERM",
            "trap 'exit 130' INT",
            "timeout --kill-after=30s 600s docker pull " + quote(config["image"]),
            'timeout --kill-after=30s 3000s docker run --rm --name "$scene_container" '
            "--gpus all --cpus 8 --memory 42g --pids-limit 1024 "
            "--ipc host --network none --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y "
            "--env PYTHONUNBUFFERED=1 --env NVIDIA_DRIVER_CAPABILITIES=all "
            '--mount "type=bind,src=$scene_inputs,dst=/inputs,readonly" '
            '--mount "type=bind,src=$scene_inputs/job-output,dst=/outputs" '
            "--entrypoint /isaac-sim/python.sh "
            + quote(config["image"])
            + " --no-ros-env /inputs/worker/scene_job.py "
            "--bundle /inputs/bundle --trajectory /inputs/trajectory.json --output /outputs",
        ]
    )
    # API loss cannot leave the remote job running indefinitely. Its signal
    # handler stops only the named job container before attempting publication.
    return "timeout --signal=TERM --kill-after=45s 3300s bash -c " + quote(body)


def _setup(config):
    registry = config["image"].split("/", 1)[0]
    commands = [
        "set -euo pipefail",
        "sudo apt-get update -qq",
        "sudo apt-get install -y --no-install-recommends libvulkan1",
    ]
    if registry == "gcr.io" or registry.endswith((".gcr.io", ".pkg.dev")):
        commands.append("gcloud auth configure-docker " + shlex.quote(registry) + " --quiet")
    return "\n".join(commands)


def _task(record, sky):
    config = record["config"]
    task = sky.Task(
        name=record["cluster"],
        setup=_setup(config),
        run=_script(record),
        workdir=record["inputs"],
        api_server_access=False,
    )
    task.set_resources(
        sky.Resources(
            infra="gcp/" + config["region"],
            accelerators="L4:1",
            cpus="8+",
            memory="48+",
            disk_size=150,
            image_id=config["machine_image"],
            use_spot=False,
            labels={"openjensen-scene": record["run_id"]},
        )
    )
    return task


def preflight(record, sky=None):
    sky = sky or _sdk()
    _task(record, sky)
    existing = _get(sky, sky.status(cluster_names=[record["cluster"]]))
    if existing:
        raise ValueError("Scene cluster name is already in use")
    write(record["preflight_output"], {"passed": True, "skypilot": SDK_VERSION})


def launch(record, sky=None):
    sky = sky or _sdk()
    saved = receipt(record)
    if saved.get("launch_attempted"):
        raise ValueError("Scene launch is already owned; it must not be repeated")
    existing = _get(sky, sky.status(cluster_names=[record["cluster"]]))
    if existing:
        saved.update(state="name_collision")
        write(record["receipt"], saved)
        raise ValueError("Scene cluster name is already in use")
    task = _task(record, sky)
    saved.update(launch_attempted=True, state="submitting")
    write(record["receipt"], saved)
    pending = _request_id(
        sky.launch(
            task,
            cluster_name=record["cluster"],
            retry_until_up=False,
            idle_minutes_to_autostop=5,
            down=True,
        )
    )
    saved.update(launch_request_id=pending)
    write(record["receipt"], saved)
    launched = sky.get(pending)
    if (
        not isinstance(launched, tuple)
        or len(launched) != 2
        or type(launched[0]) is not int
        or launched[0] <= 0
    ):
        raise ValueError("Scene launch did not produce an owned job ID")
    saved.update(job_id=launched[0], state="submitted")
    write(record["receipt"], saved)


def observe(record, sky=None):
    sky = sky or _sdk()
    saved = receipt(record)
    job_id = saved.get("job_id")
    if type(job_id) is not int or job_id <= 0:
        raise ValueError("Scene status cannot be guessed without its submitted job ID")
    rows = _get(sky, sky.queue(record["cluster"], skip_finished=False, all_users=False))
    found = [row for row in rows if row.get("job_id") == job_id]
    if len(found) != 1 or found[0].get("job_name") != record["cluster"]:
        raise ValueError("Scene queue job identity differs from its ownership receipt")
    status = getattr(found[0].get("status"), "value", found[0].get("status"))
    if status not in ACTIVE | TERMINAL:
        raise ValueError("Unknown scene queue status")
    saved.update(state=status)
    write(record["receipt"], saved)


def down(record, sky=None):
    sky = sky or _sdk()
    saved = receipt(record)
    if not saved.get("launch_attempted"):
        saved.update(resource_cleanup="not_launched")
        write(record["receipt"], saved)
        return
    pending = _request_id(sky.down(record["cluster"]))
    saved.update(down_request_id=pending, resource_cleanup="requested")
    write(record["receipt"], saved)
    sky.get(pending)
    # Refresh the exact cluster, without touching the caller's other workloads.
    from sky.utils.common import StatusRefreshMode

    remaining = _get(
        sky,
        sky.status(
            cluster_names=[record["cluster"]],
            refresh=StatusRefreshMode.ALL,
        ),
    )
    if remaining:
        raise ValueError("Scene cluster teardown is not yet confirmed")
    saved.update(resource_cleanup="cluster_removed")
    write(record["receipt"], saved)


def collect(record):
    saved = receipt(record)
    if saved.get("state") != "SUCCEEDED":
        raise ValueError("Only the exact successful scene job may publish simulation media")
    output = Path(record["output"])
    output.mkdir(parents=True, exist_ok=True)
    total, inventory = 0, {}
    for name, maximum in FILES.items():
        uri = record["results_prefix"] + "/" + name
        metadata = _cloud(record, ["storage", "objects", "describe", uri, "--format=json"])
        if len(metadata) > MAX_JSON:
            raise ValueError("Scene object metadata exceeded its limit")
        metadata = json.loads(metadata)
        size, generation = str(metadata.get("size", "")), str(metadata.get("generation", ""))
        if not size.isdigit() or not generation.isdigit() or not 0 < int(size) <= maximum:
            raise ValueError("Scene output is missing a bounded generation")
        total += int(size)
        if total > TOTAL_BYTES:
            raise ValueError("Scene output exceeded its total download limit")
        local = output / name
        if local.exists() or local.is_symlink():
            raise ValueError("Existing scene simulation evidence must not be overwritten")
        _cloud(record, ["storage", "cp", uri + "#" + generation, str(local)], timeout=180)
        if local.stat().st_size != int(size):
            raise ValueError("Scene artifact size changed while collecting its generation")
        inventory[name] = {"generation": generation, "bytes": int(size), "sha256": digest(local)}
    result = read(output / "result.json")
    if (
        result.get("status") != "succeeded"
        or result.get("scene_sha256") != record["scene_sha256"]
        or result.get("trajectory_sha256") != record["trajectory_sha256"]
    ):
        raise ValueError("Downloaded simulation does not match its scene or recorded episode")
    for media in result.get("media", []):
        if (
            not isinstance(media, dict)
            or media.get("path") not in inventory
            or media.get("sha256") != inventory[media["path"]]["sha256"]
            or media.get("bytes") != inventory[media["path"]]["bytes"]
        ):
            raise ValueError("Scene media hash or size differs from its worker receipt")
    write(Path(record["receipt"]).with_name("collection.json"), inventory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("preflight", "launch", "observe", "collect", "down"))
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    record = request(args.request)
    try:
        globals()[args.operation](record)
    except Exception as error:
        print(
            json.dumps(
                {
                    "error_type": type(error).__name__,
                    "message": "Owned scene runtime operation did not complete.",
                }
            ),
            flush=True,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
