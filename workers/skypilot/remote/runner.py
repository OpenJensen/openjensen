"""Run one container, verify its output, and publish a durable job record."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from urllib.parse import urlsplit
import uuid

import cloud


_SUCCESS = 0
_FAILURE = 1
_INVALID_OUTPUT = os.EX_DATAERR
_TIMEOUT = 124
_SIGNAL_OFFSET = 128
_RUNTIME_UID = 1234
_KILL_GRACE = 30
_CLIENT_GRACE = 60
_POLL_SECONDS = 0.25
_DEFAULT_TIMEOUT = 3600
_MAX_TIMEOUT = 172800
_SHM_SIZE = "2g"
_HASH_CHUNK_BYTES = 1024 * 1024
_HASH_LENGTH_BYTES = 8
_CONTAINER_SOURCE = "/opt/sim-worker"
_CONTAINER_OUTPUT = "/outputs"
_IMAGE_PATTERN = r"[^\s]+@sha256:[0-9a-f]{64}"
_COPY_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".git", ".venv")


class _Cancelled(Exception):
    def __init__(self, signum):
        self.code = _SIGNAL_OFFSET + signum
        super().__init__(f"Job cancelled by signal {signum}")


def _cancel(signum, _frame):
    raise _Cancelled(signum)


def _config(workdir):
    image = os.environ["SIM_IMAGE"]
    if not re.fullmatch(_IMAGE_PATTERN, image):
        raise ValueError("SIM_IMAGE must use an immutable @sha256 digest")
    if os.environ.get("ACCEPT_EULA") != "Y":
        raise ValueError("ACCEPT_EULA=Y is required")
    manifest = Path(os.environ.get("SIM_MANIFEST", "demo.yaml"))
    if manifest.is_absolute() or ".." in manifest.parts:
        raise ValueError("SIM_MANIFEST must be a relative path inside the worker source")
    resolved = (workdir / manifest).resolve()
    if not resolved.is_relative_to(workdir) or not resolved.is_file():
        raise ValueError("SIM_MANIFEST does not identify a file inside the worker source")
    timeout = int(os.environ.get("SIM_JOB_TIMEOUT", _DEFAULT_TIMEOUT))
    if not 0 < timeout <= _MAX_TIMEOUT:
        raise ValueError(f"SIM_JOB_TIMEOUT must be between 1 and {_MAX_TIMEOUT} seconds")
    destination = os.environ["SIM_RESULTS_URI"].rstrip("/")
    uri = urlsplit(destination)
    if uri.scheme != "gs" or not uri.netloc or uri.query or uri.fragment:
        raise ValueError("SIM_RESULTS_URI must be a gs:// bucket prefix")
    return {"image": image, "manifest": manifest.as_posix(), "timeout": timeout,
            "destination": destination}


def _snapshot(workdir, destination):
    # Freeze each job's source before another SkyPilot sync can change it.
    shutil.copytree(workdir, destination, ignore=_COPY_IGNORE)
    destination.chmod(0o755)
    digest = hashlib.sha256()
    for path in sorted(destination.rglob("*")):
        path.chmod(0o755 if path.is_dir() else 0o644)
        if not path.is_file():
            continue
        name = path.relative_to(destination).as_posix().encode()
        digest.update(len(name).to_bytes(_HASH_LENGTH_BYTES, "big"))
        digest.update(name)
        digest.update(path.stat().st_size.to_bytes(_HASH_LENGTH_BYTES, "big"))
        _hash_file(path, digest)
    return digest.hexdigest()


def _hash_file(path, digest):
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _command(config, state, container):
    return [
        "docker", "run", "--name", container, "--init", "--gpus", "all",
        "--shm-size", _SHM_SIZE, "--user", f"{_RUNTIME_UID}:{_RUNTIME_UID}",
        "--env", "ACCEPT_EULA=Y", "--env", "NVIDIA_DRIVER_CAPABILITIES=all",
        "--mount", f"type=bind,src={state / 'source'},dst={_CONTAINER_SOURCE},readonly",
        "--mount", f"type=bind,src={state / 'outputs'},dst={_CONTAINER_OUTPUT}",
        "--workdir", _CONTAINER_SOURCE, "--entrypoint", "/usr/bin/timeout", config["image"],
        "--signal=TERM", f"--kill-after={_KILL_GRACE}", str(config["timeout"]),
        "/isaac-sim/python.sh", "--no-ros-env", "-m", "sim_worker",
        "--manifest", f"{_CONTAINER_SOURCE}/{config['manifest']}",
        "--output-dir", _CONTAINER_OUTPUT,
    ]


def _capture(command, logfile, timeout):
    # The in-container timeout also survives termination of this Docker client.
    with logfile.open("w") as writer:
        process = subprocess.Popen(command, stdout=writer, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + timeout + _CLIENT_GRACE
        try:
            with logfile.open(errors="replace") as reader:
                while process.poll() is None:
                    print(reader.read(), end="", flush=True)
                    if time.monotonic() >= deadline:
                        raise subprocess.TimeoutExpired(command, timeout)
                    time.sleep(_POLL_SECONDS)
                print(reader.read(), end="", flush=True)
            return process.returncode if process.returncode >= _SUCCESS else _SIGNAL_OFFSET - process.returncode
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=_KILL_GRACE)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=_KILL_GRACE)


def _cleanup(container, state):
    # Docker containers belong to the daemon, not the killed client process.
    commands = (["docker", "stop", "--time", str(_KILL_GRACE), container],
                ["docker", "kill", container])
    with (state / "cleanup.log").open("w") as output:
        for command in commands:
            try:
                result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                                        timeout=_CLIENT_GRACE)
                if result.returncode == _SUCCESS:
                    break
            except (OSError, subprocess.TimeoutExpired) as error:
                output.write(str(error) + "\n")
        with (state / "container.log").open("w") as logs:
            try:
                subprocess.run(["docker", "logs", container], stdout=logs, stderr=subprocess.STDOUT,
                               timeout=_CLIENT_GRACE)
                subprocess.run(["docker", "rm", "-f", container], stdout=output,
                               stderr=subprocess.STDOUT, timeout=_CLIENT_GRACE)
            except (OSError, subprocess.TimeoutExpired) as error:
                output.write(str(error) + "\n")


def _validate(outputs):
    files = sorted(outputs.glob("*/result.json"))
    if len(files) != 1:
        raise ValueError(f"Expected one worker result; found {len(files)}")
    record = json.loads(files[0].read_text())
    if not isinstance(record, dict):
        raise ValueError("Worker result must be a JSON object")
    if record.get("status") != "succeeded":
        raise ValueError(f"Worker reported failure: {record.get('error', 'unknown error')}")
    video = files[0].parent / "video.mp4"
    if not video.is_file() or video.stat().st_size == 0:
        raise ValueError("Worker reported success without a nonempty video.mp4")
    digest = _hash_file(video, hashlib.sha256())
    if record.get("sha256") != digest:
        raise ValueError("Video SHA256 does not match worker result")
    video_uri = urlsplit(record.get("video_uri", ""))
    if video_uri.scheme != "gs" or not video_uri.netloc or not video_uri.path.endswith("/video.mp4"):
        raise ValueError("Worker result has no valid video_uri")
    return record


def _finish(state, report, destination):
    code = report["exit_code"]
    try:
        report["worker_result"] = _validate(state / "outputs")
    except (OSError, ValueError, TypeError) as error:
        if code == _SUCCESS:
            code = _INVALID_OUTPUT
        report["validation_error"] = str(error)
    report["exit_code"] = code
    report["status"] = "succeeded" if code == _SUCCESS else "failed"
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    (state / "job-result.json").write_text(json.dumps(report, indent=2) + "\n")
    try:
        cloud._publish(state, destination)
    except Exception as error:
        print(f"Diagnostics upload failed: {error}. Local files: {state}", file=sys.stderr, flush=True)
        return code if code != _SUCCESS else _FAILURE
    if code == _SUCCESS:
        print(f"Video: {report['worker_result']['video_uri']}", flush=True)
    return code


def _run(workdir, config):
    run_id = str(uuid.uuid4())
    state = Path.home() / "sim-runs" / run_id
    state.mkdir(parents=True)
    container = f"isaac-{run_id}"
    destination = f"{config['destination']}/{run_id}"
    report = {"run_id": run_id, "image": config["image"], "manifest": config["manifest"],
              "started_at": datetime.now(timezone.utc).isoformat(), "exit_code": _FAILURE}
    print(f"Job files: {state}\nJob report destination: {destination}", flush=True)
    try:
        report["source_sha256"] = _snapshot(workdir, state / "source")
        subprocess.run(["sudo", "install", "-d", "-m", "0755", "-o", str(_RUNTIME_UID),
                        "-g", str(_RUNTIME_UID), str(state / "outputs")], check=True,
                       timeout=_CLIENT_GRACE)
        cloud._login(config["image"])
        report["exit_code"] = _capture(_command(config, state, container), state / "worker.log",
                                       config["timeout"])
    except subprocess.TimeoutExpired as error:
        report.update(exit_code=_TIMEOUT, error=str(error))
    except _Cancelled as error:
        report.update(exit_code=error.code, error=str(error))
    except Exception as error:
        report["error"] = str(error)
    finally:
        _cleanup(container, state)
    return _finish(state, report, destination)


def _main():
    signal.signal(signal.SIGTERM, _cancel)
    signal.signal(signal.SIGINT, _cancel)
    workdir = Path.cwd().resolve()
    return _run(workdir, _config(workdir))


if __name__ == "__main__":
    raise SystemExit(_main())
