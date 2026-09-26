"""Run one bounded Isaac rollout and preserve artifacts through the cloud driver."""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

_CONTROL = Path(__file__).resolve().parent
_UTC = timezone.utc  # noqa: UP017 -- the GCP host can run Python 3.10.
_CLOUD = _CONTROL / "cloud.py"
_SUCCESS = 0
_FAILURE = 1
_TIMEOUT = 124
_SIGNAL_OFFSET = 128
_RUNTIME_UID = 1234
_KILL_GRACE = 30
_CLIENT_GRACE = 60
_MAX_TIMEOUT = 7200
_PORT = 8080
_SOURCE = "/opt/sim-worker"
_OUTPUT = "/outputs"
_IMAGE_PATTERN = r"[^\s]+@sha256:[0-9a-f]{64}"
_COPY_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".git", ".venv")


class _Mode(Enum):
    ROLLOUT = "rollout"
    EXPERIMENTAL = "experimental"


class _Cancelled(Exception):
    def __init__(self, signum):
        self.code = _SIGNAL_OFFSET + signum
        super().__init__(f"Rollout cancelled by signal {signum}")


def _cancel(signum, _frame):
    raise _Cancelled(signum)


def _driver(*args, **kwargs):
    return subprocess.run([sys.executable, str(_CLOUD), *args], check=True, **kwargs)


def _mode():
    value = os.environ.get("SIM_EXPERIMENTAL", "").strip().lower()
    if value in ("1", "yes"):
        return _Mode.EXPERIMENTAL
    if value in ("", "0", "no"):
        return _Mode.ROLLOUT
    raise ValueError("SIM_EXPERIMENTAL must be 1 or yes to enable experimental motion")


def _command(image, source, outputs, manifest, endpoint, container, timeout, mode=_Mode.ROLLOUT):
    command = [
        "docker",
        "run",
        "--name",
        container,
        "--init",
        "--gpus",
        "all",
        "--network",
        "host",
        "--shm-size",
        "2g",
        "--user",
        f"{_RUNTIME_UID}:{_RUNTIME_UID}",
        "--env",
        "ACCEPT_EULA=Y",
        "--env",
        "NVIDIA_DRIVER_CAPABILITIES=all",
        "--env",
        f"POLICY_ENDPOINT={endpoint}",
        "--mount",
        f"type=bind,src={source},dst={_SOURCE},readonly",
        "--mount",
        f"type=bind,src={outputs},dst={_OUTPUT}",
        "--workdir",
        _SOURCE,
        "--entrypoint",
        "/usr/bin/timeout",
        image,
        "--signal=TERM",
        f"--kill-after={_KILL_GRACE}",
        str(timeout),
        "/isaac-sim/python.sh",
        "--no-ros-env",
        "-m",
        "sim_worker.rollout",
        "--manifest",
        f"{_SOURCE}/{manifest}",
        "--output-dir",
        _OUTPUT,
    ]
    if mode == _Mode.EXPERIMENTAL:
        command.append("--experimental")
    return command


def _check_result(outputs):
    record = json.loads((outputs / "result.json").read_text())
    if not isinstance(record, dict) or record.get("status") != "succeeded":
        raise ValueError("Rollout did not report success")
    for name in ("trajectory.jsonl", "video.mp4"):
        if not (outputs / name).is_file() or not (outputs / name).stat().st_size:
            raise ValueError(f"Successful rollout has no nonempty {name}")
    return record


def _cleanup(container, state):
    # The Docker daemon can outlive cancellation of its client.
    with (state / "cleanup.log").open("w") as stream:
        try:
            subprocess.run(
                ["docker", "rm", "-f", container],
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=_CLIENT_GRACE,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            stream.write(str(error) + "\n")


def _run(workdir):
    mode = _mode()
    image = os.environ["SIM_IMAGE"]
    if not re.fullmatch(_IMAGE_PATTERN, image):
        raise ValueError("SIM_IMAGE must use an immutable @sha256 digest")
    if os.environ.get("ACCEPT_EULA") != "Y":
        raise ValueError("ACCEPT_EULA=Y is required")
    manifest = Path(os.environ["SIM_MANIFEST"])
    resolved = (workdir / manifest).resolve()
    if (
        manifest.is_absolute()
        or ".." in manifest.parts
        or not resolved.is_relative_to(workdir)
        or not resolved.is_file()
    ):
        raise ValueError("SIM_MANIFEST must identify a relative worker file")
    timeout = int(os.environ.get("SIM_JOB_TIMEOUT", "3600"))
    if not 0 < timeout <= _MAX_TIMEOUT:
        raise ValueError(f"SIM_JOB_TIMEOUT must be within {_MAX_TIMEOUT} seconds")
    run_id = uuid.uuid4().hex
    state = Path.home() / "sim-rollouts" / run_id
    source = state / "source"
    artifacts = state / "artifacts"
    artifacts.mkdir(parents=True)
    container = f"isaac-rollout-{run_id}"
    destination = os.environ["SIM_RESULTS_URI"].rstrip("/") + "/" + run_id
    report = {
        "run_id": run_id,
        "status": "failed",
        "exit_code": _FAILURE,
        "started_at": datetime.now(_UTC).isoformat(),
        "image": image,
        "mode": mode.value,
    }
    print(f"Rollout files: {artifacts}\nArtifact destination: {destination}", flush=True)
    try:
        discovery_timeout = int(os.environ.get("POLICY_DISCOVERY_TIMEOUT", "900"))
        discovered = _driver(
            "discover-policy",
            os.environ["ROLLOUT_ID"],
            "--timeout",
            str(discovery_timeout),
            capture_output=True,
            text=True,
            timeout=discovery_timeout + _CLIENT_GRACE,
        )
        endpoint = f"http://{discovered.stdout.strip()}:{_PORT}"
        shutil.copytree(workdir, source, ignore=_COPY_IGNORE)
        for path in (source, *source.rglob("*")):
            path.chmod(0o755 if path.is_dir() else 0o644)
        outputs = artifacts / "outputs"
        subprocess.run(
            [
                "sudo",
                "install",
                "-d",
                "-m",
                "0755",
                "-o",
                str(_RUNTIME_UID),
                "-g",
                str(_RUNTIME_UID),
                str(outputs),
            ],
            check=True,
            timeout=_CLIENT_GRACE,
        )
        command = _command(
            image, source, outputs, manifest.as_posix(), endpoint, container, timeout, mode
        )
        with (artifacts / "worker.log").open("w") as logs:
            result = subprocess.run(
                command,
                stdout=logs,
                stderr=subprocess.STDOUT,
                timeout=timeout + _CLIENT_GRACE,
                check=False,
            )
        report["exit_code"] = result.returncode
        if result.returncode == _SUCCESS:
            report["rollout_result"] = _check_result(outputs)
            report["status"] = "succeeded"
    except subprocess.TimeoutExpired as error:
        report.update(exit_code=_TIMEOUT, error=str(error))
    except _Cancelled as error:
        report.update(exit_code=error.code, error=str(error))
    except Exception as error:
        report.update(exit_code=_FAILURE, error=str(error))
    finally:
        _cleanup(container, artifacts)
    report["finished_at"] = datetime.now(_UTC).isoformat()
    (artifacts / "job-result.json").write_text(json.dumps(report, indent=2) + "\n")
    try:
        _driver("publish-tree", str(artifacts), destination, timeout=_MAX_TIMEOUT)
    except (OSError, subprocess.SubprocessError) as error:
        print(f"Artifact upload failed: {error}; preserve {artifacts}", file=sys.stderr)
        return _FAILURE
    return report["exit_code"]


def _main():
    signal.signal(signal.SIGINT, _cancel)
    signal.signal(signal.SIGTERM, _cancel)
    return _run(Path.cwd().resolve())


if __name__ == "__main__":
    raise SystemExit(_main())
