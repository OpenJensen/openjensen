"""Fixed remote entry point copied into an isolated SkyPilot task workdir.

This file runs with worker Python 3.11 and imports no application dependencies.
SkyPilot 0.13 supplies SKYPILOT_INTERNAL_JOB_ID and writes the job log directory
as ~/sky_logs/<job_id>-<task_name> (sky.skylet.job_lib.add_job). Its supported
``sky logs --sync-down`` operation downloads that entire directory. Refuse to
train when that layout is unavailable, rather than risking unretrievable output.
"""

import hashlib
import json
import os
import signal
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

MAX_BYTES = 20 * 1024**3
MAX_FILES = 10000
PART_BYTES = 32 * 1024**2


def publish_archive(output, log_dir):
    index = log_dir / "checkpoint-index" / "index.json"
    published = (
        {item["name"] for item in json.loads(index.read_text()).get("checkpoints", [])}
        if index.exists()
        else set()
    )
    files = []
    total = 0
    for path in sorted(output.rglob("*")):
        relative = path.relative_to(output).parts
        if len(relative) > 1 and relative[0] == "training" and relative[1] in published:
            continue
        if path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise ValueError("Remote output contains an unsupported file")
        if not path.is_file():
            continue
        total += path.stat().st_size
        if total > MAX_BYTES or len(files) >= MAX_FILES:
            raise ValueError("Remote artifacts exceed the 20 GiB / 10000 file limit")
        files.append(path)
    # SkyPilot's ZIP downloader reads one member into memory. Small archive
    # parts bound that allocation even for multi-gigabyte training checkpoints.
    log_dir = log_dir / "final-output"
    log_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="firebird-transfer-", dir=output.parent) as folder:
        archive_path = Path(folder) / "output.tar"
        with tarfile.open(archive_path, "w") as archive:
            for path in files:
                archive.add(path, arcname=path.relative_to(output).as_posix(), recursive=False)
        parts, digest = [], hashlib.sha256()
        with archive_path.open("rb") as stream:
            while chunk := stream.read(PART_BYTES):
                name = f"firebird-output.part-{len(parts):05d}"
                (log_dir / name).write_bytes(chunk)
                parts.append({"name": name, "size": len(chunk)})
                digest.update(chunk)
        descriptor = log_dir / "firebird-output.json"
        temporary = descriptor.with_suffix(".partial")
        temporary.write_text(
            json.dumps(
                {
                    "sha256": digest.hexdigest(),
                    "size": archive_path.stat().st_size,
                    "files": len(files),
                    "parts": parts,
                }
            )
        )
        temporary.replace(descriptor)


class WorkerInterrupted(BaseException):
    def __init__(self, signum):
        self.signum = signum


def terminate_worker(process, grace):
    """Terminate only the new session created for this worker, including children."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # The leader can exit before a data-loader child which ignored TERM.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def run_worker(command, *, env, timeout, terminate_grace=10):
    """Own the trainer across timeout, interrupt and the Popen return window.

    This standalone bootstrap runs on Linux. Defer signals while obtaining the
    child handle and during bounded cleanup; never signal the SkyPilot session.
    """
    handlers = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    pending = []
    process = None
    phase = "starting"

    def interrupted(signum, frame):
        nonlocal phase
        pending.append(signum)
        if phase == "waiting":
            # Enter cleanup before raising, so repeated signals are deferred.
            phase = "cleanup"
            raise WorkerInterrupted(signum)

    try:
        for number in handlers:
            signal.signal(number, interrupted)
        try:
            process = subprocess.Popen(command, start_new_session=True, env=env)
            phase = "waiting"
            if pending:
                phase = "cleanup"
                raise WorkerInterrupted(pending[0])
            code = process.wait(timeout=timeout)
            return code
        except subprocess.TimeoutExpired:
            return 124
        except WorkerInterrupted as error:
            return 128 + error.signum
        finally:
            phase = "cleanup"
            if process is not None:
                # A completed leader may leave data-loader descendants behind.
                terminate_worker(process, terminate_grace)
    finally:
        for number, handler in handlers.items():
            signal.signal(number, handler)


def main():
    workdir = Path.cwd().resolve()
    config = json.loads((workdir / "dispatch.json").read_text())
    task_id = os.environ.get("SKYPILOT_INTERNAL_JOB_ID", "")
    if not task_id.isdigit():
        raise RuntimeError("SkyPilot did not provide its job identity; training was not started")
    log_dir = Path.home() / "sky_logs" / (task_id + "-" + config["task_name"])
    if not log_dir.is_dir() or log_dir.is_symlink():
        raise RuntimeError("SkyPilot job log directory is unavailable; training was not started")
    (log_dir / "checkpoint-index").mkdir(exist_ok=True)
    (log_dir / "checkpoint-index" / "index.json").write_text('{"checkpoints": []}\n')
    output = workdir / "output"
    output.mkdir(exist_ok=False)
    payload = json.loads((workdir / "request.json").read_text())
    payload["output_dir"] = str(output)
    if payload.get("artifact"):
        payload["artifact"]["path"] = str(workdir / "inputs" / "artifact")
    if payload.get("resume_checkpoint"):
        payload["resume_checkpoint"] = str(workdir / "inputs" / "resume")
    request_path = output / "request.json"
    result_path = output / "result.json"
    request_path.write_text(json.dumps(payload))
    storage_prefix = config.get("storage_prefix")
    if storage_prefix:
        from cloud_storage import materialize

        if (
            payload.get("artifact")
            and (Path(payload["artifact"]["path"]) / "remote.json").is_file()
        ):
            print("Downloading the selected checkpoint directly from Google Cloud", flush=True)
            materialize(payload["artifact"]["path"])
        if payload.get("resume_checkpoint"):
            pointer = Path(payload["resume_checkpoint"]) / "remote-checkpoint.json"
            if pointer.is_file():
                raise ValueError("Resume cloud checkpoint must use its registered artifact")
    worker_module = config.get("worker_module", "firebird_vla.application")
    if worker_module not in {
        "firebird_vla.application",
        "firebird_vla.lerobot_application",
        "firebird_vla.psi_application",
        "policykit.cloud_quantize",
        "policykit.cloud_inference",
    }:
        raise ValueError("Unknown cloud worker module")
    code = run_worker(
        [sys.executable, "-m", worker_module, str(request_path), str(result_path)],
        timeout=config["timeout_seconds"],
        env={
            **os.environ,
            "FIREBIRD_CHECKPOINT_EXPORT_ROOT": str(log_dir),
            "PYTHONPATH": str(workdir),
            **(
                {"FIREBIRD_GCS_PREFIX": storage_prefix, "FIREBIRD_CLOUD_REQUEST": str(request_path)}
                if storage_prefix
                else {}
            ),
        },
    )
    # Timeout/cancellation still exports completed checkpoints for explicit resume.
    if not result_path.is_file():
        result_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "job_id": payload["job_id"],
                    "error": "Remote worker timed out"
                    if code == 124
                    else "Remote worker was cancelled"
                    if code in {128 + signal.SIGINT, 128 + signal.SIGTERM}
                    else "Remote worker did not finish",
                }
            )
        )
    if result_path.stat().st_size > 4 * 1024**2:
        raise ValueError("Remote worker result exceeds the size limit")
    result = json.loads(result_path.read_text())
    if storage_prefix:
        from cloud_storage import publish_result

        publish_result(output, storage_prefix)
        return code
    if result.get("artifact"):
        artifact_path = Path(result["artifact"]["path"]).resolve()
        if not artifact_path.is_relative_to(output):
            raise ValueError("Remote artifact escaped its output directory")
        result["artifact"]["path"] = artifact_path.relative_to(output).as_posix()
        result_path.write_text(json.dumps(result))
    publish_archive(output, log_dir)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
