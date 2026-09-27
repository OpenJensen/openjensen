"""Native policy admission and Isaac execution through the existing job supervisor."""

import asyncio
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path

from vla_platform.lifecycle.contracts import LifecycleResult, PolicyArtifact, SimulationTarget

MAX_ARCHIVE_BYTES = 4 * 1024**3
MAX_RECEIPT_BYTES = 4 * 1024**2
UPLOAD_ID = re.compile(r"^[a-f0-9]{32}$")


async def finish_owned(operation):
    """Drain an owned write/submission before caller cancellation can clean its files."""
    task = asyncio.create_task(operation)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        # Consume failures while preserving the caller's cancellation.
        if not task.cancelled():
            task.exception()
        raise


def profiles(settings):
    from vla_platform.lifecycle.isaac_runner import load_profiles

    return load_profiles(settings.simulation_config)


def profile_for(settings, ident):
    found = next((item for item in profiles(settings) if item.id == ident), None)
    if found is None:
        raise ValueError("Isaac simulation profile is not configured on this application host")
    return found


def options(settings):
    try:
        entries = [item.public() for item in profiles(settings)]
        return {
            "profiles": entries,
            "unavailable_reason": None if entries else "No Isaac simulation profile is configured.",
            "max_archive_bytes": MAX_ARCHIVE_BYTES,
            "scored_evaluation": False,
        }
    except OSError, ValueError:
        # Configuration may contain credentials and operator paths; never echo it.
        return {
            "profiles": [],
            "unavailable_reason": "The operator's Isaac profile is unavailable or invalid.",
            "max_archive_bytes": MAX_ARCHIVE_BYTES,
            "scored_evaluation": False,
        }


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def strict_json(path, limit=MAX_RECEIPT_BYTES):
    def pairs(entries):
        result = {}
        for key, value in entries:
            if key in result:
                raise ValueError("Duplicate JSON key in native policy record")
            result[key] = value
        return result

    def finite(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Non-finite native policy metadata")
        return number

    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
            raise ValueError("Invalid native policy metadata file")
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise ValueError("Native policy metadata exceeds its bound")
    value = json.loads(
        content.decode("utf-8"), object_pairs_hook=pairs, parse_float=finite, parse_constant=finite
    )
    if not isinstance(value, dict):
        raise ValueError("Native policy metadata must be an object")
    return value


def read_upload(settings, project_id, upload_id):
    if not isinstance(upload_id, str) or not UPLOAD_ID.fullmatch(upload_id):
        raise ValueError("Invalid model upload identity")
    directory = settings.data_dir / "model-uploads" / upload_id
    ticket_path, archive = directory / "upload.json", directory / "archive.tar"
    if any(p.is_symlink() for p in [directory, ticket_path, archive]):
        raise ValueError("Model upload links are forbidden")
    if not ticket_path.is_file() or ticket_path.stat().st_size > 4096:
        raise ValueError("Model upload is missing or unavailable")
    ticket = strict_json(ticket_path, 4096)
    if ticket.get("project_id") != project_id or ticket.get("upload_id") != upload_id:
        raise ValueError("Model upload does not belong to this project")
    if (
        not archive.is_file()
        or not 0 < archive.stat().st_size <= MAX_ARCHIVE_BYTES
        or archive.stat().st_size != ticket.get("bytes")
    ):
        raise ValueError("Model upload is incomplete")
    return directory, archive, ticket


async def validate(lifecycle, project_id, request):
    profile = profile_for(lifecycle.settings, request.simulation.profile_id)
    if request.operation == "policy.import":
        if not request.source_id or request.artifact_id:
            raise ValueError("Native import requires a project-owned model upload")
        directory, _, ticket = read_upload(lifecycle.settings, project_id, request.source_id)
        if ticket.get("profile_id") != profile.id or (directory / "claimed").exists():
            raise ValueError("Model upload profile differs or the upload was already submitted")
        return None
    if request.operation != "policy.run" or not request.simulation.experimental:
        raise ValueError("Only explicitly experimental Isaac Run is available")
    artifact = await lifecycle.artifact(project_id, request.artifact_id)
    directory = source_directory(lifecycle, artifact)
    await asyncio.to_thread(check_bundle, directory, artifact.manifest_sha256)
    return SimulationTarget(
        profile_id=profile.id,
        profile_sha256=await asyncio.to_thread(profile.identity_hash),
        source_manifest_sha256=artifact.manifest_sha256,
    )


def source_directory(lifecycle, artifact):
    if artifact.format not in {"native_checkpoint", "inference_export"}:
        raise ValueError("Isaac needs a complete native ACT or SmolVLA checkpoint; export it first")
    if artifact.metadata.get("architecture") not in {"act", "smolvla"}:
        raise ValueError("Isaac requires an explicitly identified ACT or SmolVLA checkpoint")
    directory = lifecycle.settings.data_dir / artifact.path
    jobs = lifecycle.settings.data_dir / "jobs"
    if not directory.resolve().is_relative_to(jobs.resolve()) or directory.is_symlink():
        raise ValueError("Native policy must stay in its application job directory")
    if (directory / "remote.json").exists():
        raise ValueError("Download/export the remote checkpoint before native simulation")
    return directory


def check_bundle(directory, expected_sha):
    from vla_platform.lifecycle.service import validate_bundle

    _, actual, _ = validate_bundle(directory, directory.parent)
    if actual != expected_sha:
        raise ValueError("Registered policy changed after it was saved")


async def resolve(lifecycle, profile, source, destination, receipt_path, *, archive=False):
    """Call only the registered, pinned worker module; never import ML into the API."""
    worker = profile.runner_root / "workers" / "isaac_sim"
    command = [
        str(profile.python),
        "-m",
        "sim_worker.rollout.checkpoint_package",
        "--source",
        str(source),
        "--output-dir",
        str(destination),
        "--json-output",
        str(receipt_path),
    ]
    if archive:
        command.append("--archive")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(worker)
    process = None
    log_path = receipt_path.with_suffix(".log")
    try:
        with log_path.open("xb") as log:
            process = await lifecycle.act_spawn_owned(
                *command,
                cwd=worker,
                env=environment,
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
                **({"start_new_session": True} if os.name != "nt" else {}),
            )
            await asyncio.wait_for(process.wait(), 300)
        if process.returncode != 0:
            raise ValueError(
                "Native policy import failed. Supply a complete ACT or SmolVLA export with "
                "configuration, safetensors weights, saved processors and all statistics."
            )
        return await asyncio.to_thread(check_receipt, destination, receipt_path)
    finally:
        if process is not None and process.returncode is None:
            await lifecycle.act_stop_owned(process)


def check_receipt(destination, receipt_path):
    if receipt_path.is_symlink() or not 0 < receipt_path.stat().st_size <= MAX_RECEIPT_BYTES:
        raise ValueError("Invalid native policy import receipt")
    receipt = strict_json(receipt_path)
    relative = receipt.get("directory")
    if (
        not isinstance(relative, str)
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ValueError("Native policy import returned an invalid directory")
    model = destination / relative
    if not model.is_dir() or not model.resolve().is_relative_to(destination.resolve()):
        raise ValueError("Native policy import escaped its output directory")
    files = receipt.get("files")
    if not isinstance(files, dict) or not files or len(files) > 256:
        raise ValueError("Native policy import has no bounded file inventory")
    actual = {}
    total, entries = 0, 0
    for path in destination.rglob("*"):
        entries += 1
        if entries > 256:
            raise ValueError("Native policy file inventory exceeds its bound")
        if path.is_symlink():
            raise ValueError("Native policy links are forbidden")
        mode = path.stat().st_mode
        if not stat.S_ISREG(mode) and not stat.S_ISDIR(mode):
            raise ValueError("Native policy contains a nonregular file")
        if path.is_file():
            total += path.stat().st_size
            if total > MAX_ARCHIVE_BYTES:
                raise ValueError("Native policy exceeds its byte limit")
            actual[path.relative_to(destination).as_posix()] = {
                "sha256": sha(path),
                "bytes": path.stat().st_size,
            }
    if actual != files:
        raise ValueError("Native policy files differ from the worker receipt")
    info = receipt.get("checkpoint", {})
    if (
        type(receipt.get("schema_version")) is not int
        or receipt["schema_version"] != 1
        or info.get("policy_type") not in {"act", "smolvla"}
        or not isinstance(info.get("model_id"), str)
        or not re.fullmatch(r"sha256:[a-f0-9]{64}", info["model_id"])
    ):
        raise ValueError("Invalid native policy metadata")
    from .control_provenance import policy_claims

    policy_claims(model, info)
    return model, receipt


def manifest(directory, metadata):
    files = {}
    total = 0
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Output links are forbidden")
        if path.is_file():
            files[path.relative_to(directory).as_posix()] = sha(path)
            total += path.stat().st_size
    path = directory / "manifest.json"
    with path.open("x") as stream:
        json.dump({"schema_version": 1, "files": files, "metadata": metadata}, stream, indent=2)
        stream.write("\n")
    return sha(path), total


RECORD_FILES = {
    "artifacts/job-result.json": 1024**2,
    "artifacts/outputs/result.json": 1024**2,
    "artifacts/outputs/trajectory.jsonl": 64 * 1024**2,
    "artifacts/outputs/video.mp4": 400 * 1024**2,
}


def save_record(directory, destination, report, target):
    """Publish only the four generation-bound outputs of this exact completed run."""
    if (
        report.get("execution_status") != "succeeded"
        or report.get("model_id") != target.model_id
        or report.get("profile_sha256") != target.profile_sha256
        or report.get("profile_id") != target.profile_id
        or report.get("task_success") is not None
        or report.get("calibration_verified") is not False
        or not re.fullmatch(r"[a-f0-9]{32}", str(report.get("run_id", "")))
        or strict_json(directory / "report.json") != report
    ):
        raise ValueError("Simulation report does not match this accepted model and profile")
    inventory = report.get("artifacts")
    if not isinstance(inventory, list) or len(inventory) != len(RECORD_FILES):
        raise ValueError("Simulation report is missing its required outputs")
    seen = set()
    destination.mkdir(exist_ok=False)
    for entry in inventory:
        if not isinstance(entry, dict):
            raise ValueError("Invalid simulation file inventory")
        name = entry.get("path")
        if not isinstance(name, str) or name not in RECORD_FILES or name in seen:
            raise ValueError("Invalid or repeated simulation output")
        seen.add(name)
        size, digest = entry.get("bytes"), entry.get("sha256")
        if (
            type(size) is not int
            or not 0 < size <= RECORD_FILES[name]
            or not isinstance(digest, str)
            or not re.fullmatch(r"[a-f0-9]{64}", digest)
            or not isinstance(entry.get("generation"), str)
            or not entry["generation"].isdigit()
        ):
            raise ValueError("Invalid simulation output identity")
        source = directory / name
        if any(path.is_symlink() for path in [source, *source.parents] if path != directory.parent):
            raise ValueError("Simulation output links are forbidden")
        output = destination / name
        output.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
        with os.fdopen(os.open(source, flags), "rb") as stream, output.open("xb") as copied:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size != size:
                raise ValueError("Simulation output changed or is not a regular file")
            actual, count = hashlib.sha256(), 0
            while block := stream.read(min(1024**2, size + 1 - count)):
                count += len(block)
                if count > size:
                    raise ValueError("Simulation output grew during collection")
                actual.update(block)
                copied.write(block)
        if count != size or actual.hexdigest() != digest:
            raise ValueError("Simulation output differs from its verified collection receipt")
    (destination / "report.json").write_text(json.dumps(report, allow_nan=False) + "\n")
    return manifest(
        destination,
        {
            "kind": "isaac_simulation",
            "profile_id": target.profile_id,
            "model_id": target.model_id,
            "task_object": "cup",
            "task_success": None,
            "calibration_verified": False,
        },
    )


def video_path(settings, job):
    if (
        job.status != "succeeded"
        or job.simulation_target is None
        or not isinstance(job.result, LifecycleResult)
    ):
        raise ValueError("No completed native simulation record")
    records = [item for item in job.result.artifacts if item.format == "simulation_record"]
    if len(records) != 1:
        raise ValueError("Missing simulation record")
    artifact = records[0]
    directory = settings.data_dir / artifact.path
    expected = settings.data_dir / "jobs" / job.id / "simulation" / "record"
    if directory != expected or directory.is_symlink():
        raise ValueError("Simulation record is outside its owned directory")
    check_bundle(directory, artifact.manifest_sha256)
    report = strict_json(directory / "report.json")
    if report.get("model_id") != job.simulation_target.model_id:
        raise ValueError("Simulation video model identity differs")
    return directory / "artifacts" / "outputs" / "video.mp4"


async def cleanup_evidence(lifecycle, job, execution_directory):
    path = execution_directory / "recovery.json"
    if not path.exists():
        return
    try:
        record = strict_json(path, 65536)
        status = record.get("status")
        if status not in {"cleanup_unknown", "cancellation_requested"}:
            status = "cleanup_unknown"
    except OSError, ValueError:
        status = "cleanup_unknown"
    message = f"Isaac cleanup: {status}; resource deletion remains unverified."
    if status == "cleanup_unknown":
        message += " Operator reconciliation is required; this job will not be retried."
    await lifecycle.event(job, "simulation_cleanup", message, {"status": status})
    async with lifecycle.execution.lock:
        current = await lifecycle.execution.get(job.id)
        if current is not None and message not in (current.error or ""):
            current.error = f"{current.error}\n{message}" if current.error else message
            await lifecycle.execution.save(current)


async def run(lifecycle, job):
    from vla_platform.lifecycle import isaac_runner

    request = job.request
    profile = profile_for(lifecycle.settings, request.simulation.profile_id)
    directory = lifecycle.settings.data_dir / "jobs" / job.id / "simulation"
    directory.mkdir(parents=True, exist_ok=False)
    result = LifecycleResult(decision="diagnostics_only")
    if request.operation == "policy.import":
        upload_dir, source, ticket = read_upload(
            lifecycle.settings, job.project_id, request.source_id
        )
        with (upload_dir / "claimed").open("x") as claimed:
            claimed.write(job.id)
        if await asyncio.to_thread(sha, source) != ticket["sha256"]:
            raise ValueError("Uploaded model archive changed")
        await lifecycle.event(
            job, "model_import", "Validating and copying the native policy archive"
        )
        artifact_dir = directory / "artifact"
        artifact_dir.mkdir()
        model, receipt = await resolve(
            lifecycle,
            profile,
            source,
            artifact_dir / "payload",
            directory / "import-receipt.json",
            archive=True,
        )
        if receipt.get("source_sha256") != ticket["sha256"]:
            raise ValueError("The imported archive differs from the uploaded bytes")
        info = receipt["checkpoint"]
        from .control_provenance import policy_claims

        metadata = {
            **policy_claims(model, info),
            "architecture": info["policy_type"],
            "model_id": info["model_id"],
            "policy_subdirectory": model.relative_to(artifact_dir).as_posix(),
            "checkpoint": info,
            "source_archive_sha256": ticket["sha256"],
            "inference_only": True,
            "training_resume_supported": False,
            "calibration_verified": False,
            "task_success": None,
            "runtime_verified": False,
            "provenance": "user_import; training lineage unverified",
        }
        artifact_sha, total = await asyncio.to_thread(manifest, artifact_dir, metadata)
        result.artifacts.append(
            PolicyArtifact(
                id=f"{job.id}:operation",
                project_id=job.project_id,
                job_id=job.id,
                label=f"Imported {info['policy_type']} policy",
                format="inference_export",
                path=artifact_dir.relative_to(lifecycle.settings.data_dir).as_posix(),
                manifest_sha256=artifact_sha,
                file_bytes=total,
                metadata=metadata,
            )
        )
        result.reports.append({"stage": "model_import", **metadata})
        await lifecycle.publish(job, result)
        await lifecycle.event(
            job, "model_import", "Native policy imported; task quality is unmeasured"
        )
        # The project artifact is durable. Release only this upload's temporary copy.
        source.unlink()
        return result

    artifact = await lifecycle.artifact(job.project_id, request.artifact_id)
    target = job.simulation_target
    if target is None or target.source_manifest_sha256 != artifact.manifest_sha256:
        raise ValueError("Simulation source differs from its accepted job target")
    source = source_directory(lifecycle, artifact)
    await asyncio.to_thread(check_bundle, source, target.source_manifest_sha256)
    if await asyncio.to_thread(profile.identity_hash) != target.profile_sha256:
        raise ValueError("Simulation profile changed after submission; no GPU was requested")
    await lifecycle.event(job, "simulation_preflight", "Checking the selected native policy")
    model, receipt = await resolve(
        lifecycle,
        profile,
        source,
        directory / "policy",
        directory / "policy-receipt.json",
    )
    if (
        receipt["files"].get("contents/manifest.json", {}).get("sha256")
        != target.source_manifest_sha256
        or receipt["checkpoint"]["policy_type"] != artifact.metadata["architecture"]
    ):
        raise ValueError("The copied policy differs from the accepted artifact")
    from .control_provenance import policy_claims

    policy_claims(model, artifact.metadata)
    admission = isaac_runner.admit(profile, receipt["checkpoint"])
    target.model_id = receipt["checkpoint"]["model_id"]
    async with lifecycle.execution.lock:
        current = await lifecycle.execution.get(job.id)
        if current is None or current.status != "running":
            raise asyncio.CancelledError
        current.simulation_target = target
        await lifecycle.execution.save(current)
    await lifecycle.event(
        job, "simulation_preflight", "Native policy and cup scene accepted", admission
    )

    async def event(stage, message, data=None):
        await lifecycle.event(job, stage, message, data)

    try:
        report = await isaac_runner.run(
            profile,
            model,
            directory / "execution",
            event,
            request.timeout_seconds,
            expected_profile_sha256=target.profile_sha256,
            expected_model_id=target.model_id,
        )
    finally:
        await finish_owned(cleanup_evidence(lifecycle, job, directory / "execution"))
    record = directory / "record"
    record_sha, record_bytes = await finish_owned(
        asyncio.to_thread(save_record, directory / "execution", record, report, target)
    )
    result.artifacts.append(
        PolicyArtifact(
            id=f"{job.id}:simulation",
            project_id=job.project_id,
            job_id=job.id,
            label=f"{artifact.metadata['architecture']} cup simulation record",
            format="simulation_record",
            path=record.relative_to(lifecycle.settings.data_dir).as_posix(),
            manifest_sha256=record_sha,
            file_bytes=record_bytes,
            metadata={
                "model_id": target.model_id,
                "task_success": None,
                "calibration_verified": False,
            },
        )
    )
    result.reports.append({"stage": "simulation", **report})
    # The runner's local report is bounded and never equates completion with pickup.
    await lifecycle.publish(job, result)
    return result
