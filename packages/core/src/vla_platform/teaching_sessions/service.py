"""Existing-job adapter for one local, kernel-leased teaching process group."""

import asyncio
import hashlib
import os
import secrets
import signal
import stat
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from vla_platform.contracts import TERMINAL
from vla_platform.datasets import recordings
from vla_platform.teaching_api import public_frame, public_receipt, public_state

from . import publication
from .config import TeachingError, load, options
from .contracts import TeachingCaptureResult


@dataclass
class Live:
    profile: object
    process: asyncio.subprocess.Process
    token: str
    directory: Path
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    session_id: str | None = None
    stop_sent: bool = False


async def drain(task):
    """Finish owned cleanup even through repeated cancellation."""
    interrupted = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            interrupted = True
    result = task.result()
    if interrupted:
        raise asyncio.CancelledError
    return result


async def work(function, *args):
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    return await drain(task)


def group_present(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


async def cleanup(process) -> None:
    """Only a process handle created by this API may supply a group identity."""
    if process.pid <= 1 or process.pid == os.getpgrp():
        raise RuntimeError("Invalid owned teaching group")
    if process.stdin is not None:
        process.stdin.close()
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(process.wait(), 12)
    except TimeoutError:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    await asyncio.wait_for(process.wait(), 3)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            if not group_present(process.pid):
                return
        except PermissionError:
            pass  # Kernel disappearance can briefly report EPERM; never infer absence.
        await asyncio.sleep(0.05)
    raise RuntimeError("Teaching process-group cleanup could not be verified")


async def spawn(*args, **kwargs):
    task = asyncio.create_task(asyncio.create_subprocess_exec(*args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:

        async def finish_spawn():
            process = await task
            await cleanup(process)

        await drain(asyncio.create_task(finish_spawn()))
        raise


def acquire_lease(path: Path) -> int:
    import fcntl

    with recordings.directory(path.parent):
        pass
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
            raise TeachingError("Teaching lease must be a private regular file")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except BaseException:
        os.close(fd)
        raise


class TeachingSessions:
    def __init__(self, execution):
        self.execution = execution
        self.live: dict[str, Live] = {}
        self.stops: dict[str, asyncio.Event] = {}

    def configuration(self):
        settings = self.execution.settings
        return settings.teaching_session_config, settings.recording_config

    async def profiles(self, project_id: str):
        return await work(options, *self.configuration(), project_id)

    async def validate(self, project_id: str, request):
        if not sys.platform.startswith("linux"):
            raise TeachingError(
                "Managed local Isaac requires Linux; no remote runner is enabled", 503
            )
        try:
            profile = await work(load, *self.configuration(), request.profile_id, project_id)
        except TeachingError:
            raise
        except OSError, ValueError:
            raise TeachingError(
                "Managed teaching configuration is unavailable or invalid", 503
            ) from None
        jobs_root = self.execution.settings.data_dir / "jobs"
        capture_root = profile.recording.root(project_id)
        if jobs_root.is_relative_to(capture_root) or capture_root.is_relative_to(jobs_root):
            raise TeachingError("Teaching capture publication and private jobs must be disjoint")
        if profile.identity != request.profile_sha256:
            raise TeachingError("Teaching profile changed; refresh configured profiles", 409)
        if request.timeout_seconds > profile.value.max_seconds:
            raise TeachingError("Session exceeds its operator time limit")
        return profile

    async def owned_job(self, project_id: str, job_id: str):
        job = await self.execution.get(job_id)
        if job is None or job.project_id != project_id or job.kind != "teaching.capture":
            raise TeachingError("Teaching session job not found", 404)
        return job

    async def status(self, project_id: str, job_id: str):
        job = await self.owned_job(project_id, job_id)
        live = self.live.get(job.id)
        return {
            "job": job,
            "ready": bool(
                job.status == "running"
                and live
                and live.session_id
                and live.process.returncode is None
                and not live.stop.is_set()
            ),
            "session_id": live.session_id
            if live
            else (job.result.session_id if job.result else None),
            "stop_requested": bool(self.stops.get(job.id) and self.stops[job.id].is_set()),
        }

    async def stop(self, project_id: str, job_id: str):
        job = await self.owned_job(project_id, job_id)
        if job.status not in TERMINAL:
            self.stops.setdefault(job.id, asyncio.Event()).set()
        return await self.status(project_id, job_id)

    async def upstream(self, live: Live, path: str, payload=None, maximum=65536):
        try:
            async with (
                asyncio.timeout(4),
                httpx.AsyncClient(trust_env=False, follow_redirects=False, timeout=4) as client,
            ):
                async with client.stream(
                    "POST" if payload is not None else "GET",
                    f"http://127.0.0.1:{live.profile.value.control_port}" + path,
                    json=payload,
                    headers={"Authorization": "Bearer " + live.token},
                ) as response:
                    if response.status_code < 200 or response.status_code >= 300:
                        raise TeachingError("Teaching worker did not accept the request", 409)
                    raw = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(raw) + len(chunk) > maximum:
                            raise TeachingError("Teaching response exceeded its limit", 503)
                        raw.extend(chunk)
                    return recordings.decode(bytes(raw))
        except (httpx.HTTPError, TimeoutError, OSError, ValueError) as error:
            if isinstance(error, TeachingError):
                raise
            raise TeachingError(
                "Owned teaching worker is unavailable or returned invalid data", 503
            ) from None

    async def relay(self, project_id: str, job_id: str, path: str, payload=None):
        job = await self.owned_job(project_id, job_id)
        live = self.live.get(job.id)
        if (
            job.status != "running"
            or live is None
            or live.session_id is None
            or live.stop.is_set()
            or live.process.returncode is not None
        ):
            raise TeachingError("This teaching session is not ready for controls", 409)
        if payload is not None and payload.session_id != live.session_id:
            raise TeachingError("Command belongs to a different teaching session", 409)
        started = time.monotonic_ns()
        body = payload.model_dump() if payload is not None else None
        value = await self.upstream(live, path, body, 16 * 1024**2 if path == "/frame" else 65536)
        if path == "/frame":
            return public_frame(
                value, session_id=live.session_id, elapsed_ns=time.monotonic_ns() - started
            )
        if path == "/state":
            result = public_state(value)
            if result["session_id"] != live.session_id:
                raise TeachingError("Teaching worker session identity changed", 503)
            return result
        result = public_receipt(value)
        if result.get("state") and result["state"]["session_id"] != live.session_id:
            raise TeachingError("Teaching acknowledgment belongs to a different session", 503)
        if body is not None and (
            result["command_id"] != body["command_id"]
            or result["request_sha256"] != hashlib.sha256(recordings.canonical(body)).hexdigest()
        ):
            raise TeachingError("Teaching acknowledgment does not match the submitted command", 503)
        if body is None and result["command_id"] != path.rsplit("/", 1)[-1]:
            raise TeachingError("Teaching acknowledgment identity differs", 503)
        return result

    async def ready(self, live: Live):
        raw = live.directory / "capture" / "session.json"
        preflight = live.directory / "preflight.json"
        if not raw.is_file() or not preflight.is_file():
            return False
        meta = recordings.decode(recordings.read(raw))
        expected = recordings.decode(recordings.read(preflight))
        recordings.metadata(meta)
        if (
            expected.get("metadata", {}).get("scene_sha256") != live.profile.scene_sha256
            or expected.get("settings_sha256") != live.profile.settings_sha256
            or {key: meta.get(key) for key in expected["metadata"]} != expected["metadata"]
        ):
            raise TeachingError("Worker settings differ from accepted teaching profile")
        value = public_state(await self.upstream(live, "/state"))
        if (
            value["session_id"] != meta["session_id"]
            or value["joints"] != meta["joint_names"]
            or value["mode"] not in {"idle", "running", "paused"}
        ):
            raise TeachingError("Owned teaching readiness identity did not match")
        live.session_id = meta["session_id"]
        return True

    async def run(self, job):
        process = None
        lease = None
        published = False
        try:
            profile = await self.validate(job.project_id, job.request)
            lease = acquire_lease(Path(profile.value.lease_path))
            directory = self.execution.settings.data_dir / "jobs" / job.id / "teaching"
            directory.mkdir(mode=0o700, parents=True, exist_ok=False)
            token = secrets.token_urlsafe(48)
            token_path = directory / "control-token"
            with os.fdopen(
                os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w"
            ) as stream:
                stream.write(token)
            publication.new_json(
                directory / "ownership.json",
                {
                    "job_id": job.id,
                    "project_id": job.project_id,
                    "profile_sha256": profile.identity,
                },
            )
            async with self.execution.lock:
                current = await self.execution.get(job.id)
                if current is None or current.status in TERMINAL:
                    return
                if self.stops.get(job.id) and self.stops[job.id].is_set():
                    current.status = "cancelled"
                    current.error = "Stopped before local worker launch; no capture was published."
                    await self.execution.save(current)
                    return
                current.status, current.stage = "running", "starting_teaching"
                await self.execution.save(current)
            args = [
                sys.executable,
                "-m",
                "firebird_teaching.managed",
                "run",
                "--isaac-python",
                profile.value.isaac_python,
                "--settings",
                profile.value.settings_path,
                "--output",
                str(directory / "capture"),
                "--terminal",
                str(directory / "terminal.json"),
                "--log",
                str(directory / "worker.log"),
                "--preflight",
                str(directory / "preflight.json"),
                "--port",
                str(profile.value.control_port),
                "--seconds",
                str(job.request.timeout_seconds),
                "--max-bytes",
                str(profile.value.max_capture_bytes),
                "--lease-fd",
                str(lease),
            ]
            process = await spawn(
                *args,
                env=profile.environment(token_path),
                cwd=profile.value.worker_root,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                pass_fds=(lease,),
                start_new_session=True,
            )
            live = Live(
                profile, process, token, directory, self.stops.setdefault(job.id, asyncio.Event())
            )
            self.live[job.id] = live
            started = time.monotonic()
            async with asyncio.timeout(job.request.timeout_seconds + 15):
                while process.returncode is None:
                    if live.stop.is_set() and not live.stop_sent:
                        live.stop_sent = True
                        process.stdin.write(b"stop\n")
                        await process.stdin.drain()
                    if live.session_id is None and not live.stop.is_set():
                        try:
                            ready = await self.ready(live)
                        except TeachingError as error:
                            if error.status != 503:
                                raise
                            ready = False
                        if ready:
                            async with self.execution.lock:
                                current = await self.execution.get(job.id)
                                if current.status not in TERMINAL:
                                    current.stage = "teaching_ready"
                                    await self.execution.save(current)
                        elif time.monotonic() - started > min(120, job.request.timeout_seconds):
                            raise TeachingError(
                                "Isaac did not become ready within its startup bound"
                            )
                    await asyncio.sleep(0.1)
                await process.wait()
            owner = process

            async def cleanup_owner():
                nonlocal process
                await cleanup(owner)
                # This executes inside the shielded cleanup even if drain must
                # propagate cancellation after successful reaping. Keep ownership
                # only when cleanup itself failed or remains unverified.
                process = None

            await drain(asyncio.create_task(cleanup_owner()))
            terminal = recordings.decode(recordings.read(directory / "terminal.json", 65536))
            if (
                terminal
                != {
                    "schema_version": 1,
                    "reason": "requested_stop",
                    "worker_exit_code": 0,
                    "supervisor_pid": owner.pid,
                    "group_id": owner.pid,
                    "group_kill_required": True,
                }
                or owner.returncode != -signal.SIGKILL
                or live.session_id is None
            ):
                raise TeachingError(
                    "Teaching ended without a verified graceful stop; "
                    "raw evidence is retained privately"
                )
            if (await self.validate(job.project_id, job.request)).identity != profile.identity:
                raise TeachingError("Teaching source changed before capture verification")
            verifier = await spawn(
                sys.executable,
                "-m",
                "firebird_teaching.managed",
                "verify",
                "--capture",
                str(directory / "capture"),
                "--result",
                str(directory / "verified.json"),
                "--max-bytes",
                str(profile.value.max_capture_bytes),
                cwd=profile.value.worker_root,
                env=profile.environment(token_path),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )
            try:
                await asyncio.wait_for(verifier.wait(), 120)
                if verifier.returncode != 0:
                    raise TeachingError(
                        "Finalized capture content validation failed; raw evidence retained"
                    )
            finally:
                await drain(asyncio.create_task(cleanup(verifier)))
            value = await work(
                publication.proof,
                directory / "capture",
                directory / "verified.json",
                profile.value.max_capture_bytes,
            )
            if value["session_id"] != live.session_id:
                raise TeachingError("Final capture belongs to a different executor session")
            result = TeachingCaptureResult(
                profile_id=job.request.profile_id,
                profile_sha256=profile.identity,
                recording_configuration_sha256=profile.recording.identity,
                **{
                    key: value[key]
                    for key in (
                        "session_id",
                        "session_sha256",
                        "inventory_sha256",
                        "episodes",
                        "origin",
                        "lineage_group",
                    )
                },
            )
            root = profile.recording.root(job.project_id)
            stage = root.parent / (".teaching-" + job.id)
            await work(
                publication.stage_capture,
                directory / "capture",
                stage,
                value,
                profile.value.max_capture_bytes,
            )

            async def commit():
                nonlocal published
                async with self.execution.lock:
                    current = await self.execution.get(job.id)
                    if current is None or current.status in TERMINAL:
                        return
                    final = await work(
                        load, *self.configuration(), job.request.profile_id, job.project_id
                    )
                    if final.identity != profile.identity:
                        raise TeachingError("Teaching profile changed before publication")
                    publication.publish(stage, root, live.session_id)
                    published = True
                    current.status, current.stage, current.result = (
                        "succeeded",
                        "capture_published",
                        result,
                    )
                    await self.execution.save(current)

            await drain(asyncio.create_task(commit()))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            async with self.execution.lock:
                current = await self.execution.get(job.id)
                if current is not None and current.status not in TERMINAL:
                    current.status = "failed"
                    if isinstance(error, publication.PublicationUncertain):
                        current.error = (
                            "Capture publication or rollback could not be verified; a verified "
                            "capture may remain in the catalog. Inspect this job and retained "
                            "capture evidence before retrying."
                        )
                    elif published:
                        current.error = (
                            "A verified capture was published, but its job receipt was not saved; "
                            "inspect this job before retrying."
                        )
                    else:
                        current.error = (
                            "Managed teaching did not complete. No capture was published; "
                            "inspect the retained private job evidence."
                        )
                    await self.execution.save(current)
        finally:
            try:
                if process is not None:

                    async def final_cleanup():
                        try:
                            await cleanup(process)
                        except Exception:
                            async with self.execution.lock:
                                current = await self.execution.get(job.id)
                                if current is not None:
                                    note = (
                                        "Owned teaching cleanup is unverified; the kernel lease "
                                        "may remain held. No capture may be adopted."
                                    )
                                    current.error = (
                                        (current.error + "\n") if current.error else ""
                                    ) + note
                                    await self.execution.save(current)
                            raise

                    await drain(asyncio.create_task(final_cleanup()))
            finally:
                # Closing, never explicitly unlocking, preserves an inherited lease
                # if cleanup is uncertain and a child still holds its descriptor.
                if lease is not None:
                    os.close(lease)
                self.live.pop(job.id, None)
                self.stops.pop(job.id, None)
