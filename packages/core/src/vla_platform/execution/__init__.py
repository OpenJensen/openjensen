"""One workspace supervisor, with JSON-only isolated worker boundaries."""

import asyncio
import logging
import os
import re
import time
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy import insert, select, update
from sqlalchemy.exc import IntegrityError

from vla_platform.augmentation.contracts import AugmentationRequest
from vla_platform.augmentation.service import Augmentation
from vla_platform.contracts import (
    TERMINAL,
    IntakeRequest,
    Job,
    JobCancellation,
    WorkerRequest,
    WorkerResult,
    now,
)
from vla_platform.datasets.cache import Inspections
from vla_platform.datasets.recordings import Recordings
from vla_platform.decision_contracts import DecisionRequest
from vla_platform.environment import getenv
from vla_platform.frozen_commands import intake_command
from vla_platform.job_observability import JobObservability, redact
from vla_platform.lifecycle.contracts import PolicyRequest, SimulationTarget
from vla_platform.lifecycle.service import Lifecycle
from vla_platform.scenes.agent import SceneError
from vla_platform.scenes.contracts import SceneRequest
from vla_platform.scenes.models import resolve as resolve_scene_agent
from vla_platform.scenes.service import Scenes
from vla_platform.settings import Settings
from vla_platform.storage import Storage, job_submissions, jobs
from vla_platform.submissions import Submission, SubmissionConflict, SubmissionRequest, lookup
from vla_platform.teaching_sessions.contracts import TeachingCaptureRequest
from vla_platform.teaching_sessions.service import TeachingSessions

# Recovery state machine: terminal states are absorbing; retries require a new job ID.
RECOVERY_TRANSITIONS = {"queued": "interrupted", "running": "interrupted"}
SIMULATION_RECOVERY_SECONDS = 30
SIMULATION_RECOVERY_BUDGET_SECONDS = 60
SIMULATION_RECOVERY_RECORD_LIMIT = 256
_LOG = logging.getLogger(__name__)


class Execution:
    def __init__(self, storage: Storage, settings: Settings):
        self.storage, self.settings = storage, settings
        self.observability = JobObservability(settings.data_dir)
        self.tasks: dict[str, asyncio.Task] = {}
        self.lock = asyncio.Lock()
        self.native_slots = asyncio.Semaphore(1)
        self.cloud_slots = asyncio.Semaphore(2)
        self.target_slots: dict[str, asyncio.Semaphore] = {}
        self.simulation_slots = asyncio.Semaphore(1)
        self._simulation_cancellation_requested: set[str] = set()
        self.lifecycle = Lifecycle(self)
        self.augmentation = Augmentation(self)
        self.inspections = Inspections(self)
        self.recordings = Recordings(self)
        self.teaching = TeachingSessions(self)
        self.scenes = Scenes(settings.data_dir)
        self.scene_explorer = None
        self.scene_slots = asyncio.Semaphore(1)
        self._scene_admission_lock = asyncio.Lock()
        self.slots = asyncio.Semaphore(2)  # Metadata jobs only; not a GPU admission policy.
        self._submission_locks = [asyncio.Lock() for _ in range(64)]
        self._accepting: set[asyncio.Task] = set()
        self._closing = False
        self.cloud_cleanup_pending = False

    async def get(self, job_id: str) -> Job | None:
        async with self.storage.engine.connect() as connection:
            value = (
                await connection.execute(select(jobs.c.record).where(jobs.c.id == job_id))
            ).scalar_one_or_none()
            return Job.model_validate(value) if value else None

    async def list(self, project_id: str) -> list[Job]:
        async with self.storage.engine.connect() as connection:
            values = (
                await connection.execute(
                    select(jobs.c.record).where(jobs.c.project_id == project_id)
                )
            ).scalars()
            return sorted(
                [Job.model_validate(value) for value in values], key=lambda j: j.created_at
            )

    async def save(self, job: Job) -> None:
        if job.error:
            job.error = redact(job.error)
        job.updated_at = now()
        async with self.storage.engine.begin() as connection:
            previous = (
                await connection.execute(select(jobs.c.record).where(jobs.c.id == job.id))
            ).scalar_one_or_none()
            await connection.execute(
                update(jobs)
                .where(jobs.c.id == job.id)
                .values(status=job.status, record=job.model_dump())
            )
        if previous and (previous.get("status"), previous.get("stage"), previous.get("error")) != (
            job.status,
            job.stage,
            job.error,
        ):
            self.observability.try_record(
                job.id,
                job.error or f"Job {job.status}",
                stage=job.stage or job.status,
                severity="error" if job.status == "failed" else "info",
                data={"status": job.status},
            )
            if job.status == "failed":
                self.observability.try_failure(
                    job.id,
                    stage=job.stage or "execution",
                    message=job.error or "Job failed",
                    code="WORKER_FAILED",
                )

    async def submit(
        self, project_id: str, request: SubmissionRequest, *, idempotency_key: str | None = None
    ) -> Job:
        if isinstance(request, SceneRequest):
            # Scene artifact identities are global. Serialize admission across
            # client keys and projects before reserving that identity durably.
            async with self._scene_admission_lock:
                return await self._submit_registered(
                    project_id, request, idempotency_key=idempotency_key
                )
        return await self._submit_registered(project_id, request, idempotency_key=idempotency_key)

    async def _submit_registered(
        self, project_id: str, request: SubmissionRequest, *, idempotency_key: str | None
    ) -> Job:
        if isinstance(request, SceneRequest) and idempotency_key is None:
            idempotency_key = request.request_id
        if idempotency_key is None:
            return await self._submit(project_id, request)
        submission = Submission.create(project_id, request, idempotency_key)
        # Bounded stripes serialize a key without retaining unbounded client keys in memory.
        import hashlib

        scope = f"{project_id}\0{submission.operation}\0{submission.key}".encode()
        lock = self._submission_locks[hashlib.sha256(scope).digest()[0] % 64]
        async with lock:
            previous = await lookup(
                self.storage,
                project_id,
                submission.operation,
                submission.key,
                fingerprint=submission.request_sha256,
            )
            if previous is not None:
                return previous
            prepared = request.model_copy(deep=True)
            if (
                isinstance(prepared, IntakeRequest)
                and prepared.source == "local"
                and prepared.recordings is None
                and prepared.library_id is None
                and self.settings.local_root is None
            ):
                raise ValueError("Local intake is disabled; configure JENSEN_LOCAL_DATA_ROOT")
            if isinstance(prepared, IntakeRequest) and not prepared.snapshot_for_training:
                resolved = await self.inspections.prepare(prepared)
                if resolved is not None:
                    async with self.inspections.lock:
                        cached = await self.inspections.find(project_id, resolved)
                        if cached is not None:
                            return await self._accept(cached, submission, new_job=False)
                        return await self._submit(project_id, resolved, submission)
            return await self._submit(project_id, prepared, submission)

    async def _submit(
        self, project_id: str, request: SubmissionRequest, submission: Submission | None = None
    ) -> Job:
        compute_target = None
        if isinstance(request, SceneRequest):
            async with self.storage.engine.connect() as connection:
                accepted = (
                    await connection.execute(
                        select(jobs.c.id)
                        .where(
                            jobs.c.record["kind"].as_string() == "scene.generate",
                            jobs.c.record["request"]["request_id"].as_string()
                            == request.request_id,
                        )
                        .limit(1)
                    )
                ).scalar_one_or_none()
            if accepted is not None:
                raise SubmissionConflict(
                    "Scene request ID already belongs to an accepted job. "
                    "Retry with the original submission key or start a new request."
                )
            source = await self.get(request.source_job_id)
            if source is None or source.project_id != project_id:
                raise ValueError("Choose an inspected dataset in this project")
            await self.scenes.validate(project_id, request, source)
            agent = resolve_scene_agent(request, self.scenes.options())
            # Freeze the selected catalog default into the accepted job, so a
            # later configuration change cannot change an already queued run.
            request = request.model_copy(update={"model": agent.model})
        elif isinstance(request, DecisionRequest):
            from vla_platform.decision_api import configuration

            configuration()
        elif isinstance(request, TeachingCaptureRequest):
            await self.teaching.validate(project_id, request)
        elif isinstance(request, PolicyRequest):
            compute_target = await self.lifecycle.validate(project_id, request)
        elif isinstance(request, AugmentationRequest):
            await self.augmentation.validate(project_id, request)
        elif request.recordings is not None:
            await self.recordings.validate(project_id, request)
        elif request.library_id:
            from vla_platform.datasets.library import DatasetLibrary

            DatasetLibrary(self.settings.data_dir, reconcile=False).resolve(
                request.library_id, project_id
            )
        job = Job(
            id=str(uuid4()),
            project_id=project_id,
            kind=request.operation
            if isinstance(
                request,
                (
                    PolicyRequest,
                    AugmentationRequest,
                    TeachingCaptureRequest,
                    DecisionRequest,
                    SceneRequest,
                ),
            )
            else "dataset.inspect",
            request=request,
            compute_target=None if isinstance(compute_target, SimulationTarget) else compute_target,
            simulation_target=compute_target
            if isinstance(compute_target, SimulationTarget)
            else None,
            created_at=now(),
            updated_at=now(),
        )
        from vla_platform.compute.contracts import ExecutionBinding, bind_runtime

        if isinstance(request, PolicyRequest):
            runtime = self.lifecycle.runtime(request.runtime_id)
            if runtime is not None:
                job.execution = bind_runtime(runtime)
                if request.execution_target_id:
                    snapshot = self.lifecycle.configured_target(request)
                    definition = snapshot.definition
                    job.execution = job.execution.model_copy(
                        update={
                            "adapter": "skypilot-target"
                            if definition.adapter == "skypilot"
                            else "local",
                            "infrastructure": definition.infrastructure,
                            "target_id": definition.id,
                            "label": definition.label,
                            "target_snapshot": snapshot.model_dump(mode="json"),
                        }
                    )
                    if definition.adapter == "skypilot":
                        import hashlib
                        import json

                        from vla_platform import cloud_compute_catalog
                        from vla_platform.compute_targets import ComputeTargets
                        from vla_platform.lifecycle import sky_runner

                        connection = ComputeTargets(self.settings.data_dir).connection(definition)
                        connection = dict(connection)
                        connection["sky_api_endpoint"] = connection.get(
                            "sky_api_endpoint"
                        ) or await cloud_compute_catalog.configured_sky_endpoint(
                            sky_runner.executable()
                        )
                        if definition.infrastructure == "gcp":
                            from vla_platform.cloud_connections import CloudConnections

                            configured = (
                                CloudConnections(self.settings.data_dir).list().providers[0].config
                            )
                            project = connection.get("project_id") or (
                                configured.project_id if configured else None
                            )
                            if not project:
                                raise ValueError("Configure the target's cloud billing project")
                            region = definition.location or (
                                configured.region if configured else None
                            )
                            if not isinstance(region, str) or not re.fullmatch(
                                r"[a-z][a-z0-9-]+[0-9]", region
                            ):
                                raise ValueError(
                                    "Choose a Google Cloud region in the target "
                                    "or its cloud settings"
                                )
                            connection["project_id"] = project
                            connection["region"] = region
                            connection["workspace"] = connection.get(
                                "workspace"
                            ) or cloud_compute_catalog.sky_workspace_name(project)
                        connection.setdefault("workspace", "default")
                        encoded = json.dumps(
                            connection, sort_keys=True, separators=(",", ":")
                        ).encode()
                        directory = self.settings.data_dir / "jobs" / job.id
                        if directory.parent.is_symlink() or directory.is_symlink():
                            raise ValueError("Job execution storage must not be a symbolic link")
                        directory.mkdir(parents=True, exist_ok=True)
                        path = directory / "execution-connection.json"
                        import tempfile

                        descriptor, temporary = tempfile.mkstemp(
                            prefix=".connection-", dir=directory
                        )
                        try:
                            with os.fdopen(descriptor, "wb") as stream:
                                stream.write(encoded)
                                stream.flush()
                                os.fsync(stream.fileno())
                            os.replace(temporary, path)
                        finally:
                            Path(temporary).unlink(missing_ok=True)
                        job.execution = job.execution.model_copy(
                            update={"connection_sha256": hashlib.sha256(encoded).hexdigest()}
                        )
            elif request.simulation is not None:
                job.execution = ExecutionBinding(
                    adapter="simulation",
                    infrastructure="gcp",
                    target_id=request.simulation.profile_id,
                    label="Simulation",
                )
        elif isinstance(request, SceneRequest):
            job.execution = ExecutionBinding(
                adapter="scene",
                infrastructure="configured",
                target_id="scene",
                label="Scene pipeline",
            )
        elif isinstance(request, DecisionRequest):
            job.execution = ExecutionBinding(
                adapter="local",
                infrastructure="local",
                target_id="decision",
                label="Decision scorer",
            )
        elif isinstance(request, AugmentationRequest):
            job.execution = ExecutionBinding(
                adapter="provider-api",
                infrastructure="google",
                target_id="augmentation",
                label="Video generation",
            )
        elif isinstance(request, TeachingCaptureRequest):
            job.execution = ExecutionBinding(
                adapter="teaching",
                infrastructure="configured",
                target_id="teaching",
                label="Teaching executor",
            )
        else:
            job.execution = ExecutionBinding(
                adapter="local",
                infrastructure="local",
                target_id="datasets",
                label="Dataset worker",
            )
        if isinstance(request, PolicyRequest):
            from vla_platform.lifecycle.telemetry import capture

            await capture(self.lifecycle, job)
        return await self._accept(job, submission)

    async def _accept(
        self, job: Job, submission: Submission | None, *, new_job: bool = True
    ) -> Job:
        if self._closing:
            raise RuntimeError("Application is stopping; no new job was accepted")
        # Only the short durable commit + dispatch boundary is shielded. Source validation
        # remains cancellable. Shutdown waits for this owner before draining worker tasks.
        task = asyncio.create_task(self._commit_and_schedule(job, submission, new_job=new_job))
        self._accepting.add(task)
        task.add_done_callback(self._accepting.discard)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
            task.result()  # Surface a failed commit rather than losing its exception.
            raise

    async def _commit_and_schedule(
        self, job: Job, submission: Submission | None, *, new_job: bool
    ) -> Job:
        try:
            async with self.storage.engine.begin() as connection:
                if new_job:
                    await connection.execute(
                        insert(jobs).values(
                            id=job.id,
                            project_id=job.project_id,
                            status=job.status,
                            record=job.model_dump(),
                        )
                    )
                if submission is not None:
                    await connection.execute(
                        insert(job_submissions).values(
                            project_id=submission.project_id,
                            operation=submission.operation,
                            idempotency_key=submission.key,
                            fingerprint_version=1,
                            request_sha256=submission.request_sha256,
                            request_record=submission.request_record,
                            job_id=job.id,
                            accepted_response=job.model_dump(),
                            created_at=now(),
                        )
                    )
        except IntegrityError:
            if submission is not None:
                previous = await lookup(
                    self.storage,
                    submission.project_id,
                    submission.operation,
                    submission.key,
                    fingerprint=submission.request_sha256,
                )
                if previous is not None:
                    return previous
            raise
        if new_job:
            self.observability.try_record(
                job.id,
                "Job accepted",
                stage="queued",
                data={
                    "operation": job.kind,
                    "execution": job.execution.model_dump() if job.execution else None,
                },
            )
            self._schedule(job.id)
        return job

    def _schedule(self, job_id: str) -> None:
        """Only the winning atomic insertion dispatches; existing receipts never dispatch."""
        task = asyncio.create_task(self.run(job_id))
        self.tasks[job_id] = task
        task.add_done_callback(lambda _: self.tasks.pop(job_id, None))

    async def cancel(self, job_id: str) -> Job | None:
        async with self.lock:
            job = await self.get(job_id)
            if job is None or job.status in TERMINAL:
                return job
            # Terminal status fences late publication while cleanup is separately tracked.
            job.status = "cancelled"
            job.cancellation = JobCancellation(requested_at=now())
            job.error = "Cancellation requested; waiting for execution cleanup."
            if isinstance(job.request, AugmentationRequest):
                job.error += " Submitted provider work may still run and incur charges."
            await self.save(job)
            task = self.tasks.get(job_id)
            if task:
                task.cancel()

        async def drain():
            outcomes = await asyncio.gather(task, return_exceptions=True) if task else []
            errors = [
                value
                for value in outcomes
                if isinstance(value, BaseException)
                and not isinstance(value, asyncio.CancelledError)
            ]
            async with self.lock:
                current = await self.get(job_id)
                if current is not None and current.cancellation is not None:
                    if errors:
                        self._unconfirmed_cancellation(
                            current, errors[0], stage=current.stage or "cleanup"
                        )
                    cleanup_failed = current.cancellation.cleanup == "failed"
                    current.cancellation.finished_at = now()
                    current.cancellation.cleanup = "failed" if cleanup_failed else "succeeded"
                    if not cleanup_failed:
                        current.error = "Cancelled by the user; owned execution has stopped."
                        if isinstance(current.request, AugmentationRequest):
                            current.error += (
                                " Submitted provider work may still run and incur charges."
                            )
                    await self.save(current)
                    self.observability.try_record(
                        current.id,
                        current.error or "Cancelled",
                        stage="cleanup",
                        severity="error" if cleanup_failed else "info",
                        data={"cleanup": current.cancellation.cleanup},
                    )
                return current

        owned = asyncio.create_task(drain())
        while not owned.done():
            try:
                await asyncio.shield(owned)
            except asyncio.CancelledError:
                continue
        return owned.result()

    def _unconfirmed_cancellation(self, job: Job, error, *, stage: str):
        """Record evidence of an interrupted cleanup owner without guessing its cause.

        Callers hold the execution lock and persist the updated job before returning.
        A task exception after cancellation means cleanup was not confirmed even if
        its text does not contain a particular provider's cleanup vocabulary.
        """
        if job.cancellation is None:
            return
        detail = redact(str(error)) or type(error).__name__
        job.cancellation.cleanup = "failed"
        job.cancellation.finished_at = now()
        job.error = (
            "Cancellation stopped with an error; execution cleanup could not be confirmed. "
            + detail[:2000]
        )
        self.observability.try_failure(
            job.id,
            stage=stage,
            message=job.error,
            code="CANCELLATION_UNCONFIRMED",
            cause=f"{type(error).__name__}: {detail}",
            cleanup="failed",
        )

    async def finish(self, job_id: str, response: WorkerResult) -> None:
        async with self.lock:
            job = await self.get(job_id)
            if (
                job
                and job.status == "cancelled"
                and job.cancellation
                and job.cancellation.cleanup == "pending"
                and response.error
            ):
                self._unconfirmed_cancellation(job, response.error, stage=job.stage or "execution")
                await self.save(job)
            elif job and job.status not in TERMINAL:
                job.status = "succeeded" if response.result else "failed"
                job.result, job.error = response.result, response.error
                await self.save(job)

    def write_request(self, job: Job) -> tuple[Path, Path]:
        directory = self.settings.data_dir / "jobs" / job.id
        directory.mkdir(parents=True, exist_ok=True)
        request_path, result_path = directory / "request.json", directory / "result.json"
        intake = job.request
        local_root = self.settings.local_root
        if isinstance(intake, IntakeRequest) and intake.library_id:
            from vla_platform.datasets.library import DatasetLibrary

            library = DatasetLibrary(self.settings.data_dir, reconcile=False)
            path = library.resolve(intake.library_id, job.project_id)
            local_root = library.root
            intake = intake.model_copy(update={"path": str(path), "library_id": None})
        from vla_platform.datasets.local_preview import reader_python

        request_path.write_text(
            WorkerRequest(
                intake=intake,
                snapshot_store=str(self.settings.data_dir / "dataset-snapshots"),
                local_root=str(local_root) if local_root else None,
                reader_python=getenv("JENSEN_CPU_READER_PYTHON")
                or str(reader_python(self.settings.data_dir)),
            ).model_dump_json(),
            encoding="utf-8",
        )
        return request_path, result_path

    async def observe_output(self, job_id, stream, name, stage="execution"):
        from vla_platform.compute.streams import output_lines

        async for line in output_lines(stream):
            self.observability.try_record(
                job_id, line, kind="log", component="worker", stage=stage, stream=name
            )

    async def run(self, job_id: str) -> None:
        initial = await self.get(job_id)
        if initial and isinstance(initial.request, DecisionRequest):
            from vla_platform.decision_jobs import run

            await run(self, initial)
            return
        if initial and isinstance(initial.request, SceneRequest):
            await self.run_scene(initial)
            return
        if initial and isinstance(initial.request, TeachingCaptureRequest):
            await self.teaching.run(initial)
            return
        if initial and isinstance(initial.request, AugmentationRequest):
            await self.augmentation.run(initial)
            return
        if initial and isinstance(initial.request, PolicyRequest):
            await self.run_policy(initial)
            return
        if initial and initial.request.recordings is not None:
            await self.recordings.run(initial)
            return
        process = None
        drains = []
        try:
            async with self.slots:
                async with self.lock:
                    job = await self.get(job_id)
                    if job is None or job.status in TERMINAL:
                        return
                    job.status = "running"
                    await self.save(job)
                request_path, result_path = await asyncio.to_thread(self.write_request, job)
                process = await self.lifecycle.act_spawn_owned(
                    *intake_command(request_path, result_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=os.name == "posix",
                )
                drains = [
                    asyncio.create_task(
                        self.observe_output(job.id, process.stdout, "stdout", "inspection")
                    ),
                    asyncio.create_task(
                        self.observe_output(job.id, process.stderr, "stderr", "inspection")
                    ),
                ]
                await asyncio.wait_for(
                    asyncio.gather(process.wait(), *drains),
                    timeout=180 if job.request.snapshot_for_training else 90,
                )
                if process.returncode != 0 or not result_path.is_file():
                    raise RuntimeError(
                        f"Metadata worker exited without a result (code {process.returncode})"
                    )
                if result_path.stat().st_size > 4 * 1024 * 1024:
                    raise ValueError("Worker result exceeds the 4 MiB result limit")
                response = WorkerResult.model_validate_json(
                    await asyncio.to_thread(result_path.read_bytes)
                )
                await self.finish(job_id, response)
        except asyncio.CancelledError:
            # The caller persists cancellation; shutdown reconciliation persists interruption.
            raise
        except Exception as exc:
            await self.finish(
                job_id, WorkerResult(error=f"{type(exc).__name__}: {str(exc)[:1000]}")
            )
        finally:
            if process is not None:
                await self.lifecycle.act_stop_owned(process)
            if drains:
                await asyncio.gather(*drains, return_exceptions=True)

    async def run_scene(self, job: Job) -> None:
        from vla_platform.datasets.explore import DatasetExplorer

        explorer = self.scene_explorer
        owned_explorer = False
        try:
            async with self.scene_slots:
                async with self.lock:
                    current = await self.get(job.id)
                    if current is None or current.status in TERMINAL:
                        return
                    current.status = "running"
                    await self.save(current)
                source = await self.get(current.request.source_job_id)
                if source is None or source.project_id != current.project_id:
                    raise SceneError("The inspected source dataset is unavailable.")
                if explorer is None:
                    explorer = DatasetExplorer(data_dir=self.settings.data_dir)
                    owned_explorer = True
                await self.scenes.run_job(current, source, explorer, self)
                latest = await self.get(job.id)
                if latest is not None and latest.status not in TERMINAL:
                    raise SceneError("Scene processing stopped before publishing its result.")
        except asyncio.CancelledError:
            async with self.lock:
                current = await self.get(job.id)
                if current and current.status not in TERMINAL:
                    current.status = "interrupted"
                    current.error = (
                        "Application stopped during scene generation. Evidence is retained; "
                        "no provider request or simulation was repeated."
                    )
                    await self.save(current)
            raise
        except Exception as exc:
            self.observability.try_failure(
                job.id,
                stage=job.stage or "scene",
                message=str(exc) or type(exc).__name__,
                cause=f"{type(exc).__name__}: {exc}",
            )
            async with self.lock:
                current = await self.get(job.id)
                if current and current.status not in TERMINAL:
                    current.status = "failed"
                    current.error = (
                        str(exc)[:2000]
                        if isinstance(exc, SceneError)
                        else "Scene generation stopped. Check source access and runtime setup; "
                        "completed evidence was retained."
                    )
                    await self.save(current)
                elif current and current.status == "cancelled" and current.cancellation:
                    self._unconfirmed_cancellation(current, exc, stage=current.stage or "scene")
                    await self.save(current)
        finally:
            if owned_explorer:
                await explorer.close()

    async def run_policy(self, job: Job) -> None:
        try:
            # Each cloud run owns an isolated GPU; local native work shares one.
            slots = (
                self.simulation_slots
                if job.simulation_target is not None
                else self.cloud_slots
                if job.compute_target is not None
                or (job.execution and job.execution.adapter == "skypilot-target")
                else self.native_slots
            )
            target_slots = asyncio.Semaphore(1)
            if job.execution and job.execution.target_snapshot:
                definition = job.execution.target_snapshot["definition"]
                target_slots = self.target_slots.setdefault(
                    definition["id"], asyncio.Semaphore(definition["resources"]["concurrency"])
                )
            async with target_slots, slots:
                async with self.lock:
                    current = await self.get(job.id)
                    if current.status in TERMINAL:
                        return
                    current.status = "running"
                    await self.save(current)
                await asyncio.wait_for(self.lifecycle.run(current), job.request.timeout_seconds)
                async with self.lock:
                    current = await self.get(job.id)
                    if current.status not in TERMINAL:
                        current.status = "succeeded"
                        await self.save(current)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.observability.try_failure(
                job.id,
                stage=job.stage or "execution",
                message=redact(str(exc)) or type(exc).__name__,
                code="JOB_DEADLINE_EXCEEDED"
                if isinstance(exc, TimeoutError)
                else "INVALID_EXECUTION"
                if isinstance(exc, ValueError)
                else "WORKER_FAILED",
                cause=f"{type(exc).__name__}: {exc}",
                cleanup="failed" if "cleanup" in str(exc).lower() else "unknown",
            )
            await self.lifecycle.event(job, "failed", f"{type(exc).__name__}: {exc}")
            async with self.lock:
                current = await self.get(job.id)
                if current.status not in TERMINAL:
                    current.status = "failed"
                    current.error = redact(f"{type(exc).__name__}: {str(exc)[:2000]}")
                    await self.save(current)
                elif current.status == "cancelled" and current.cancellation:
                    self._unconfirmed_cancellation(current, exc, stage=current.stage or "execution")
                    await self.save(current)

    async def reconcile(self) -> None:
        """Fence abandoned work before serving requests, or after shutdown drains tasks.

        SIGKILL cannot run the supervisor's child cleanup. An orphan metadata worker
        may still write its job's result.json; recovery neither adopts that result nor
        reruns the request. Preserve those files as evidence. No persisted PID is safe
        to signal after restart (PID reuse): orphan cleanup requires an operator to
        verify process identity. A blocked orphan may need explicit termination; the
        old supervisor's timeout no longer exists. Workers cannot write application
        records. Completed native stage artifacts remain registered; an orphan
        cannot publish a later stage.
        """
        recovered_jobs = []
        # Serialize with cancel/finish and commit the entire recovery batch atomically.
        # A crash before commit rolls back; after commit, repeated recovery is a no-op.
        async with self.lock:
            async with self.storage.engine.begin() as connection:
                values = (
                    (
                        await connection.execute(
                            select(jobs.c.record).where(jobs.c.status.in_(RECOVERY_TRANSITIONS))
                        )
                    )
                    .scalars()
                    .all()
                )
                for value in values:
                    job = Job.model_validate(value)
                    previous_status, previous_update = job.status, job.updated_at
                    job.status = RECOVERY_TRANSITIONS[previous_status]
                    job.updated_at = now()
                    evidence = (
                        f"Application stopped before completion; recovered {previous_status} "
                        f"job (last update {previous_update}) as interrupted at {job.updated_at}. "
                        "No result was published or work retried. Submit a new inspection to retry."
                    )
                    if previous_status == "running":
                        evidence += (
                            " An orphan metadata worker may still be running; late result files "
                            "are not adopted. Verify process identity before manual cleanup."
                        )
                    if isinstance(job.request, AugmentationRequest):
                        job.error = (
                            "Application stopped during augmentation. No outputs published or "
                            "paid requests retried. An already submitted Gemini request may "
                            "still run and incur charges. Submit a new job to retry explicitly."
                        )
                        job.result = None
                    elif isinstance(job.request, TeachingCaptureRequest):
                        job.result = None
                        job.error = (
                            "Application stopped during managed teaching. The parent-liveness "
                            "supervisor owns bounded local cleanup. No saved PID was signalled, "
                            "capture adopted, "
                            "or request redispatched. Retained raw data requires explicit review."
                        )
                    elif isinstance(job.request, (PolicyRequest, SceneRequest)):
                        evidence = evidence.replace("new inspection", "new job")
                        evidence = evidence.replace(
                            "orphan metadata worker", "orphan native worker"
                        )
                        evidence += (
                            " Completed stage artifacts are retained; no late stage is adopted."
                        )
                        job.error = f"{job.error}\n{evidence}" if job.error else evidence
                    else:
                        job.error = f"{job.error}\n{evidence}" if job.error else evidence
                        job.result = None
                    # Compare-and-swap keeps the indexed state and public record in sync.
                    await connection.execute(
                        update(jobs)
                        .where(jobs.c.id == job.id, jobs.c.status == previous_status)
                        .values(status=job.status, record=job.model_dump())
                    )
                    recovered_jobs.append((job.id, job.error))
        for ident, message in recovered_jobs:
            self.observability.try_record(
                ident,
                message or "Job interrupted after restart",
                stage="recovery",
                severity="warning",
            )
        # SkyPilot clusters outlive the API process. Reconcile only durable,
        # job-owned dispatch records, never unrelated operator clusters.
        from vla_platform.lifecycle.sky_runner import recover

        failures = await recover(self.settings.data_dir)
        self.cloud_cleanup_pending = any("cleanup" in item["error"].lower() for item in failures)
        for failure in failures:
            job = await self.get(failure["job_id"])
            if job is None:
                continue
            message = failure["error"]
            await self.lifecycle.event(job, "cloud_cleanup", message)
            async with self.lock:
                current = await self.get(job.id)
                if current and message not in (current.error or ""):
                    current.error = f"{current.error}\n{message}" if current.error else message
                    await self.save(current)

        # A process restart cannot prove that a local orphan stopped. Keep
        # cancellation unresolved rather than reporting successful cleanup.
        failed_ids = {item["job_id"] for item in failures}
        async with self.storage.engine.connect() as connection:
            cancelled = (
                (
                    await connection.execute(
                        select(jobs.c.record).where(jobs.c.status == "cancelled")
                    )
                )
                .scalars()
                .all()
            )
        for value in cancelled:
            current = Job.model_validate(value)
            if current.cancellation and current.cancellation.cleanup in {"pending", "failed"}:
                remote = bool(
                    current.compute_target
                    or (
                        current.execution
                        and current.execution.adapter in {"skypilot", "skypilot-target"}
                    )
                )
                if current.cancellation.cleanup == "failed" and not remote:
                    # A restart cannot establish that a local orphan was stopped.
                    continue
                current.cancellation.finished_at = now()
                current.cancellation.cleanup = (
                    "succeeded" if remote and current.id not in failed_ids else "failed"
                )
                current.error = (
                    "Cancellation reconciled after restart."
                    if current.cancellation.cleanup == "succeeded"
                    else "Cancellation was interrupted; execution cleanup needs verification."
                )
                await self.save(current)
        await self.reconcile_simulations()

    async def reconcile_simulations(self) -> None:
        """Reconcile only registered Isaac dispatches, without retrying the work.

        The startup budget bounds remote attempts; the runner may take another
        three seconds to reap its owned cancellation subprocess after timeout.
        Completed published records are retained. A cancellation request is never
        reported as evidence that the cloud resources were deleted.
        """
        from vla_platform.lifecycle import simulation

        try:
            async with self.storage.engine.connect() as connection:
                values = (
                    (
                        await connection.execute(
                            select(jobs.c.record)
                            .where(
                                jobs.c.record["simulation_target"]["profile_id"]
                                .as_string()
                                .is_not(None)
                            )
                            .order_by(
                                (jobs.c.status == "succeeded").asc(),
                                jobs.c.record["updated_at"].as_string().desc(),
                            )
                            .limit(SIMULATION_RECOVERY_RECORD_LIMIT)
                        )
                    )
                    .scalars()
                    .all()
                )
        except Exception:
            _LOG.warning(
                "Native simulation recovery records are unavailable; operator review required"
            )
            return
        deadline = time.monotonic() + SIMULATION_RECOVERY_BUDGET_SECONDS
        for value in values:
            try:
                job = Job.model_validate(value)
            except ValueError:
                _LOG.warning("Invalid native simulation recovery record; operator review required")
                continue
            target = job.simulation_target
            if (
                target is None
                or not isinstance(job.request, PolicyRequest)
                or job.request.operation != "policy.run"
                or job.request.simulation is None
                or job.request.simulation.profile_id != target.profile_id
                or job.id in self._simulation_cancellation_requested
            ):
                continue
            if (
                job.status == "succeeded"
                and job.result is not None
                and any(
                    artifact.format == "simulation_record"
                    and artifact.metadata.get("model_id") == target.model_id
                    for artifact in job.result.artifacts
                )
                and any(
                    report.get("stage") == "simulation"
                    and report.get("execution_status") == "succeeded"
                    for report in job.result.reports
                )
            ):
                continue
            record = {
                "status": "cleanup_unknown",
                "resource_deletion": "unverified",
                "resubmitted": False,
            }
            safe_path = False
            try:
                if str(UUID(job.id)) != job.id:
                    raise ValueError("Noncanonical job identity")
                root = self.settings.data_dir / "jobs"
                parts = [
                    root,
                    root / job.id,
                    root / job.id / "simulation",
                    root / job.id / "simulation" / "execution",
                ]
                if any(part.is_symlink() for part in parts):
                    raise ValueError("Simulation recovery links are forbidden")
                directory = parts[-1]
                if not directory.exists():
                    continue  # No saved dispatch directory: no remote action was begun here.
                if not directory.is_dir() or not directory.resolve().is_relative_to(root.resolve()):
                    raise ValueError("Invalid owned simulation execution directory")
                safe_path = True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    record["reason"] = "startup_recovery_budget_exhausted"
                else:
                    async with asyncio.timeout(min(SIMULATION_RECOVERY_SECONDS, remaining)):
                        profile = simulation.profile_for(self.settings, target.profile_id)
                        saved = await asyncio.to_thread(
                            simulation.strict_json, directory / "request.json", 65536
                        )
                        if (
                            target.model_id is None
                            or saved.get("profile_id") != target.profile_id
                            or saved.get("profile_sha256") != target.profile_sha256
                            or saved.get("model_id") != target.model_id
                            or await asyncio.to_thread(profile.identity_hash)
                            != target.profile_sha256
                        ):
                            raise ValueError("Saved simulation identity changed")
                        from vla_platform.lifecycle import isaac_runner

                        observed = await isaac_runner.recover(profile, directory)
                        if (
                            not isinstance(observed, dict)
                            or observed.get("status")
                            not in {"cleanup_unknown", "cancellation_requested"}
                            or observed.get("resubmitted") is not False
                            or observed.get("resource_deletion") != "unverified"
                        ):
                            raise ValueError("Invalid simulation recovery result")
                        record["status"] = observed["status"]
                        if record["status"] == "cancellation_requested":
                            self._simulation_cancellation_requested.add(job.id)
            except TimeoutError:
                record["reason"] = "recovery_timed_out"
            except Exception:
                # Never include operator paths, credentials or arbitrary helper errors.
                record["reason"] = "owned_identity_or_recovery_unavailable"
            await self.record_simulation_recovery(job, record, safe_path=safe_path)

    async def record_simulation_recovery(self, job, record, *, safe_path):
        message = (
            "Isaac restart recovery: "
            + record["status"]
            + "; resource deletion is unverified. No simulation was resubmitted "
            "and no late result was adopted."
        )
        if record["status"] == "cleanup_unknown":
            message += " Operator reconciliation is required."
        if safe_path:
            try:
                await self.lifecycle.event(job, "simulation_recovery", message, record)
            except Exception:
                _LOG.warning("Native simulation recovery event could not be saved")
        try:
            async with self.lock:
                current = await self.get(job.id)
                if current is not None and message not in (current.error or ""):
                    current.error = f"{current.error}\n{message}" if current.error else message
                    await self.save(current)
        except Exception:
            _LOG.warning("Native simulation recovery status could not be saved")

    async def close(self) -> None:
        self._closing = True
        await asyncio.gather(*list(self._accepting), return_exceptions=True)
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.reconcile()
