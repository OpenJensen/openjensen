"""One workspace supervisor, with JSON-only isolated worker boundaries."""

import asyncio
import sys
from pathlib import Path
from uuid import uuid4

from sqlalchemy import insert, select, update

from vla_platform.contracts import TERMINAL, IntakeRequest, Job, WorkerRequest, WorkerResult, now
from vla_platform.lifecycle.contracts import PolicyRequest
from vla_platform.lifecycle.service import Lifecycle
from vla_platform.settings import Settings
from vla_platform.storage import Storage, jobs


class Execution:
    def __init__(self, storage: Storage, settings: Settings):
        self.storage, self.settings = storage, settings
        self.tasks: dict[str, asyncio.Task] = {}
        self.lock = asyncio.Lock()
        self.native_slots = asyncio.Semaphore(1)
        self.lifecycle = Lifecycle(self)
        self.slots = asyncio.Semaphore(2)  # Metadata jobs only; not a GPU admission policy.

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
        job.updated_at = now()
        async with self.storage.engine.begin() as connection:
            await connection.execute(
                update(jobs)
                .where(jobs.c.id == job.id)
                .values(status=job.status, record=job.model_dump())
            )

    async def submit(self, project_id: str, request: IntakeRequest | PolicyRequest) -> Job:
        if isinstance(request, PolicyRequest):
            await self.lifecycle.validate(project_id, request)
        job = Job(
            id=str(uuid4()),
            project_id=project_id,
            kind=request.operation if isinstance(request, PolicyRequest) else "dataset.inspect",
            request=request,
            created_at=now(),
            updated_at=now(),
        )
        async with self.storage.engine.begin() as connection:
            await connection.execute(
                insert(jobs).values(
                    id=job.id, project_id=project_id, status=job.status, record=job.model_dump()
                )
            )
        task = asyncio.create_task(self.run(job.id))
        self.tasks[job.id] = task
        task.add_done_callback(lambda _: self.tasks.pop(job.id, None))
        return job

    async def cancel(self, job_id: str) -> Job | None:
        async with self.lock:
            job = await self.get(job_id)
            if job is None or job.status in TERMINAL:
                return job
            job.status = "cancelled"
            job.error = "Cancelled by the user; no later stage will be published."
            await self.save(job)
            task = self.tasks.get(job_id)
            if task:
                task.cancel()
        if task:
            await asyncio.gather(task, return_exceptions=True)
        return job

    async def finish(self, job_id: str, response: WorkerResult) -> None:
        async with self.lock:
            job = await self.get(job_id)
            if job and job.status not in TERMINAL:
                job.status = "succeeded" if response.result else "failed"
                job.result, job.error = response.result, response.error
                await self.save(job)

    def write_request(self, job: Job) -> tuple[Path, Path]:
        directory = self.settings.data_dir / "jobs" / job.id
        directory.mkdir(parents=True, exist_ok=True)
        request_path, result_path = directory / "request.json", directory / "result.json"
        request_path.write_text(
            WorkerRequest(
                intake=job.request,
                local_root=str(self.settings.local_root) if self.settings.local_root else None,
            ).model_dump_json(),
            encoding="utf-8",
        )
        return request_path, result_path

    async def run(self, job_id: str) -> None:
        initial = await self.get(job_id)
        if initial and isinstance(initial.request, PolicyRequest):
            await self.run_policy(initial)
            return
        process = None
        try:
            async with self.slots:
                async with self.lock:
                    job = await self.get(job_id)
                    if job is None or job.status in TERMINAL:
                        return
                    job.status = "running"
                    await self.save(job)
                request_path, result_path = await asyncio.to_thread(self.write_request, job)
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-m",
                    "vla_platform.datasets.worker",
                    str(request_path),
                    str(result_path),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(process.wait(), timeout=90)
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
            if process is not None and process.returncode is None:
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=5)
                except TimeoutError:
                    process.kill()
                    await process.wait()
                except ProcessLookupError:
                    pass

    async def run_policy(self, job: Job) -> None:
        try:
            async with self.native_slots:
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
            await self.lifecycle.event(job, "failed", f"{type(exc).__name__}: {exc}")
            async with self.lock:
                current = await self.get(job.id)
                if current.status not in TERMINAL:
                    current.status = "failed"
                    current.error = f"{type(exc).__name__}: {str(exc)[:2000]}"
                    await self.save(current)

    async def reconcile(self) -> None:
        async with self.storage.engine.connect() as connection:
            values = (
                (
                    await connection.execute(
                        select(jobs.c.record).where(jobs.c.status.in_(["queued", "running"]))
                    )
                )
                .scalars()
                .all()
            )
        for value in values:
            job = Job.model_validate(value)
            job.status = "interrupted"
            job.error = (
                "Application stopped before completion. Submit an explicit new job to retry; "
                "completed stage artifacts are preserved."
            )
            await self.save(job)

    async def close(self) -> None:
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.reconcile()
