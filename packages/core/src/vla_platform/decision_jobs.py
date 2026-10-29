"""Scoring owned by the same durable supervisor as other project jobs."""

import asyncio

from fastapi import HTTPException

from vla_platform.contracts import TERMINAL
from vla_platform.decision_api import _busy, configuration, supervised


class ConnectedJob:
    async def is_disconnected(self):
        # Browser navigation does not cancel a durable job. Its owner handles cancellation.
        return False


async def run(execution, initial):
    acquired = False
    try:
        paths = configuration()
        if not _busy.acquire(blocking=False):
            raise ValueError("The decision scorer is busy. Submit a new job when it is available.")
        acquired = True
        async with execution.lock:
            job = await execution.get(initial.id)
            if job is None or job.status in TERMINAL:
                return
            job.status, job.stage = "running", "scoring"
            await execution.save(job)
        result = await supervised(
            job.request.model_dump(exclude={"operation"}), paths, ConnectedJob()
        )
        async with execution.lock:
            job = await execution.get(initial.id)
            if job is not None and job.status not in TERMINAL:
                job.status, job.stage, job.result = "succeeded", "complete", result
                await execution.save(job)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        async with execution.lock:
            job = await execution.get(initial.id)
            if job is not None and job.status == "cancelled" and job.cancellation:
                execution._unconfirmed_cancellation(job, exc, stage=job.stage or "scoring")
                await execution.save(job)
            elif job is not None and job.status not in TERMINAL:
                job.status = "failed"
                job.error = str(exc.detail if isinstance(exc, HTTPException) else exc)[:2000]
                await execution.save(job)
    finally:
        if acquired:
            _busy.release()
