"""Project-scoped diagnostics; no arbitrary file paths or cross-project reads."""

import asyncio
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request, Response

from vla_platform import __version__
from vla_platform.job_observability import (
    JobFailure,
    JobLogPage,
    JobObservability,
    ObservabilityError,
    ObservabilityPolicy,
)
from vla_platform.scenes.agent import SceneError
from vla_platform.workspace_jobs import core_job, get_job

router = APIRouter(prefix="/api/v1", tags=["Job diagnostics"])


async def context(request, project_id, key):
    state = request.app.state
    if await state.projects.get(project_id) is None:
        raise HTTPException(404, "Project not found")
    job = await get_job(state, project_id, key)
    if job is None:
        raise HTTPException(404, "Job not found in this project")
    if job.id.startswith("scene:"):
        try:
            run = state.scenes.get(job.source_id)
        except SceneError:
            raise HTTPException(404, "Job not found in this project") from None
        if run.job_id:
            owner = await state.execution.get(run.job_id)
            if (
                owner is None
                or owner.project_id != project_id
                or owner.kind != "scene.generate"
                or getattr(owner.request, "request_id", None) != run.id
            ):
                raise HTTPException(404, "Job not found in this project")
            # A scene alias remains stable in detail navigation, but its core
            # supervisor owns diagnostics and the exported accepted request.
            job = core_job(owner)
    try:
        store = getattr(state, "job_observability", None)
        if store is None:
            store = JobObservability(state.execution.settings.data_dir)
    except ObservabilityError:
        raise HTTPException(503, "Job diagnostic storage is unavailable") from None
    return store, job


@router.get("/projects/{project_id}/work-jobs/{key}/logs", response_model=JobLogPage)
async def logs(
    project_id: str,
    key: str,
    request: Request,
    after: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=500),
    severity: Literal["debug", "info", "warning", "error"] | None = None,
    stream: Literal["stdout", "stderr"] | None = None,
    query: str | None = Query(default=None, max_length=200),
):
    store, job = await context(request, project_id, key)
    try:
        return await asyncio.to_thread(
            store.page,
            job.id,
            after=after,
            limit=limit,
            severity=severity,
            stream=stream,
            query=query,
        )
    except ObservabilityError:
        raise HTTPException(503, "Job diagnostic history is unavailable") from None


@router.get("/projects/{project_id}/work-jobs/{key}/failure", response_model=JobFailure | None)
async def failure(project_id: str, key: str, request: Request):
    store, job = await context(request, project_id, key)
    try:
        return await asyncio.to_thread(store.get_failure, job.id)
    except ObservabilityError:
        raise HTTPException(503, "Saved job failure diagnostics are unavailable") from None


@router.get("/projects/{project_id}/work-jobs/{key}/diagnostics.zip")
async def diagnostics(project_id: str, key: str, request: Request):
    store, job = await context(request, project_id, key)
    specification = job.model_dump(mode="json")
    specification["application_version"] = __version__
    try:
        bundle = await asyncio.to_thread(store.export, job.id, specification)
    except ObservabilityError:
        raise HTTPException(503, "Job diagnostic export is unavailable") from None
    return Response(
        bundle,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="job-{job.source_id}-diagnostics.zip"',
            "Cache-Control": "no-store",
        },
    )


@router.get("/job-diagnostics-settings", response_model=ObservabilityPolicy)
async def settings(request: Request):
    try:
        store = getattr(request.app.state, "job_observability", None)
        return (
            store.policy
            if store
            else JobObservability(request.app.state.execution.settings.data_dir).policy
        )
    except ObservabilityError:
        raise HTTPException(503, "Job diagnostic retention settings are unavailable") from None
