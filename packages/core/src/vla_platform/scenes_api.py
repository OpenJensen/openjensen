"""Local scene-generation API; all jobs stay bound to their source project."""

from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse

from vla_platform.contracts import Job
from vla_platform.scenes.agent import SceneError
from vla_platform.scenes.contracts import SceneRequest, SceneRun
from vla_platform.submissions import SubmissionConflict, validate_key

router = APIRouter(prefix="/api/v1", tags=["Scene generation"])


async def project(request, ident):
    if await request.app.state.projects.get(ident) is None:
        raise HTTPException(404, "Project not found")


@router.get("/scene-generation/options")
async def options(request: Request):
    return request.app.state.scenes.options()


@router.get("/projects/{project_id}/scenes", response_model=list[SceneRun])
async def scenes(project_id: str, request: Request):
    await project(request, project_id)
    return request.app.state.scenes.list(project_id)


@router.post("/projects/{project_id}/scenes", response_model=SceneRun, status_code=202)
async def create(project_id: str, payload: SceneRequest, request: Request):
    await project(request, project_id)
    source = await request.app.state.execution.get(payload.source_job_id)
    if source is None or source.project_id != project_id:
        raise HTTPException(404, "Dataset source not found in this project")
    raise HTTPException(409, "Start Real to sim as a job using this project's scene-jobs endpoint.")


@router.post(
    "/projects/{project_id}/scene-jobs",
    response_model=Job,
    status_code=202,
    responses={
        409: {"description": "Submission conflicts with an accepted request or runtime setup"},
        503: {
            "description": "Saved acceptance cannot be verified; no replacement work is submitted"
        },
    },
)
async def create_job(
    project_id: str,
    payload: SceneRequest,
    request: Request,
    response: Response,
    idempotency_key: Annotated[str | None, Header()] = None,
):
    await project(request, project_id)
    if len(request.headers.getlist("idempotency-key")) > 1:
        raise HTTPException(422, "Supply exactly one Idempotency-Key header")
    key = idempotency_key if idempotency_key is not None else payload.request_id
    try:
        validate_key(key)
        job = await request.app.state.execution.submit(project_id, payload, idempotency_key=key)
    except SubmissionConflict:
        raise
    except SceneError as exc:
        raise HTTPException(409, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    response.headers["Idempotency-Key"] = key
    response.headers["Cache-Control"] = "no-store"
    return job


async def bound_scene(job_id, request):
    job = await request.app.state.execution.get(job_id)
    if job is None or not isinstance(job.request, SceneRequest):
        raise HTTPException(404, "Scene job not found")
    try:
        run = request.app.state.scenes.get(job.request.request_id)
    except SceneError as exc:
        raise HTTPException(404, str(exc)) from None
    if run.job_id != job.id or run.project_id != job.project_id:
        raise HTTPException(404, "Scene job result not found")
    return run


@router.get("/jobs/{job_id}/scene", response_model=SceneRun)
async def job_scene(job_id: str, request: Request):
    return await bound_scene(job_id, request)


@router.get("/jobs/{job_id}/scene/artifact")
async def job_artifact(job_id: str, name: str, request: Request):
    run = await bound_scene(job_id, request)
    return await artifact(run.id, name, request)


@router.get("/scenes/{ident}", response_model=SceneRun)
async def scene(ident: str, request: Request):
    try:
        return request.app.state.scenes.get(ident)
    except SceneError as exc:
        raise HTTPException(404, str(exc)) from None


@router.post("/scenes/{ident}/cancel", response_model=SceneRun)
async def cancel(ident: str, request: Request):
    try:
        run = request.app.state.scenes.get(ident)
        if run.job_id:
            await bound_scene(run.job_id, request)
            await request.app.state.execution.cancel(run.job_id)
            return request.app.state.scenes.get(ident)
        return await request.app.state.scenes.cancel(ident)
    except SceneError as exc:
        raise HTTPException(404, str(exc)) from None


@router.get("/scenes/{ident}/artifact")
async def artifact(ident: str, name: str, request: Request):
    try:
        path = request.app.state.scenes.artifact(ident, name)
    except SceneError as exc:
        raise HTTPException(404, str(exc)) from None
    return FileResponse(
        path,
        filename=path.name,
        content_disposition_type="inline"
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".mp4"}
        else "attachment",
    )
