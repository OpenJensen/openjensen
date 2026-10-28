"""Local scene-generation API; all jobs stay bound to their source project."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from vla_platform.scenes.agent import SceneError
from vla_platform.scenes.contracts import SceneRequest, SceneRun

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
    try:
        return await request.app.state.scenes.create(
            project_id, payload, source, request.app.state.explorer
        )
    except (SceneError, ValueError) as exc:
        raise HTTPException(409, str(exc)) from None


@router.get("/scenes/{ident}", response_model=SceneRun)
async def scene(ident: str, request: Request):
    try:
        return request.app.state.scenes.get(ident)
    except SceneError as exc:
        raise HTTPException(404, str(exc)) from None


@router.post("/scenes/{ident}/cancel", response_model=SceneRun)
async def cancel(ident: str, request: Request):
    try:
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
        content_disposition_type="inline" if path.suffix == ".jpg" else "attachment",
    )
