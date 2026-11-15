"""Move projects in from other workspaces on this computer, and remove projects."""

import asyncio

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from vla_platform import project_transfer
from vla_platform.contracts import Project

router = APIRouter(prefix="/api/v1", tags=["Projects"])


class NearbyProject(BaseModel):
    id: str
    name: str
    created_at: str
    datasets: int
    jobs: int


class NearbyWorkspace(BaseModel):
    path: str
    label: str
    available: bool
    message: str | None
    projects: list[NearbyProject]


class ProjectMove(BaseModel):
    model_config = ConfigDict(extra="forbid")
    workspace: str = Field(min_length=1, max_length=4096)
    project_id: str = Field(min_length=1, max_length=200)


class ProjectRemoval(BaseModel):
    id: str
    name: str
    datasets: int
    jobs: int


def workspace(request: Request):
    return request.app.state.execution.settings.data_dir


@router.get("/projects/nearby", response_model=list[NearbyWorkspace])
async def nearby(request: Request):
    """Other OpenJensen workspaces on this computer, with the projects each holds."""
    return await asyncio.to_thread(project_transfer.nearby, workspace(request))


@router.post("/projects/move", response_model=Project)
async def move(payload: ProjectMove, request: Request):
    """Move a project from another workspace on this computer into this one."""
    here = workspace(request)
    listed = await asyncio.to_thread(project_transfer.nearby, here)
    if payload.workspace not in {item["path"] for item in listed}:
        raise HTTPException(404, "That workspace is not one of the workspaces on this computer.")
    try:
        moved = await asyncio.to_thread(
            project_transfer.move, payload.workspace, here, payload.project_id
        )
    except project_transfer.TransferError as error:
        raise HTTPException(error.status, str(error)) from None
    project = await request.app.state.projects.get(moved["id"])
    if project is None:
        raise HTTPException(500, "The project moved but could not be read back.")
    return project


@router.delete("/projects/{project_id}", response_model=ProjectRemoval)
async def remove(project_id: str, request: Request):
    """Delete a project and its datasets, jobs and recordings from this workspace."""
    if await request.app.state.projects.get(project_id) is None:
        raise HTTPException(404, "Project not found")
    try:
        return await asyncio.to_thread(project_transfer.remove, workspace(request), project_id)
    except project_transfer.TransferError as error:
        raise HTTPException(error.status, str(error)) from None
