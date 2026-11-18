"""Clean and augment datasets: pipeline runs, episode review and saving the result."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import Field

from vla_platform.lifecycle.contracts import StrictRecord
from vla_platform.pipelines.service import RECIPES, Pipelines

router = APIRouter(prefix="/api/v1", tags=["Dataset pipelines"])

DatasetId = Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]
EntryId = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")]


def service(request: Request) -> Pipelines:
    return request.app.state.pipelines


PipelinesDep = Annotated[Pipelines, Depends(service)]


async def project(request, ident):
    if await request.app.state.projects.get(ident) is None:
        raise HTTPException(404, "Project not found")


class PipelineCreate(StrictRecord):
    dataset_id: DatasetId
    recipe: Literal["check", "check_free", "augment"] = "check"
    copies: int = Field(default=2, ge=1, le=4, strict=True)
    name: str | None = Field(default=None, min_length=1, max_length=100)


class Review(StrictRecord):
    decisions: dict[EntryId, bool] = Field(max_length=5000)


class Save(StrictRecord):
    name: str = Field(min_length=1, max_length=100)
    augment_copies: int = Field(default=0, ge=0, le=4, strict=True)


@router.get("/projects/{project_id}/dataset-pipelines")
async def list_pipelines(project_id: str, request: Request, pipelines: PipelinesDep):
    await project(request, project_id)
    return {"pipelines": pipelines.records(project_id)}


@router.post("/projects/{project_id}/dataset-pipelines", status_code=201)
async def create_pipeline(
    project_id: str, payload: PipelineCreate, request: Request, pipelines: PipelinesDep
):
    """Start a one-click recipe: check episodes for mistakes, or make augmented copies."""
    await project(request, project_id)
    steps = [dict(step) for step in RECIPES[payload.recipe]]
    if payload.recipe == "augment":
        steps[1] = {"processor": "jitter", "params": {"copies": payload.copies}}
    return pipelines.create(
        project_id, steps=steps, name=payload.name, dataset_id=payload.dataset_id
    )


@router.get("/projects/{project_id}/dataset-pipelines/{ident}")
async def get_pipeline(project_id: str, ident: str, pipelines: PipelinesDep):
    return pipelines.get(ident, project_id)


@router.get("/projects/{project_id}/dataset-pipelines/{ident}/episodes")
async def pipeline_episodes(project_id: str, ident: str, pipelines: PipelinesDep):
    return pipelines.episodes(ident, project_id)


@router.get("/projects/{project_id}/dataset-pipelines/{ident}/steps/{index}/manifest")
async def step_manifest(project_id: str, ident: str, index: int, pipelines: PipelinesDep):
    """The full episode manifest one step wrote, for inspection."""
    _, entries = pipelines.manifest(ident, index, project_id)
    return {"entries": entries}


@router.get("/projects/{project_id}/dataset-pipelines/{ident}/media")
async def pipeline_media(
    project_id: str,
    ident: str,
    pipelines: PipelinesDep,
    path: Annotated[str, Query(min_length=1, max_length=300)],
):
    """A still frame or review strip that a step of this pipeline wrote."""
    pipelines.get(ident, project_id)
    return FileResponse(pipelines.media_path(ident, path), media_type="image/jpeg")


@router.put("/projects/{project_id}/dataset-pipelines/{ident}/review")
async def review_pipeline(project_id: str, ident: str, payload: Review, pipelines: PipelinesDep):
    return await pipelines.set_review(ident, project_id, payload.decisions)


@router.post("/projects/{project_id}/dataset-pipelines/{ident}/save", status_code=202)
async def save_pipeline(project_id: str, ident: str, payload: Save, pipelines: PipelinesDep):
    """Apply the review, optionally add augmented copies, and save a new dataset."""
    steps = [{"processor": "human_review"}]
    if payload.augment_copies:
        steps.append({"processor": "jitter", "params": {"copies": payload.augment_copies}})
    steps.append({"processor": "export_dataset", "params": {"name": payload.name}})
    return await pipelines.extend(ident, project_id, steps)


@router.post("/projects/{project_id}/dataset-pipelines/{ident}/continue", status_code=202)
async def continue_pipeline(project_id: str, ident: str, pipelines: PipelinesDep):
    return await pipelines.resume(ident, project_id)
