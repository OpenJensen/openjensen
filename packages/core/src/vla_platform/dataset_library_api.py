"""Browser folder import and project-owned dataset library."""

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import Field
from starlette.background import BackgroundTask

from vla_platform.datasets.library import DatasetLibrary, LibraryError, contained
from vla_platform.lifecycle.contracts import StrictRecord

router = APIRouter(prefix="/api/v1", tags=["Dataset library"])


def service(request: Request) -> DatasetLibrary:
    return request.app.state.dataset_library


LibraryDep = Annotated[DatasetLibrary, Depends(service)]


class UploadCreate(StrictRecord):
    name: str = Field(min_length=1, max_length=100)


class Conversion(StrictRecord):
    fps: int = Field(default=30, ge=1, le=240, strict=True)
    task: str = Field(default="Recorded robot task", min_length=1, max_length=1000)
    robot_type: str = Field(default="unspecified", min_length=1, max_length=100)
    mapping: dict | None = None


class Labels(StrictRecord):
    revision: int = Field(ge=0, strict=True)
    views: dict[str, Annotated[str, Field(max_length=100)]] = Field(
        default_factory=dict, max_length=8
    )
    frames: dict[str, Annotated[str, Field(max_length=1000)]] = Field(
        default_factory=dict, max_length=12000
    )


async def project(request, ident):
    if await request.app.state.projects.get(ident) is None:
        raise HTTPException(404, "Project not found")


@router.get("/datasets")
async def datasets(request: Request, library: LibraryDep, project_id: str | None = None):
    if project_id:
        await project(request, project_id)
    return await library.list(request.app.state.projects, request.app.state.execution, project_id)


@router.post("/projects/{project_id}/dataset-uploads", status_code=201)
async def begin_upload(
    project_id: str, payload: UploadCreate, request: Request, library: LibraryDep
):
    await project(request, project_id)
    return library.begin(project_id, payload.name)


@router.put("/dataset-uploads/{ident}/file")
async def upload_file(
    ident: str,
    request: Request,
    library: LibraryDep,
    path: Annotated[str, Query(min_length=1, max_length=1024)],
):
    return await library.upload(ident, path, request.stream())


@router.post("/dataset-uploads/{ident}/finish")
async def finish_upload(ident: str, library: LibraryDep):
    return await library.finalize(ident)


@router.post("/projects/{project_id}/datasets/example", status_code=202)
async def example(project_id: str, request: Request, library: LibraryDep):
    await project(request, project_id)
    return await library.example(project_id)


@router.get("/datasets/{ident}")
async def get_dataset(ident: str, library: LibraryDep):
    return library.public(library.get(ident))


@router.post("/datasets/{ident}/convert", status_code=202)
async def convert_dataset(ident: str, payload: Conversion, library: LibraryDep):
    return await library.start_conversion(ident, payload.model_dump())


@router.get("/datasets/{ident}/samples")
async def samples(ident: str, library: LibraryDep):
    library.get(ident)
    return library.samples(ident)


@router.get("/datasets/{ident}/preview/{name}")
async def preview(ident: str, name: str, library: LibraryDep):
    library.get(ident)
    if name not in {r["path"] for r in library.samples(ident)}:
        raise LibraryError("Sample image not found", 404)
    path = contained(library.directory(ident) / "previews", name)
    return FileResponse(path, media_type="image/jpeg")


@router.put("/datasets/{ident}/annotations")
async def annotations(ident: str, payload: Labels, library: LibraryDep):
    return await asyncio.to_thread(library.annotate, ident, payload.model_dump())


@router.get("/datasets/{ident}/download")
async def download(ident: str, library: LibraryDep):
    path = await asyncio.to_thread(library.export, ident)
    return FileResponse(
        path,
        media_type="application/zip",
        filename="lerobot-dataset.zip",
        background=BackgroundTask(path.unlink, missing_ok=True),
    )
