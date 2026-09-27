"""Read-only, project-scoped catalog of operator-published local captures."""

import asyncio

from fastapi import APIRouter, HTTPException, Request, Response

from vla_platform.datasets import recordings
from vla_platform.datasets.recording_contracts import RecordingCatalog, RecordingOptions

router = APIRouter()


async def context(request, project_id):
    if await request.app.state.projects.get(project_id) is None:
        raise HTTPException(404, "Project not found")
    return request.app.state.execution.settings.recording_config


@router.get("/api/v1/projects/{project_id}/recordings/options", response_model=RecordingOptions)
async def recording_options(project_id: str, request: Request, response: Response):
    config = await context(request, project_id)
    response.headers["Cache-Control"] = "no-store"
    return await asyncio.to_thread(recordings.options, config, project_id)


@router.get("/api/v1/projects/{project_id}/recordings", response_model=RecordingCatalog)
async def recording_catalog(project_id: str, request: Request, response: Response):
    config = await context(request, project_id)
    response.headers["Cache-Control"] = "no-store"
    try:
        loaded = await asyncio.to_thread(recordings.load_config, config)
        return await asyncio.to_thread(recordings.catalog, loaded, project_id)
    except recordings.RecordingError as error:
        raise HTTPException(error.status, str(error)) from error
    except (OSError, ValueError) as error:
        raise HTTPException(
            422, "Recording catalog is unavailable or invalid; review operator setup"
        ) from error
