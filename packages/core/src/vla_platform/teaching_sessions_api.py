"""Project-owned local teaching routes. Existing manual relay routes are unchanged."""

import re
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict

from vla_platform.contracts import Job
from vla_platform.submissions import IdempotencyKey
from vla_platform.teaching_api import IDENTITY, TeachingCommand
from vla_platform.teaching_sessions.contracts import TeachingCaptureRequest

router = APIRouter(prefix="/api/v1/projects/{project_id}/teaching", tags=["Managed teaching"])


class SessionStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    job: Job
    ready: bool
    session_id: str | None
    stop_requested: bool


async def context(request: Request, project_id: str, response: Response):
    response.headers["Cache-Control"] = "no-store"
    if await request.app.state.projects.get(project_id) is None:
        raise HTTPException(404, "Project not found")
    return request.app.state.execution


@router.get("/profiles")
async def profiles(project_id: str, request: Request, response: Response):
    execution = await context(request, project_id, response)
    return await execution.teaching.profiles(project_id)


@router.post("/sessions", response_model=Job, status_code=202)
async def start(
    project_id: str,
    payload: TeachingCaptureRequest,
    request: Request,
    response: Response,
    idempotency_key: Annotated[IdempotencyKey | None, Header()] = None,
):
    execution = await context(request, project_id, response)
    if len(request.headers.getlist("idempotency-key")) > 1:
        raise HTTPException(422, "Supply exactly one Idempotency-Key header")
    job = await execution.submit(project_id, payload, idempotency_key=idempotency_key)
    if idempotency_key is not None:
        response.headers["Idempotency-Key"] = idempotency_key
    return job


@router.get("/sessions/{job_id}", response_model=SessionStatus)
async def status(project_id: str, job_id: str, request: Request, response: Response):
    execution = await context(request, project_id, response)
    return await execution.teaching.status(project_id, job_id)


@router.post("/sessions/{job_id}/stop", response_model=SessionStatus)
async def stop(project_id: str, job_id: str, request: Request, response: Response):
    execution = await context(request, project_id, response)
    return await execution.teaching.stop(project_id, job_id)


@router.get("/sessions/{job_id}/state")
async def state(project_id: str, job_id: str, request: Request, response: Response):
    execution = await context(request, project_id, response)
    return await execution.teaching.relay(project_id, job_id, "/state")


@router.get("/sessions/{job_id}/frame")
async def frame(project_id: str, job_id: str, request: Request, response: Response):
    execution = await context(request, project_id, response)
    return await execution.teaching.relay(project_id, job_id, "/frame")


@router.post("/sessions/{job_id}/commands", status_code=202)
async def command(
    project_id: str, job_id: str, payload: TeachingCommand, request: Request, response: Response
):
    execution = await context(request, project_id, response)
    return await execution.teaching.relay(project_id, job_id, "/commands", payload)


@router.get("/sessions/{job_id}/commands/{command_id}")
async def receipt(
    project_id: str, job_id: str, command_id: str, request: Request, response: Response
):
    execution = await context(request, project_id, response)
    if not re.fullmatch(IDENTITY, command_id):
        raise HTTPException(422, "Invalid command identity")
    return await execution.teaching.relay(project_id, job_id, "/commands/" + command_id)
