"""Private server administration; the public gateway never forwards these routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select

from vla_platform.application_updates import (
    ApplicationUpdates,
    UpdateError,
    UpdateRequest,
    UpdateStatus,
)
from vla_platform.storage import jobs

router = APIRouter(prefix="/api/v1/updates", tags=["Application updates"])


def service(request: Request) -> ApplicationUpdates:
    return request.app.state.application_updates


UpdatesDep = Annotated[ApplicationUpdates, Depends(service)]


@router.get("", response_model=UpdateStatus)
async def status(updates: UpdatesDep):
    return updates.status()


@router.get("/check", response_model=UpdateStatus)
async def check(updates: UpdatesDep, refresh: bool = False):
    return await updates.check(force=refresh)


async def require_idle(request):
    execution = request.app.state.execution
    async with execution.storage.engine.connect() as connection:
        active = (
            await connection.execute(
                select(jobs.c.id).where(jobs.c.status.in_(["queued", "running"])).limit(1)
            )
        ).first()
    setup = getattr(execution.lifecycle, "local_worker_setup", None)
    if (
        getattr(getattr(request.app.state, "agent_control", None), "pending", False)
        or request.app.state.active_writes
        or getattr(request.app.state, "module_install_reserving", False)
        or getattr(request.app.state, "module_installations", None)
        and request.app.state.module_installations.pending
        or active
        or any(not task.done() for task in execution.tasks.values())
        or any(not task.done() for task in execution._accepting)
        or execution.cloud_cleanup_pending
        or setup
        and setup.state.status == "running"
    ):
        raise UpdateError("Finish or cancel active jobs before updating this server.")
    robot = request.app.state.robot_lab
    if robot.config_path.exists():
        try:
            value = await robot.request("GET", "status")
            session = value["session"]
            if session["mode"] != "idle" or session["torque_enabled"]:
                raise ValueError
        except Exception as exc:
            raise UpdateError(
                "Stop Robot Lab and release the motors before updating this server."
            ) from exc


@router.get("/readiness")
async def readiness(request: Request):
    try:
        await require_idle(request)
    except UpdateError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    return {"idle": True}


@router.post("/apply", response_model=UpdateStatus, status_code=202)
async def apply(payload: UpdateRequest, request: Request, updates: UpdatesDep):
    try:
        previous = updates.operation()
        if payload.operation_id and previous and previous.id == payload.operation_id:
            if previous.target_commit != payload.target_commit:
                raise UpdateError("This confirmation already belongs to a different update.")
            return updates.status()
        # Refresh the comparison before accepting the exact revision the user reviewed.
        if updates.pending:
            raise UpdateError("An update is already in progress.")
        updates.configuration()
        await updates.check(force=True)
        if updates.pending:
            raise UpdateError("An update is already in progress.")
        updates.reserving = True
        try:
            await require_idle(request)
            return await updates.start(payload)
        finally:
            updates.reserving = False
    except UpdateError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
