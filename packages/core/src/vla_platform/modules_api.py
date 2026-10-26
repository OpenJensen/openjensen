"""Private module management and bounded, allowlisted dependency installation."""

import asyncio

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import select

from vla_platform.module_installations import (
    InstallOperation,
    InstallRequest,
    ModuleInstallationStatus,
)
from vla_platform.modules import (
    ModuleId,
    ModulePreference,
    set_module_enabled,
    workspace_modules,
)
from vla_platform.storage import jobs

router = APIRouter(prefix="/api/v1", tags=["Workspace modules"])


@router.put("/modules/{ident}")
async def enable(ident: ModuleId, payload: ModulePreference, request: Request):
    try:
        if payload.enabled:
            settings = request.app.state.execution.settings
            modules = await asyncio.to_thread(
                lambda: workspace_modules(
                    settings,
                    request.app.state.robot_lab.config_path,
                    {
                        key: request.app.state.module_installations.inspect(key).ready
                        for key in ("robot", "decision")
                    },
                )
            )
            module = next(item for item in modules if item.id == ident)
            if not module.available:
                raise ValueError(module.setup)
        await asyncio.to_thread(
            set_module_enabled,
            request.app.state.execution.settings.data_dir,
            ident,
            payload.enabled,
        )
    except (ValueError, OSError) as exc:
        raise HTTPException(
            409,
            str(exc) if isinstance(exc, ValueError) else "Module preferences could not be saved.",
        ) from None
    return {"enabled": payload.enabled}


@router.get("/module-installations", response_model=ModuleInstallationStatus)
async def status(request: Request):
    return await request.app.state.module_installations.status(request.app.state.robot_lab)


@router.post("/module-installations", response_model=InstallOperation, status_code=202)
async def install(payload: InstallRequest, request: Request):
    if getattr(request.app.state, "module_install_reserving", False):
        raise HTTPException(409, "A module installation is being prepared. Try again shortly.")
    request.app.state.module_install_reserving = True
    try:
        execution = request.app.state.execution
        async with execution.storage.engine.connect() as connection:
            active = (
                await connection.execute(
                    select(jobs.c.id).where(jobs.c.status.in_(["queued", "running"])).limit(1)
                )
            ).first()
        setup = getattr(execution.lifecycle, "local_worker_setup", None)
        if active or setup and (setup.state.status == "running" or setup.lock.locked()):
            raise HTTPException(
                409, "Finish active jobs and training setup before installing dependencies."
            )
        return await request.app.state.module_installations.start(payload)
    except (ValueError, OSError) as exc:
        raise HTTPException(
            409,
            str(exc)
            if isinstance(exc, ValueError)
            else "The installer could not start on this server.",
        ) from None
    finally:
        request.app.state.module_install_reserving = False


@router.post("/module-installations/cancel", response_model=InstallOperation | None)
async def cancel(request: Request):
    try:
        return await request.app.state.module_installations.cancel()
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None
