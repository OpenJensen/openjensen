"""Private, declarative compute configuration; checks never allocate resources."""

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import select

from vla_platform.compute_targets import (
    ComputeTarget,
    ComputeTargetList,
    ComputeTargets,
    PublicComputeTarget,
    TargetError,
)
from vla_platform.storage import jobs

router = APIRouter(prefix="/api/v1/compute-targets", tags=["Compute targets"])


def service(request):
    execution = request.app.state.execution
    if not hasattr(request.app.state, "compute_targets"):
        request.app.state.compute_targets = ComputeTargets(execution.settings.data_dir)
    return request.app.state.compute_targets, execution.lifecycle.catalog


@router.get("", response_model=ComputeTargetList)
async def targets(request: Request):
    registry, catalog = service(request)
    try:
        return registry.list(catalog)
    except TargetError as exc:
        raise HTTPException(exc.status, str(exc)) from None


@router.put("/{ident}", response_model=PublicComputeTarget)
async def save(ident: str, payload: ComputeTarget, request: Request):
    registry, catalog = service(request)
    if ident != payload.id:
        raise HTTPException(422, "Target identity does not match the requested address")
    async with request.app.state.execution.lock:
        await require_idle_target(ident, request)
        try:
            result = registry.public(registry.save(payload), catalog)
        except TargetError as exc:
            raise HTTPException(exc.status, str(exc)) from None
        request.app.state.execution.target_slots.pop(ident, None)
        return result


@router.post("/{ident}/check", response_model=PublicComputeTarget)
async def check(ident: str, request: Request):
    registry, catalog = service(request)
    try:
        return await registry.check(ident, catalog)
    except TargetError as exc:
        raise HTTPException(exc.status, str(exc)) from None


async def require_idle_target(ident, request):
    execution = request.app.state.execution
    async with execution.storage.engine.connect() as connection:
        records = (await connection.execute(select(jobs.c.record))).scalars()
        for record in records:
            binding = record.get("execution") or {}
            selected = (record.get("request") or {}).get("execution_target_id")
            if binding.get("target_id") != ident and selected != ident:
                continue
            cancellation = record.get("cancellation") or {}
            cleanup_pending = cancellation and (
                not cancellation.get("finished_at") or cancellation.get("cleanup") == "failed"
            )
            if record.get("status") in {"queued", "running"} or cleanup_pending:
                raise HTTPException(
                    409,
                    (
                        "Finish or cancel active jobs and reconcile cleanup "
                        "before changing this target"
                    ),
                )


@router.delete("/{ident}")
async def remove(ident: str, request: Request):
    registry, _ = service(request)
    execution = request.app.state.execution
    async with execution.lock:
        await require_idle_target(ident, request)
        try:
            registry.remove(ident)
        except TargetError as exc:
            raise HTTPException(exc.status, str(exc)) from None
        execution.target_slots.pop(ident, None)
    return {"removed": True}
