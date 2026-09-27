from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import Field

from vla_platform.compute_settings import ComputeSettingsResponse, ComputeSettingsUpdate
from vla_platform.lifecycle.contracts import StrictRecord
from vla_platform.lifecycle.runtime import PublicRuntime
from vla_platform.lifecycle.service import Lifecycle
from vla_platform.local_worker_discovery import (
    LocalWorkerDiscovery,
    LocalWorkerDiscoveryService,
)

router = APIRouter(prefix="/api/v1/compute-settings", tags=["Compute settings"])


def lifecycle_service(request: Request) -> Lifecycle:
    return request.app.state.execution.lifecycle


LifecycleDep = Annotated[Lifecycle, Depends(lifecycle_service)]


@router.get("", response_model=ComputeSettingsResponse)
async def get_compute_settings(lifecycle: LifecycleDep) -> ComputeSettingsResponse:
    return lifecycle.compute.public(lifecycle.catalog)


@router.put("", response_model=ComputeSettingsResponse)
async def update_compute_settings(
    payload: ComputeSettingsUpdate, lifecycle: LifecycleDep
) -> ComputeSettingsResponse:
    try:
        lifecycle.compute.update(payload)
    except OSError as exc:
        raise HTTPException(500, "Could not save compute settings") from exc
    return lifecycle.compute.public(lifecycle.catalog)


@router.post("/gcp/check", response_model=ComputeSettingsResponse)
async def check_gcp_compute(lifecycle: LifecycleDep) -> ComputeSettingsResponse:
    await lifecycle.compute.check_gcp()
    return lifecycle.compute.public(lifecycle.catalog)


@router.post("/gcp/prepare", response_model=ComputeSettingsResponse)
async def prepare_gcp_compute(lifecycle: LifecycleDep) -> ComputeSettingsResponse:
    await lifecycle.compute.prepare_gcp()
    return lifecycle.compute.public(lifecycle.catalog)


class LocalWorkerAddRequest(StrictRecord):
    candidate_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")


class LocalWorkerAddResponse(StrictRecord):
    runtime: PublicRuntime
    compute: ComputeSettingsResponse
    discovery: LocalWorkerDiscovery


async def local_worker_discovery(lifecycle: LifecycleDep) -> LocalWorkerDiscoveryService:
    # No await between lookup and assignment: one service/lock per application owner.
    if not hasattr(lifecycle, "local_worker_discovery"):
        lifecycle.local_worker_discovery = LocalWorkerDiscoveryService(lifecycle)
    return lifecycle.local_worker_discovery


LocalDiscoveryDep = Annotated[LocalWorkerDiscoveryService, Depends(local_worker_discovery)]


@router.post("/local/check", response_model=LocalWorkerDiscovery)
async def check_local_workers(discovery: LocalDiscoveryDep) -> LocalWorkerDiscovery:
    return await discovery.check()


@router.post("/local/workers", response_model=LocalWorkerAddResponse)
async def add_local_worker(
    payload: LocalWorkerAddRequest,
    discovery: LocalDiscoveryDep,
    lifecycle: LifecycleDep,
) -> LocalWorkerAddResponse:
    try:
        runtime = await discovery.add(payload.candidate_id)
    except ValueError as exc:
        raise HTTPException(
            409, "The worker could not be added. Check this machine again and review its status."
        ) from exc
    except OSError as exc:
        raise HTTPException(500, "Could not save the local worker. Check and retry.") from exc
    compute = lifecycle.compute.public(lifecycle.catalog)
    public = next((item for item in compute.runtimes if item.id == runtime.id), None)
    if public is None or discovery.last_result is None:
        raise HTTPException(500, "Worker registration could not be confirmed. Check again.")
    return LocalWorkerAddResponse(runtime=public, compute=compute, discovery=discovery.last_result)
