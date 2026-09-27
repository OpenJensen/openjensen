from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from vla_platform.compute_settings import ComputeSettingsResponse, ComputeSettingsUpdate
from vla_platform.lifecycle.service import Lifecycle

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
