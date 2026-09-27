from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from vla_platform.cloud_connections import (
    CloudConnection,
    CloudConnections,
    CloudConnectionsResponse,
    ConnectionConfig,
    Provider,
)

router = APIRouter(prefix="/api/v1/cloud-connections", tags=["Cloud connections"])


def cloud_connections_service(request: Request) -> CloudConnections:
    return request.app.state.cloud_connections


CloudConnectionsDep = Annotated[CloudConnections, Depends(cloud_connections_service)]


@router.get("", response_model=CloudConnectionsResponse)
async def list_cloud_connections(service: CloudConnectionsDep) -> CloudConnectionsResponse:
    return service.list()


@router.post("/{provider}/connect", response_model=CloudConnection)
async def connect_cloud_provider(
    provider: Provider, payload: ConnectionConfig, service: CloudConnectionsDep, request: Request
) -> CloudConnection:
    try:
        with request.app.state.execution.lifecycle.compute.cloud_connection_update():
            result = await service.connect(provider, payload)
            if result.status == "connected":
                request.app.state.execution.lifecycle.compute.enable_cloud()
            return result
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(
            500, "Connected to Google Cloud but could not save training settings"
        ) from exc


@router.post("/{provider}/recheck", response_model=CloudConnection)
async def recheck_cloud_provider(
    provider: Provider, service: CloudConnectionsDep, request: Request
) -> CloudConnection:
    with request.app.state.execution.lifecycle.compute.cloud_connection_update():
        return await service.recheck(provider)


@router.post("/{provider}/disconnect", response_model=CloudConnection)
async def disconnect_cloud_provider(
    provider: Provider, service: CloudConnectionsDep, request: Request
) -> CloudConnection:
    try:
        with request.app.state.execution.lifecycle.compute.cloud_connection_update():
            return await service.disconnect(provider)
    except OSError as exc:
        raise HTTPException(500, "Could not remove the saved cloud connection") from exc
