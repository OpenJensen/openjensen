"""Manage installed API integrations without returning keys or contacting providers."""

import asyncio

from fastapi import APIRouter, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute

from vla_platform.api_connections import (
    ApiConnections,
    ApiConnectionsStatus,
    ApiConnectionStatus,
    ApiKeyError,
    ApiKeyInput,
    ApiProvider,
)


class SecretInputRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def safe_handler(request: Request):
            try:
                response = await handler(request)
                response.headers["Cache-Control"] = "no-store"
                return response
            except RequestValidationError:
                raise HTTPException(
                    422,
                    "Enter a valid API key and provider.",
                    headers={"Cache-Control": "no-store"},
                ) from None
            except HTTPException as exc:
                exc.headers = {**(exc.headers or {}), "Cache-Control": "no-store"}
                raise

        return safe_handler


router = APIRouter(
    prefix="/api/v1/api-connections", tags=["Cloud and APIs"], route_class=SecretInputRoute
)


def service(request: Request) -> ApiConnections:
    return ApiConnections(request.app.state.execution.settings.data_dir)


def require_installed(request: Request, ident: ApiProvider):
    from vla_platform.modules import module_installed

    try:
        installed = module_installed(request.app.state.execution.settings.data_dir, ident)
    except ValueError:
        raise HTTPException(
            409, "The saved module registry needs administrator attention."
        ) from None
    if not installed:
        raise HTTPException(
            409, "Install this integration in Modules before configuring its API key."
        )


@router.get("", response_model=ApiConnectionsStatus)
async def connections(request: Request) -> ApiConnectionsStatus:
    return await asyncio.to_thread(service(request).status)


@router.put("/{ident}", response_model=ApiConnectionStatus)
async def save(ident: ApiProvider, payload: ApiKeyInput, request: Request) -> ApiConnectionStatus:
    require_installed(request, ident)
    try:
        return await asyncio.to_thread(service(request).save, ident, payload.api_key)
    except ApiKeyError as exc:
        raise HTTPException(422, str(exc)) from None
    except OSError:
        raise HTTPException(500, "The API key could not be saved securely.") from None


@router.delete("/{ident}", response_model=ApiConnectionStatus)
async def disconnect(ident: ApiProvider, request: Request) -> ApiConnectionStatus:
    require_installed(request, ident)
    try:
        return await asyncio.to_thread(service(request).disconnect, ident)
    except ApiKeyError as exc:
        raise HTTPException(409, str(exc)) from None
    except OSError:
        raise HTTPException(500, "The saved API key could not be removed securely.") from None
