from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute

from vla_platform.huggingface_connection import (
    CredentialError,
    HuggingFaceConnection,
    HuggingFaceStatus,
    HuggingFaceTokenInput,
)


class SecretInputRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def sanitized_handler(request: Request):
            try:
                response = await handler(request)
                response.headers["Cache-Control"] = "no-store"
                return response
            except RequestValidationError:
                # FastAPI's usual validation detail includes raw submitted input.
                raise HTTPException(422, "Enter a valid Hugging Face access token.") from None

        return sanitized_handler


router = APIRouter(
    prefix="/api/v1/huggingface-connection", tags=["Hugging Face"], route_class=SecretInputRoute
)


def connection_service(request: Request) -> HuggingFaceConnection:
    return request.app.state.huggingface_connection


ConnectionDep = Annotated[HuggingFaceConnection, Depends(connection_service)]


@router.get("", response_model=HuggingFaceStatus)
async def huggingface_status(service: ConnectionDep) -> HuggingFaceStatus:
    return service.status()


@router.put("", response_model=HuggingFaceStatus)
@router.post("", response_model=HuggingFaceStatus)
async def save_huggingface_token(
    payload: HuggingFaceTokenInput, service: ConnectionDep
) -> HuggingFaceStatus:
    try:
        return await service.save(payload.token)
    except CredentialError as exc:
        raise HTTPException(422, str(exc)) from None
    except OSError:
        raise HTTPException(500, "Could not save the Hugging Face credential securely.") from None


@router.delete("", response_model=HuggingFaceStatus)
async def delete_huggingface_token(service: ConnectionDep) -> HuggingFaceStatus:
    try:
        return await service.disconnect()
    except OSError:
        raise HTTPException(500, "Could not remove the saved Hugging Face credential.") from None
