"""Publish a dataset or model to Hugging Face, or add one from a link, as jobs."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import Field

from vla_platform.datasets.library import LibraryError
from vla_platform.huggingface_sharing import HuggingFaceSharing
from vla_platform.huggingface_transfer import TransferError
from vla_platform.lifecycle.contracts import StrictRecord
from vla_platform.workspace_jobs import WorkJob, share_job

router = APIRouter(prefix="/api/v1", tags=["Hugging Face sharing"])
Kind = Literal["dataset", "model"]


def service(request: Request) -> HuggingFaceSharing:
    return request.app.state.huggingface_sharing


SharingDep = Annotated[HuggingFaceSharing, Depends(service)]


class SharingAccount(StrictRecord):
    connected: bool
    username: str | None = None
    source: Literal["settings", "machine"] | None = None
    message: str | None = None


class Published(StrictRecord):
    repo_id: str
    url: str
    private: bool
    revision: str
    published_at: str


class Destination(StrictRecord):
    kind: Kind
    id: str
    name: str
    account: SharingAccount
    repo_id: str | None = None
    url: str | None = None
    private: bool = True
    active_job: str | None = None
    published: Published | None = None


class PublishRequest(StrictRecord):
    kind: Kind
    id: str = Field(min_length=1, max_length=300)
    repo_id: str | None = Field(default=None, max_length=200)
    # Private unless the person explicitly chooses otherwise.
    private: bool = True


class FetchRequest(StrictRecord):
    kind: Kind
    link: str = Field(min_length=1, max_length=500)


async def project(request, ident):
    if await request.app.state.projects.get(ident) is None:
        raise HTTPException(404, "Project not found")


def refused(exc):
    if isinstance(exc, (TransferError, LibraryError)):
        return HTTPException(exc.status, str(exc))
    return HTTPException(422, str(exc))


@router.get("/huggingface-sharing/account", response_model=SharingAccount)
async def account(sharing: SharingDep):
    """Whose account a publish goes to: the token in Settings, else this computer's login."""
    return await sharing.account()


@router.get("/projects/{project_id}/huggingface-sharing/destination", response_model=Destination)
async def destination(
    project_id: str,
    request: Request,
    sharing: SharingDep,
    kind: Kind,
    ident: Annotated[str, Query(alias="id", min_length=1, max_length=300)],
):
    """Where a dataset or model would be published, before anything is sent."""
    await project(request, project_id)
    try:
        value = await sharing.destination(project_id, kind, ident)
    except (TransferError, LibraryError) as exc:
        raise refused(exc) from None
    if value["published"]:
        value["published"] = {key: value["published"][key] for key in Published.model_fields}
    return value


@router.post(
    "/projects/{project_id}/huggingface-sharing/publish", response_model=WorkJob, status_code=202
)
async def publish(project_id: str, payload: PublishRequest, request: Request, sharing: SharingDep):
    """Start publishing a saved dataset or a model. The repository is private by default."""
    await project(request, project_id)
    try:
        record = await sharing.publish(
            project_id, payload.kind, payload.id, payload.repo_id, payload.private
        )
    except (TransferError, LibraryError) as exc:
        raise refused(exc) from None
    return share_job(record)


@router.post(
    "/projects/{project_id}/huggingface-sharing/fetch", response_model=WorkJob, status_code=202
)
async def fetch(project_id: str, payload: FetchRequest, request: Request, sharing: SharingDep):
    """Start adding a dataset or model from a Hugging Face link or ``owner/name``."""
    await project(request, project_id)
    try:
        record = await sharing.fetch(project_id, payload.kind, payload.link)
    except (TransferError, LibraryError) as exc:
        raise refused(exc) from None
    return share_job(record)
