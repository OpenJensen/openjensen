from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
import pyarrow as pa
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from filelock import FileLock, Timeout
from starlette.middleware.trustedhost import TrustedHostMiddleware

from vla_platform import __version__
from vla_platform.capabilities import registry
from vla_platform.contracts import (
    Capability,
    EpisodePage,
    EpisodePreview,
    IntakeRequest,
    Job,
    Project,
    ProjectCreate,
)
from vla_platform.datasets.explore import DatasetExplorer, ExplorationError
from vla_platform.execution import Execution
from vla_platform.lifecycle.contracts import JobEvent, PolicyArtifact, PolicyRequest
from vla_platform.projects import Projects
from vla_platform.settings import Settings
from vla_platform.storage import Storage

LOCAL_ORIGINS = {
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://[::1]:8000",
}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        lock = FileLock(settings.data_dir / "owner.lock")
        try:
            lock.acquire(timeout=0)
        except Timeout as exc:
            raise RuntimeError("Another application already owns this workspace") from exc
        storage = Storage(settings.data_dir)
        execution = Execution(storage, settings)
        explorer = DatasetExplorer()
        initialized = False
        try:
            await storage.initialize()
            initialized = True
            await execution.reconcile()
            app.state.projects, app.state.execution = Projects(storage), execution
            app.state.explorer = explorer
            yield
        finally:
            try:
                if initialized:
                    await execution.close()
            finally:
                try:
                    try:
                        await explorer.close()
                    finally:
                        await storage.close()
                finally:
                    lock.release()

    app = FastAPI(
        title="Firebird local VLA application",
        version=__version__,
        lifespan=lifespan,
        # The web app owns /docs so the reference shares the product's design system.
        # Keep /openapi.json available as the runtime source of truth.
        docs_url=None,
        redoc_url=None,
    )
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=sorted(LOCAL_ORIGINS),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def local_access_boundary(request: Request, call_next):
        origin = request.headers.get("origin")
        same_origin = str(request.base_url).rstrip("/")
        if origin and origin not in LOCAL_ORIGINS and origin != same_origin:
            return JSONResponse(
                {"detail": "Origin is not allowed for the local application"}, status_code=403
            )
        return await call_next(request)

    def projects_service(request: Request) -> Projects:
        return request.app.state.projects

    def execution_service(request: Request) -> Execution:
        return request.app.state.execution

    ProjectsDep = Annotated[Projects, Depends(projects_service)]
    ExecutionDep = Annotated[Execution, Depends(execution_service)]

    @app.exception_handler(ExplorationError)
    async def exploration_error(_: Request, exc: ExplorationError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=exc.status_code)

    async def explore(request: Request, job: Job, **kwargs):
        explorer = request.app.state.explorer
        try:
            if "episode_index" in kwargs:
                return await explorer.preview(job, **kwargs)
            return await explorer.page(job, **kwargs)
        except ExplorationError:
            raise
        except TimeoutError as exc:
            raise HTTPException(504, "Dataset preview timed out; try again") from exc
        except httpx.HTTPError as exc:
            raise HTTPException(502, "Hugging Face could not serve this dataset preview") from exc
        except (ValueError, pa.ArrowException) as exc:
            raise HTTPException(422, "Dataset preview metadata or Parquet data is invalid") from exc

    @app.get("/api/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/api/v1/capabilities", response_model=list[Capability])
    async def capabilities(execution: ExecutionDep) -> list[Capability]:
        return [
            *registry(
                bool(execution.lifecycle.catalog.runtimes),
                any(
                    r.training_python and r.training_root
                    for r in execution.lifecycle.catalog.runtimes
                ),
            ),
            Capability(
                stage="Dataset",
                operation="dataset.inspect.local",
                status="untested" if settings.local_root else "planned",
                description=(
                    "Local metadata intake is configured; target evidence awaits CAP-001."
                    if settings.local_root
                    else "Local intake is disabled; configure FIREBIRD_LOCAL_DATA_ROOT."
                ),
            ),
        ]

    @app.get("/api/v1/projects", response_model=list[Project])
    async def list_projects(service: ProjectsDep) -> list[Project]:
        return await service.list()

    @app.post("/api/v1/projects", response_model=Project, status_code=201)
    async def create_project(payload: ProjectCreate, service: ProjectsDep) -> Project:
        return await service.create(payload.name)

    @app.get("/api/v1/projects/{project_id}/jobs", response_model=list[Job])
    async def list_jobs(
        project_id: str, projects: ProjectsDep, execution: ExecutionDep
    ) -> list[Job]:
        if await projects.get(project_id) is None:
            raise HTTPException(404, "Project not found")
        return await execution.list(project_id)

    @app.post("/api/v1/projects/{project_id}/intakes", response_model=Job, status_code=202)
    async def inspect_dataset(
        project_id: str,
        payload: IntakeRequest,
        projects: ProjectsDep,
        execution: ExecutionDep,
    ) -> Job:
        if await projects.get(project_id) is None:
            raise HTTPException(404, "Project not found")
        if payload.source == "local" and settings.local_root is None:
            raise HTTPException(422, "Local intake is disabled; configure FIREBIRD_LOCAL_DATA_ROOT")
        return await execution.submit(project_id, payload)

    @app.get("/api/v1/policy-options")
    async def policy_options(execution: ExecutionDep) -> dict:
        return {
            **execution.lifecycle.catalog.public(),
            "training_methods": [
                {
                    "id": "lora",
                    "label": "LoRA",
                    "description": "Train adapters over a floating base",
                },
                {
                    "id": "qlora",
                    "label": "QLoRA",
                    "description": "Train adapters over an NF4 base to reduce memory",
                },
            ],
            "default_training_method": "lora",
            "quantization_defaults": {
                "cuda": {"language": "Q4_0", "vision": None},
                "cpu": {"language": "Q8_0", "vision": None},
                "note": (
                    "Starting recipes from small target-specific pilots; "
                    "new policies still require evaluation."
                ),
            },
        }

    @app.post("/api/v1/projects/{project_id}/policy-jobs", response_model=Job, status_code=202)
    async def policy_job(
        project_id: str, payload: PolicyRequest, projects: ProjectsDep, execution: ExecutionDep
    ) -> Job:
        if await projects.get(project_id) is None:
            raise HTTPException(404, "Project not found")
        try:
            return await execution.submit(project_id, payload)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/v1/projects/{project_id}/artifacts", response_model=list[PolicyArtifact])
    async def artifacts(project_id: str, projects: ProjectsDep, execution: ExecutionDep):
        if await projects.get(project_id) is None:
            raise HTTPException(404, "Project not found")
        return await execution.lifecycle.artifacts(project_id)

    @app.get("/api/v1/projects/{project_id}/artifacts/{artifact_id}/download")
    async def download_artifact(project_id: str, artifact_id: str, execution: ExecutionDep):
        try:
            path = await execution.lifecycle.download(project_id, artifact_id)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return FileResponse(path, media_type="application/x-tar", filename=path.name)

    @app.get("/api/v1/jobs/{job_id}/events", response_model=list[JobEvent])
    async def job_events(job_id: str, execution: ExecutionDep, after: int = 0):
        if await execution.get(job_id) is None:
            raise HTTPException(404, "Job not found")
        return execution.lifecycle.events(job_id, after)

    @app.get("/api/v1/jobs/{job_id}", response_model=Job)
    async def get_job(job_id: str, execution: ExecutionDep) -> Job:
        job = await execution.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        return job

    @app.get("/api/v1/jobs/{job_id}/episodes", response_model=EpisodePage)
    async def list_episodes(
        job_id: str,
        request: Request,
        execution: ExecutionDep,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=24)] = 12,
    ) -> EpisodePage:
        job = await get_job(job_id, execution)
        return await explore(request, job, offset=offset, limit=limit)

    @app.get("/api/v1/jobs/{job_id}/episodes/{episode_index}", response_model=EpisodePreview)
    async def get_episode(
        job_id: str,
        episode_index: int,
        request: Request,
        execution: ExecutionDep,
    ) -> EpisodePreview:
        job = await get_job(job_id, execution)
        return await explore(request, job, episode_index=episode_index)

    @app.post("/api/v1/jobs/{job_id}/cancel", response_model=Job)
    async def cancel_job(job_id: str, execution: ExecutionDep) -> Job:
        job = await execution.cancel(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        return job

    if settings.static_dir:
        app.mount("/", StaticFiles(directory=settings.static_dir, html=True), name="web")
    return app
