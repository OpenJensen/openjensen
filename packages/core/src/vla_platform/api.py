import asyncio
import hashlib
import json
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import uuid4

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from filelock import FileLock, Timeout
from starlette.background import BackgroundTask
from starlette.middleware.trustedhost import TrustedHostMiddleware

from vla_platform import __version__
from vla_platform.augmentation.contracts import AugmentationOptions, AugmentationRequest
from vla_platform.capabilities import registry
from vla_platform.cloud_api import router as cloud_connections_router
from vla_platform.cloud_connections import CloudConnections
from vla_platform.cloud_runs import CloudRunsFeed, read_cloud_runs
from vla_platform.compute_api import router as compute_settings_router
from vla_platform.contracts import (
    Capability,
    EpisodePage,
    EpisodePreview,
    IntakeRequest,
    Job,
    Project,
    ProjectCreate,
)
from vla_platform.dataset_library_api import router as dataset_library_router
from vla_platform.datasets.explore import DatasetExplorer, ExplorationError
from vla_platform.datasets.library import DatasetLibrary, LibraryError
from vla_platform.datasets.recordings import RecordingError
from vla_platform.decision_api import router as decision_router
from vla_platform.execution import Execution
from vla_platform.huggingface_api import router as huggingface_router
from vla_platform.huggingface_connection import HuggingFaceConnection
from vla_platform.lifecycle import telemetry
from vla_platform.lifecycle.contracts import (
    JobEvent,
    PolicyArtifact,
    PolicyRequest,
    TrainingTelemetry,
)
from vla_platform.lifecycle.training_catalog import public_training_models
from vla_platform.projects import Projects
from vla_platform.recordings_api import router as recordings_router
from vla_platform.settings import Settings
from vla_platform.storage import Storage
from vla_platform.submissions import (
    IdempotencyKey,
    SubmissionConflict,
    SubmissionOperation,
    SubmissionUnavailable,
)
from vla_platform.submissions import (
    lookup as submission_lookup,
)
from vla_platform.teaching_api import router as teaching_router
from vla_platform.teaching_sessions.config import TeachingError
from vla_platform.teaching_sessions_api import router as teaching_sessions_router

LOCAL_ORIGINS = {
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://[::1]:8000",
}


def submission_key(
    request: Request,
    idempotency_key: Annotated[
        IdempotencyKey | None,
        Header(
            description=(
                "Optional durable key scoped to project and operation. Identical requests "
                "return the original job's current state, including terminal/interrupted jobs; "
                "changed "
                "requests return 409. Preserve the key before sending. No automatic retries occur."
            )
        ),
    ] = None,
) -> str | None:
    """An optional opaque client key; duplicate headers are never silently selected."""
    if len(request.headers.getlist("idempotency-key")) > 1:
        raise HTTPException(422, "Supply exactly one Idempotency-Key header")
    return idempotency_key


SubmissionKeyDep = Annotated[str | None, Depends(submission_key)]
SUBMISSION_RESPONSES = {
    409: {"description": "The key already identifies a different normalized request"},
    503: {"description": "Saved acceptance cannot be verified; no replacement work is submitted"},
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
            app.state.dataset_library = DatasetLibrary(settings.data_dir)
            app.state.cloud_connections = CloudConnections(settings.data_dir)
            app.state.huggingface_connection = HuggingFaceConnection(settings.data_dir)
            yield
        finally:
            try:
                if initialized:
                    try:
                        if hasattr(app.state, "dataset_library"):
                            await app.state.dataset_library.close()
                        if hasattr(execution.lifecycle, "local_worker_setup"):
                            await execution.lifecycle.local_worker_setup.close()
                    finally:
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
        title="OPEN JENSEN local VLA application",
        version=__version__,
        lifespan=lifespan,
        # The web app owns /docs so the reference shares the product's design system.
        # Keep /openapi.json available as the runtime source of truth.
        docs_url=None,
        redoc_url=None,
    )

    @app.exception_handler(TeachingError)
    async def teaching_error(_request: Request, exc: TeachingError):
        return JSONResponse({"detail": str(exc)}, status_code=exc.status)

    @app.exception_handler(SubmissionConflict)
    async def submission_conflict(_request: Request, exc: SubmissionConflict):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(SubmissionUnavailable)
    async def submission_unavailable(_request: Request, exc: SubmissionUnavailable):
        return JSONResponse({"detail": str(exc)}, status_code=503)

    app.include_router(cloud_connections_router)
    app.include_router(compute_settings_router)
    app.include_router(huggingface_router)
    app.include_router(teaching_router)
    app.include_router(teaching_sessions_router)
    app.include_router(decision_router)
    app.include_router(recordings_router)
    app.include_router(dataset_library_router)

    @app.exception_handler(LibraryError)
    async def dataset_library_error(_request: Request, exc: LibraryError):
        return JSONResponse({"detail": str(exc)}, status_code=exc.status)

    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=sorted(LOCAL_ORIGINS),
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Content-Type", "Idempotency-Key"],
        expose_headers=["Idempotency-Key"],
    )

    @app.middleware("http")
    async def local_access_boundary(request: Request, call_next):
        origin = request.headers.get("origin")
        same_origin = str(request.base_url).rstrip("/")
        if (origin and origin not in LOCAL_ORIGINS and origin != same_origin) or (
            not origin and request.headers.get("sec-fetch-site") == "cross-site"
        ):
            response = JSONResponse(
                {"detail": "Origin is not allowed for the local application"}, status_code=403
            )
        else:
            response = await call_next(request)
        if request.url.path.startswith("/api/v1/"):
            # Include handled errors: a cached failure must not hide a later job state.
            response.headers["Cache-Control"] = "no-store"
        return response

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
        except ValueError as exc:
            raise HTTPException(422, "Dataset preview metadata or Parquet data is invalid") from exc

    @app.get("/api/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/api/v1/cloud-runs", response_model=CloudRunsFeed)
    async def cloud_runs(response: Response) -> CloudRunsFeed:
        """Read bounded operator snapshots; never connect to or control a cloud host."""
        response.headers["Cache-Control"] = "no-store"
        return await asyncio.to_thread(read_cloud_runs, settings.cloud_runs_dir)

    @app.get("/api/v1/capabilities", response_model=list[Capability])
    async def capabilities(execution: ExecutionDep) -> list[Capability]:
        return [
            *registry(
                native_configured=any(
                    not (
                        r.export_only
                        or r.native_quantization_only
                        or r.native_distillation_only
                        or r.native_replay_only
                        or r.training_only
                    )
                    for r in execution.lifecycle.catalog.runtimes
                ),
                training_configured=any(
                    r.training_python and r.training_root
                    for r in execution.lifecycle.catalog.runtimes
                ),
                act_export_configured=any(
                    r.act_export_python and r.act_export_root
                    for r in execution.lifecycle.catalog.runtimes
                ),
                native_quantization_configured=any(
                    item["native_quantization"]
                    for item in execution.lifecycle.catalog.public()["runtimes"]
                ),
                native_replay_configured=any(
                    item["native_replay"]
                    for item in execution.lifecycle.catalog.public()["runtimes"]
                ),
                native_distillation_configured=any(
                    item["native_distillation"]
                    for item in execution.lifecycle.catalog.public()["runtimes"]
                ),
                cloud_configured=bool(execution.lifecycle.compute.cloud_runtimes()),
            ),
            Capability(
                stage="Dataset",
                operation="dataset.augment",
                status="untested" if execution.augmentation.options().configured else "planned",
                description=(
                    "Gemini Omni appearance edits for selected camera clips; review required."
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

    @app.post(
        "/api/v1/projects/{project_id}/intakes",
        response_model=Job,
        status_code=202,
        responses=SUBMISSION_RESPONSES,
    )
    async def inspect_dataset(
        project_id: str,
        payload: IntakeRequest,
        projects: ProjectsDep,
        execution: ExecutionDep,
        response: Response,
        idempotency_key: SubmissionKeyDep,
    ) -> Job:
        if await projects.get(project_id) is None:
            raise HTTPException(404, "Project not found")
        if (
            idempotency_key is None
            and payload.source == "local"
            and payload.recordings is None
            and payload.library_id is None
            and settings.local_root is None
        ):
            raise HTTPException(422, "Local intake is disabled; configure FIREBIRD_LOCAL_DATA_ROOT")
        try:
            job = await execution.inspections.submit(
                project_id, payload, idempotency_key=idempotency_key
            )
            if idempotency_key is not None:
                response.headers["Idempotency-Key"] = idempotency_key
                response.headers["Cache-Control"] = "no-store"
            return job
        except SubmissionConflict:
            raise
        except RecordingError as exc:
            raise HTTPException(exc.status, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise HTTPException(
                504, "Checking the latest dataset revision timed out; try again"
            ) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(
                502, "Hugging Face could not resolve this dataset revision"
            ) from exc

    @app.get("/api/v1/augmentation-options", response_model=AugmentationOptions)
    async def augmentation_options(execution: ExecutionDep):
        return execution.augmentation.options()

    @app.post(
        "/api/v1/projects/{project_id}/augmentations",
        response_model=Job,
        status_code=202,
        responses=SUBMISSION_RESPONSES,
    )
    async def augment_dataset(
        project_id: str,
        payload: AugmentationRequest,
        projects: ProjectsDep,
        execution: ExecutionDep,
        response: Response,
        idempotency_key: SubmissionKeyDep,
    ) -> Job:
        if await projects.get(project_id) is None:
            raise HTTPException(404, "Project not found")
        try:
            job = await execution.submit(project_id, payload, idempotency_key=idempotency_key)
            if idempotency_key is not None:
                response.headers["Idempotency-Key"] = idempotency_key
                response.headers["Cache-Control"] = "no-store"
            return job
        except SubmissionConflict:
            raise
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/v1/jobs/{job_id}/augmentation/download")
    async def download_augmentation(job_id: str, execution: ExecutionDep):
        try:
            path = await execution.augmentation.download(job_id)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return FileResponse(
            path, media_type="application/zip", filename=f"augmentation-{job_id}.zip"
        )

    @app.get("/api/v1/jobs/{job_id}/augmentation/clips/{index}")
    async def augmentation_clip(
        job_id: str,
        index: int,
        execution: ExecutionDep,
        original: bool = False,
    ):
        try:
            path = await execution.augmentation.download(job_id, index, original)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return FileResponse(path, media_type="video/mp4")

    @app.get("/api/v1/policy-options")
    async def policy_options(execution: ExecutionDep) -> dict:
        return {
            **execution.lifecycle.compute.public_catalog(execution.lifecycle.catalog),
            "compute": execution.lifecycle.compute.preferences().model_dump(),
            "training_models": public_training_models(
                execution.lifecycle.compute.available_catalog(execution.lifecycle.catalog)
            ),
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
                {
                    "id": "full",
                    "label": "Native training",
                    "description": "Use the architecture's native training recipe",
                },
            ],
            "default_training_method": "lora",
            "quantization_defaults": {
                "cuda": {"language": "Q8_0", "vision": None},
                "cpu": {"language": "Q8_0", "vision": None},
                "note": (
                    "Q8 is the initial comparison candidate on CPU and CUDA, "
                    "not a quality guarantee. "
                    "Q4 and vision packing are experimental; validate every policy on its target."
                ),
            },
        }

    @app.get("/api/v1/simulation-options")
    async def simulation_options(execution: ExecutionDep) -> dict:
        from vla_platform.lifecycle.simulation import options

        return await asyncio.to_thread(options, execution.settings)

    @app.post("/api/v1/projects/{project_id}/model-imports", response_model=Job, status_code=202)
    async def import_native_policy(
        project_id: str,
        request: Request,
        projects: ProjectsDep,
        execution: ExecutionDep,
        profile_id: str = Query(pattern=r"^[\w-]{1,100}$"),
    ) -> Job:
        """Upload a complete native ACT/SmolVLA TAR; import is an observable local job."""
        from vla_platform.lifecycle.simulation import MAX_ARCHIVE_BYTES, finish_owned, profile_for

        if request.headers.getlist("idempotency-key"):
            raise HTTPException(
                422,
                "Binary model uploads do not support Idempotency-Key; "
                "reconcile the existing upload job before another upload",
            )
        if await projects.get(project_id) is None:
            raise HTTPException(404, "Project not found")
        try:
            profile_for(execution.settings, profile_id)
        except (OSError, ValueError) as exc:
            raise HTTPException(422, "The selected native policy profile is unavailable") from exc
        if request.headers.get("content-type", "").split(";")[0] not in {
            "application/x-tar",
            "application/octet-stream",
            "application/gzip",
        }:
            raise HTTPException(415, "Send a TAR archive as the request body")
        size_header = request.headers.get("content-length")
        if size_header is not None and (
            not size_header.isdigit() or not 0 < int(size_header) <= MAX_ARCHIVE_BYTES
        ):
            raise HTTPException(413, "Model archive exceeds the 4 GiB upload limit or is empty")
        upload_id = uuid4().hex
        directory = execution.settings.data_dir / "model-uploads" / upload_id
        directory.mkdir(parents=True, mode=0o700)
        accepted = False
        try:
            digest = hashlib.sha256()
            total = 0
            async with asyncio.timeout(300):
                with (directory / "archive.tar").open("xb") as stream:
                    async for chunk in request.stream():
                        total += len(chunk)
                        if total > MAX_ARCHIVE_BYTES:
                            raise HTTPException(413, "Model archive exceeds the 4 GiB upload limit")
                        await finish_owned(asyncio.to_thread(stream.write, chunk))
                        digest.update(chunk)
            if not total or (size_header is not None and total != int(size_header)):
                raise HTTPException(422, "The uploaded model archive is incomplete")
            (directory / "upload.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "upload_id": upload_id,
                        "project_id": project_id,
                        "profile_id": profile_id,
                        "bytes": total,
                        "sha256": digest.hexdigest(),
                    }
                )
            )
            payload = PolicyRequest.model_validate(
                {
                    "operation": "policy.import",
                    "runtime_id": profile_id,
                    "source_id": upload_id,
                    "simulation": {"profile_id": profile_id},
                    "timeout_seconds": 600,
                }
            )

            async def submit_owned():
                nonlocal accepted
                job = await execution.submit(project_id, payload)
                accepted = True
                return job

            return await finish_owned(submit_owned())
        except TimeoutError as exc:
            raise HTTPException(408, "Model upload timed out; no import job was submitted") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            if not accepted:
                shutil.rmtree(directory)

    @app.get("/api/v1/jobs/{job_id}/simulation-media/video")
    async def simulation_video(job_id: str, execution: ExecutionDep):
        from vla_platform.lifecycle.simulation import video_path

        job = await execution.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        try:
            path = await asyncio.to_thread(video_path, execution.settings, job)
        except (OSError, ValueError) as exc:
            raise HTTPException(422, "Verified simulation video is unavailable") from exc
        return FileResponse(path, media_type="video/mp4")

    @app.post(
        "/api/v1/projects/{project_id}/policy-jobs",
        response_model=Job,
        status_code=202,
        responses=SUBMISSION_RESPONSES,
    )
    async def policy_job(
        project_id: str,
        payload: PolicyRequest,
        projects: ProjectsDep,
        execution: ExecutionDep,
        response: Response,
        idempotency_key: SubmissionKeyDep,
    ) -> Job:
        if await projects.get(project_id) is None:
            raise HTTPException(404, "Project not found")
        try:
            job = await execution.submit(project_id, payload, idempotency_key=idempotency_key)
            if idempotency_key is not None:
                response.headers["Idempotency-Key"] = idempotency_key
                response.headers["Cache-Control"] = "no-store"
            return job
        except SubmissionConflict:
            raise
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get(
        "/api/v1/projects/{project_id}/submissions/{key}",
        response_model=Job,
        responses={
            404: {"description": "No accepted job is bound to this scope and key"},
            503: {"description": "Saved submission identity cannot be verified"},
        },
    )
    async def saved_submission(
        project_id: str,
        key: IdempotencyKey,
        operation: SubmissionOperation,
        projects: ProjectsDep,
        execution: ExecutionDep,
        response: Response,
    ) -> Job:
        """Observe an accepted job without resolving inputs, dispatching or retrying it."""
        if await projects.get(project_id) is None:
            raise HTTPException(404, "Project not found")
        job = await submission_lookup(execution.storage, project_id, operation, key)
        if job is None:
            raise HTTPException(
                404,
                "No accepted submission found for this project, operation and key",
                headers={"Idempotency-Key": key, "Cache-Control": "no-store"},
            )
        response.headers["Idempotency-Key"] = key
        response.headers["Cache-Control"] = "no-store"
        return job

    @app.get("/api/v1/projects/{project_id}/artifacts", response_model=list[PolicyArtifact])
    async def artifacts(project_id: str, projects: ProjectsDep, execution: ExecutionDep):
        if await projects.get(project_id) is None:
            raise HTTPException(404, "Project not found")
        return await execution.lifecycle.artifacts(project_id)

    @app.get("/api/v1/projects/{project_id}/artifacts/{artifact_id}/replay")
    async def replay_record(
        project_id: str,
        artifact_id: str,
        projects: ProjectsDep,
        execution: ExecutionDep,
        response: Response,
    ) -> dict:
        """Read verified saved CPU predictions; never execute a policy or simulator."""
        from vla_platform.lifecycle.native_replay import read_record

        if await projects.get(project_id) is None:
            raise HTTPException(404, "Project not found")
        try:
            value = await read_record(execution.lifecycle, project_id, artifact_id)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, "Saved replay record is unavailable or invalid") from exc
        response.headers["Cache-Control"] = "no-store"
        return value

    @app.get("/api/v1/projects/{project_id}/artifacts/{artifact_id}/download")
    async def download_artifact(project_id: str, artifact_id: str, execution: ExecutionDep):
        try:
            path = await execution.lifecycle.download(project_id, artifact_id)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        from vla_platform.lifecycle.cloud_download import CloudDownload

        if isinstance(path, CloudDownload):
            return StreamingResponse(
                path,
                media_type="application/x-tar",
                headers={"Content-Disposition": f'attachment; filename="{path.filename}"'},
                background=BackgroundTask(path.aclose),
            )
        return FileResponse(path, media_type="application/x-tar", filename=path.name)

    @app.get("/api/v1/jobs/{job_id}/events", response_model=list[JobEvent])
    async def job_events(job_id: str, execution: ExecutionDep, after: int = 0):
        if await execution.get(job_id) is None:
            raise HTTPException(404, "Job not found")
        try:
            return execution.lifecycle.events(job_id, after)
        except ValueError as exc:
            raise HTTPException(
                422, "Job event history is corrupt; preserve it for inspection"
            ) from exc

    @app.get("/api/v1/jobs/{job_id}/training", response_model=TrainingTelemetry)
    async def training_telemetry(job_id: str, execution: ExecutionDep):
        job = await execution.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        try:
            return await telemetry.snapshot(execution.lifecycle, job)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/v1/jobs/{job_id}/training/reproducibility")
    async def training_reproducibility(job_id: str, execution: ExecutionDep):
        job = await execution.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        if not telemetry.training_job(job):
            raise HTTPException(422, "This job is not a training run")
        return JSONResponse(
            await telemetry.reproducibility(execution.lifecycle, job),
            headers={"Content-Disposition": f'attachment; filename="training-{job.id}.json"'},
        )

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

        @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
        async def workspace_home(request: Request):
            # Canonicalize before rendering: a delayed client-side redirect can
            # otherwise overwrite the user's first sidebar navigation.
            prefix = request.scope.get("root_path", "").rstrip("/")
            return RedirectResponse(request.url.replace(path=f"{prefix}/dashboard/"))

        app.mount("/", StaticFiles(directory=settings.static_dir, html=True), name="web")
    return app
