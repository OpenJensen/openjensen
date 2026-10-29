"""Private Agent Control sessions, explicitly separate from Teaching advice."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from vla_platform.agent_control.contracts import (
    Approval,
    Event,
    Heartbeat,
    PlanRequest,
    ProviderInput,
    ProviderStatus,
    Session,
    SessionCreate,
    TaskReview,
)
from vla_platform.agent_control.service import AgentControl, AgentError
from vla_platform.agent_control.settings import ProfileError
from vla_platform.teaching_intelligence_api import PrivateRoute

router = APIRouter(prefix="/api/v1", tags=["Agent Control"], route_class=PrivateRoute)


def service(request: Request) -> AgentControl:
    return request.app.state.agent_control


AgentDep = Annotated[AgentControl, Depends(service)]


@router.get("/agent/openai/models", response_model=list[str])
async def openai_models(agent: AgentDep):
    from vla_platform.agent_control.model_catalog import ModelCatalogError, list_openai_models

    if not agent.profiles.integration_installed():
        raise AgentError("Install OpenAI in Modules before loading models", 409)
    key = agent.profiles.shared_key()
    if not key:
        raise AgentError("Save an OpenAI API key in Cloud & APIs before loading models", 422)
    try:
        return await list_openai_models(key)
    except ModelCatalogError as error:
        raise AgentError(str(error), 503) from None


@router.get("/agent/providers", response_model=list[ProviderStatus])
async def providers(agent: AgentDep):
    return agent.profiles.list()


@router.put("/agent/providers/{provider_id}", response_model=ProviderStatus)
async def save_provider(provider_id: str, payload: ProviderInput, agent: AgentDep):
    import re

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", provider_id):
        raise AgentError("Invalid provider identity", 422)
    return agent.profiles.save(provider_id, payload)


@router.delete("/agent/providers/{provider_id}")
async def delete_provider(provider_id: str, agent: AgentDep):
    agent.profiles.remove(provider_id)
    return {"removed": True}


@router.post("/agent/providers/{provider_id}/check", response_model=ProviderStatus)
async def check_provider(provider_id: str, agent: AgentDep):
    return await agent.check_provider(provider_id)


async def project(request, project_id):
    if await request.app.state.projects.get(project_id) is None:
        raise HTTPException(404, "Project not found")


@router.get("/projects/{project_id}/agent-sessions", response_model=list[Session])
async def sessions(project_id: str, request: Request, agent: AgentDep):
    await project(request, project_id)
    return await agent.list(project_id)


@router.post("/projects/{project_id}/agent-sessions", response_model=Session, status_code=201)
async def create_session(
    project_id: str, payload: SessionCreate, request: Request, agent: AgentDep
):
    await project(request, project_id)
    return await agent.create(project_id, payload)


@router.get("/agent-sessions/{session_id}", response_model=Session)
async def session(session_id: str, agent: AgentDep):
    return await agent.get(session_id)


@router.get("/agent-sessions/{session_id}/events", response_model=list[Event])
async def events(session_id: str, agent: AgentDep, after: int = Query(default=0, ge=0)):
    return await agent.events(session_id, after)


@router.post("/agent-sessions/{session_id}/observe", response_model=Session)
async def observe(session_id: str, agent: AgentDep):
    return await agent.observe(session_id)


@router.post("/agent-sessions/{session_id}/proposals", response_model=Session, status_code=202)
async def propose(session_id: str, payload: PlanRequest, agent: AgentDep):
    return await agent.propose(session_id, payload)


@router.post("/agent-sessions/{session_id}/approve", response_model=Session, status_code=202)
async def approve(session_id: str, payload: Approval, agent: AgentDep):
    return await agent.approve(session_id, payload)


@router.post("/agent-sessions/{session_id}/cancel-inference", response_model=Session)
async def cancel(session_id: str, agent: AgentDep):
    return await agent.pause_planning(session_id, "cancel_inference")


@router.post("/agent-sessions/{session_id}/pause", response_model=Session)
async def pause(session_id: str, agent: AgentDep):
    return await agent.pause_planning(session_id, "pause")


@router.post("/agent-sessions/{session_id}/stop", response_model=Session)
async def stop(session_id: str, agent: AgentDep):
    return await agent.interrupt(session_id, "stop")


def register_errors(app):
    from fastapi.responses import JSONResponse

    @app.exception_handler(AgentError)
    async def agent_error(_request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=exc.status)

    @app.exception_handler(ProfileError)
    async def profile_error(_request, exc):
        return JSONResponse(
            {
                "detail": (
                    "Provider profile is unavailable or invalid. Replace its private "
                    "settings explicitly."
                )
            },
            status_code=422,
        )


@router.post("/agent-sessions/{session_id}/run", response_model=Session, status_code=202)
async def run_task(session_id: str, agent: AgentDep):
    return await agent.run_task(session_id)


@router.post("/agent-sessions/{session_id}/heartbeat", response_model=Session)
async def heartbeat(session_id: str, payload: Heartbeat, agent: AgentDep):
    return await agent.heartbeat(session_id, payload)


@router.post("/agent-sessions/{session_id}/execution-observation", response_model=Session)
async def execution_observation(session_id: str, agent: AgentDep):
    return await agent.execution_observe(session_id)


@router.get("/agent-sessions/{session_id}/observations/{observation_id}/image")
async def observation_image(session_id: str, observation_id: str, agent: AgentDep):
    from fastapi import Response

    data, mime = await agent.observation_image(session_id, observation_id)
    return Response(
        data,
        media_type=mime,
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )


@router.post("/agent-sessions/{session_id}/review", response_model=Session)
async def review_task(session_id: str, payload: TaskReview, agent: AgentDep):
    return await agent.review_task(session_id, payload)
