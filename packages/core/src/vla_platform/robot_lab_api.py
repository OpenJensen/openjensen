"""Browser-facing Robot Lab API; credentials remain on the operator's server."""

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    ValidationError,
    model_validator,
)

from vla_platform.robot_lab_bridge import RobotLabError, safe_id, strict_json
from vla_platform.robot_policy_install import PolicyInstallRequest, install_artifact

router = APIRouter(prefix="/api/v1", tags=["Robot Lab"])


class PolicyLoadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    policy_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")


class PolicyStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confirm: Literal["RUN POLICY"]
    operator_present: Literal[True]
    workspace_clear: Literal[True]
    camera_index: StrictInt = Field(ge=0, le=63)
    duration_seconds: StrictInt = Field(ge=1, le=120)
    expected_generation: StrictInt


class PolicyPreflightRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    camera_index: StrictInt = Field(ge=0, le=63)


class PolicyHeartbeatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_generation: StrictInt


class PolicyStopRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


POLICY_REQUESTS = {
    "policy/load": PolicyLoadRequest,
    "policy/start": PolicyStartRequest,
    "policy/preflight": PolicyPreflightRequest,
    "policy/heartbeat": PolicyHeartbeatRequest,
    "policy/stop": PolicyStopRequest,
}


class DeviceConfigurationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=100)
    port: str = Field(max_length=256)
    name: str = Field(max_length=64)
    role: Literal["follower", "leader"]


class DeviceVisibilityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=100)


class CalibrationImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    robot_id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    role: Literal["follower", "leader"]
    document: dict
    port: str | None = None
    replace_existing: StrictBool = False
    confirm: Literal["IMPORT CALIBRATION"]


class RecordingCollectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    episode_time_s: StrictInt = Field(default=30, ge=1, le=300)
    reset_time_s: StrictInt = Field(default=10, ge=0, le=300)
    num_episodes: StrictInt = Field(default=10, ge=1, le=1000)
    countdown_s: StrictInt = Field(default=3, ge=0, le=30)


class RecordingStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    task: str = Field(min_length=1, max_length=1000)
    fps: StrictInt | None = Field(default=None, ge=1, le=40)
    camera_index: StrictInt | None = Field(default=0, ge=0, le=63)
    collection: RecordingCollectionRequest | None = None


class RecordingCollectionActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RecordingFinishRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "allOf": [
                {
                    "if": {
                        "properties": {"abandon_empty": {"const": True}},
                        "required": ["abandon_empty"],
                    },
                    "then": {
                        "required": ["confirm"],
                        "properties": {"confirm": {"const": "ABANDON EMPTY RECORDING"}},
                    },
                }
            ],
        },
    )
    abandon_empty: StrictBool = False
    confirm: Literal["ABANDON EMPTY RECORDING"] | None = None

    @model_validator(mode="after")
    def require_abandonment_confirmation(self):
        if self.abandon_empty and self.confirm != "ABANDON EMPTY RECORDING":
            raise ValueError("Confirm abandoning an empty recording explicitly")
        return self


@router.get("/robot-lab/status")
async def status(request: Request):
    return await request.app.state.robot_lab.status()


async def proxy(path: str, request: Request):
    bridge = request.app.state.robot_lab
    try:
        payload = None
        if request.method == "POST":
            chunks, size = [], 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > 64 * 1024:
                    raise HTTPException(413, "Robot Lab requests support at most 64 KiB")
                chunks.append(chunk)
            try:
                raw = b"".join(chunks)
                payload = strict_json(raw) if raw else {}
                if not isinstance(payload, dict):
                    raise ValueError
            except ValueError, UnicodeDecodeError, RecursionError:
                raise HTTPException(422, "Supply a JSON object for this operation") from None
        if path in POLICY_REQUESTS:
            try:
                POLICY_REQUESTS[path].model_validate(payload)
            except ValidationError:
                raise HTTPException(
                    422, "Choose valid policy settings and confirm attended operation"
                ) from None
        if path == "devices/configure":
            try:
                DeviceConfigurationRequest.model_validate(payload)
            except ValidationError:
                raise HTTPException(422, "Choose a device, name and default role") from None
        if path in {"devices/remove", "devices/restore"}:
            try:
                DeviceVisibilityRequest.model_validate(payload)
            except ValidationError:
                raise HTTPException(422, "Choose a saved device") from None
        if path == "calibration/import":
            try:
                CalibrationImportRequest.model_validate(payload)
            except ValidationError:
                raise HTTPException(
                    422, "Choose a calibration file, destination profile and arm role"
                ) from None
        if path == "recording/start" or path.startswith("recording/collection/"):
            try:
                schema = (
                    RecordingStartRequest
                    if path == "recording/start"
                    else RecordingCollectionActionRequest
                )
                schema.model_validate(payload)
            except ValidationError:
                raise HTTPException(422, "Choose valid recording settings") from None
        if path == "recording/finish":
            try:
                RecordingFinishRequest.model_validate(payload)
            except ValidationError:
                raise HTTPException(
                    422, "Confirm abandoning an empty recording explicitly"
                ) from None
        if len(str(request.url.query)) > 2048:
            raise HTTPException(413, "Robot Lab query is too large")
        image = path == "camera.jpg"
        result = await bridge.request(
            request.method,
            path,
            payload=payload,
            query=list(request.query_params.multi_items()),
            image=image,
        )
        if image:
            content, media_type = result
            return Response(content, media_type=media_type, headers={"Cache-Control": "no-store"})
        return JSONResponse(result, headers={"Cache-Control": "no-store"})
    except RobotLabError as exc:
        raise HTTPException(exc.status, str(exc)) from None


PROTOCOL = {
    "GET": {
        "devices",
        "datasets",
        "camera.jpg",
        "diagnostics",
        "calibration/export",
        "policy/catalog",
    },
    "POST": {
        *POLICY_REQUESTS,
        "devices/configure",
        "devices/remove",
        "devices/restore",
        "probe",
        "setup/motor",
        "mapping/wiggle",
        "mapping/confirm",
        "setup/assembled/start",
        "setup/assembled/observe",
        "setup/assembled/wiggle",
        "setup/assembled/accept",
        "setup/assembled/finish",
        "setup/assembled/cancel",
        "calibration/start",
        "calibration/import",
        "calibration/center",
        "calibration/finish",
        "calibration/cancel",
        "agent/acquire",
        "agent/observe",
        "agent/execute",
        "agent/heartbeat",
        "agent/release",
        "control/start",
        "control/rest/save",
        "control/rest/return",
        "control/jog",
        "control/heartbeat",
        "control/stop",
        "recording/start",
        "recording/collection/start",
        "recording/collection/pause",
        "recording/collection/resume",
        "recording/collection/next",
        "recording/collection/rerecord",
        "recording/episode/start",
        "recording/episode/save",
        "recording/episode/discard",
        "recording/finish",
        "stop",
    },
}


def proxy_endpoint(path):
    async def endpoint(request: Request):
        return await proxy(path, request)

    return endpoint


def request_schema(model):
    schema = model.model_json_schema()
    if model is RecordingStartRequest:
        # Inline nested settings: this schema is embedded in a request body,
        # so Pydantic's root-relative $defs references would not resolve.
        schema.pop("$defs", None)
        schema["properties"]["collection"] = {
            "anyOf": [RecordingCollectionRequest.model_json_schema(), {"type": "null"}],
            "default": None,
        }
    return schema


# Register the finite protocol as real routes, so API reference/client generation
# describe every operation and never expose an arbitrary proxy-path parameter.
for method, protocol_paths in PROTOCOL.items():
    for protocol_path in sorted(protocol_paths):
        ident = (
            "robot_lab_" + method.lower() + "_" + protocol_path.replace("/", "_").replace(".", "_")
        )
        router.add_api_route(
            "/robot-lab/" + protocol_path,
            proxy_endpoint(protocol_path),
            methods=[method],
            operation_id=ident,
            name=ident,
            openapi_extra={
                "requestBody": {
                    "required": False,
                    "content": {
                        "application/json": {
                            "schema": request_schema(
                                POLICY_REQUESTS[protocol_path]
                                if protocol_path in POLICY_REQUESTS
                                else DeviceConfigurationRequest
                                if protocol_path == "devices/configure"
                                else DeviceVisibilityRequest
                                if protocol_path in {"devices/remove", "devices/restore"}
                                else CalibrationImportRequest
                                if protocol_path == "calibration/import"
                                else RecordingStartRequest
                                if protocol_path == "recording/start"
                                else RecordingCollectionActionRequest
                                if protocol_path.startswith("recording/collection/")
                                else RecordingFinishRequest
                            ),
                        }
                    },
                }
            }
            if protocol_path
            in {
                *POLICY_REQUESTS,
                "recording/finish",
                "recording/start",
                "recording/collection/start",
                "recording/collection/pause",
                "recording/collection/resume",
                "recording/collection/next",
                "recording/collection/rerecord",
                "calibration/import",
                "devices/configure",
                "devices/remove",
                "devices/restore",
            }
            else None,
        )


@router.get("/robot-lab/agent/executions/{command_id}")
async def agent_execution(command_id: str, request: Request):
    import re

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", command_id):
        raise HTTPException(422, "Invalid agent command identity")
    try:
        value = await request.app.state.robot_lab.request("GET", "agent/executions/" + command_id)
        return JSONResponse(value, headers={"Cache-Control": "no-store"})
    except RobotLabError as error:
        raise HTTPException(error.status, str(error)) from None


@router.post("/projects/{project_id}/robot-lab/datasets/{dataset_id}/import", status_code=201)
async def import_recording(project_id: str, dataset_id: str, request: Request):
    if await request.app.state.projects.get(project_id) is None:
        raise HTTPException(404, "Project not found")
    try:
        safe_id(dataset_id)
        return await request.app.state.robot_lab.import_dataset(
            project_id, dataset_id, request.app.state.dataset_library
        )
    except RobotLabError as exc:
        raise HTTPException(exc.status, str(exc)) from None


@router.post("/projects/{project_id}/robot-lab/policies/{artifact_id}/install", status_code=201)
async def install_project_policy(
    project_id: str, artifact_id: str, payload: PolicyInstallRequest, request: Request
):
    try:
        return await install_artifact(request.app, project_id, artifact_id, payload)
    except RobotLabError as error:
        raise HTTPException(error.status, str(error)) from None
