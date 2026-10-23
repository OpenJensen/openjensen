"""Browser-facing Robot Lab API; credentials remain on the operator's server."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from vla_platform.robot_lab_bridge import RobotLabError, safe_id, strict_json

router = APIRouter(prefix="/api/v1", tags=["Robot Lab"])


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
    "GET": {"devices", "datasets", "camera.jpg"},
    "POST": {
        "probe",
        "setup/motor",
        "setup/assembled/start",
        "setup/assembled/observe",
        "setup/assembled/accept",
        "setup/assembled/finish",
        "setup/assembled/cancel",
        "calibration/start",
        "calibration/center",
        "calibration/finish",
        "calibration/cancel",
        "control/start",
        "control/jog",
        "control/heartbeat",
        "control/stop",
        "recording/start",
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
        )


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
