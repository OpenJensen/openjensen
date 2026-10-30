"""Local companion admission and exact recording transfers, without robot movement."""

import asyncio
import hashlib
import json
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from vla_platform.datasets.library import DatasetLibrary, LibraryError
from vla_platform.robot_lab_api import router
from vla_platform.robot_lab_bridge import RobotLabBridge, RobotLabError

TOKEN = "private-companion-token-" + "a" * 32


def connection(root, **changes):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "robot-lab-connection.json"
    path.write_text(json.dumps({"url": "http://127.0.0.1:18101", "token": TOKEN, **changes}))
    path.chmod(0o600)
    return path


def bridge(root, handler):
    return RobotLabBridge(root, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def browser(root, companion):
    class Projects:
        async def get(self, ident):
            return {"id": ident} if ident == "project-one" else None

    app = FastAPI()
    app.include_router(router)
    app.state.robot_lab = companion
    app.state.projects = Projects()
    app.state.dataset_library = DatasetLibrary(root)

    @app.exception_handler(LibraryError)
    async def library_error(request, exc):
        from fastapi.responses import JSONResponse

        return JSONResponse({"detail": str(exc)}, status_code=exc.status)

    return TestClient(app)


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:18101",
        "http://localhost:18101",
        "http://example.com:18101",
        "http://127.0.0.1:80",
        "http://127.0.0.1:18101/private",
        "http://user:password@127.0.0.1:18101",
        "http://127.0.0.1:18101?redirect=example.com",
        "http://127.0.0.1:\n18101",
    ],
)
def test_configuration_only_allows_private_loopback_operator_targets(tmp_path, url):
    connection(tmp_path, url=url)
    companion = bridge(tmp_path, lambda _: pytest.fail("Invalid config contacted a target"))
    with pytest.raises(RobotLabError, match="operator setup"):
        companion.config()
    asyncio.run(companion.close())


def test_connection_file_must_be_private_owned_regular_file(tmp_path):
    path = connection(tmp_path)
    companion = bridge(tmp_path, lambda _: pytest.fail("Invalid config contacted a target"))
    path.chmod(0o644)
    with pytest.raises(RobotLabError):
        companion.config()
    path.chmod(0o600)
    path.rename(tmp_path / "actual.json")
    path.symlink_to(tmp_path / "actual.json")
    with pytest.raises(RobotLabError):
        companion.config()
    asyncio.run(companion.close())


def test_status_reports_missing_or_unavailable_companion_without_secret(tmp_path):
    def offline(request):
        raise httpx.ConnectError("network diagnostics " + TOKEN)

    companion = bridge(tmp_path, offline)
    with browser(tmp_path, companion) as client:
        missing = client.get("/api/v1/robot-lab/status")
        assert missing.status_code == 200 and missing.json()["connected"] is False
        connection(tmp_path)
        unavailable = client.get("/api/v1/robot-lab/status")
        assert unavailable.json()["connected"] is False
        assert TOKEN not in unavailable.text
        devices = client.get("/api/v1/robot-lab/devices")
        assert devices.status_code == 502 and TOKEN not in devices.text
    asyncio.run(companion.close())


def test_proxy_preserves_json_uses_bearer_and_redacts_response_secrets(tmp_path):
    connection(tmp_path, label="Lab Mac")
    calls = []

    def handler(request):
        assert request.headers["authorization"] == "Bearer " + TOKEN
        calls.append((request.method, request.url.path, request.content))
        return httpx.Response(200, json={"ok": True, "token": TOKEN, "message": "value " + TOKEN})

    companion = bridge(tmp_path, handler)
    with browser(tmp_path, companion) as client:
        payload = {"port": "/dev/cu.usbmodem123", "baudrate": 1000000}
        response = client.post("/api/v1/robot-lab/probe", json=payload)
        assert response.status_code == 200
        assert calls[0][:2] == ("POST", "/v1/probe")
        assert json.loads(calls[0][2]) == payload
        assert TOKEN not in response.text and "token" not in response.json()
        assert response.headers["cache-control"] == "no-store"
        status = client.get("/api/v1/robot-lab/status").json()
        assert status["connected"] and status["label"] == "Lab Mac"
    asyncio.run(companion.close())


def test_route_body_and_redirect_boundaries_never_become_a_general_proxy(tmp_path):
    connection(tmp_path)
    calls = []

    def handler(request):
        calls.append(request.url)
        return httpx.Response(307, headers={"location": "https://example.com/?secret=" + TOKEN})

    companion = bridge(tmp_path, handler)
    with browser(tmp_path, companion) as client:
        for path in ("config", "datasets/example/files", "http://example.com", "control/unknown"):
            assert client.post("/api/v1/robot-lab/" + path, json={}).status_code == 404
        assert not calls
        assert client.post("/api/v1/robot-lab/probe", content=b"x" * 65537).status_code == 413
        assert client.post("/api/v1/robot-lab/probe", json=[]).status_code == 422
        assert client.post("/api/v1/robot-lab/probe", content=b'{"delta": NaN}').status_code == 422
        assert not calls
        response = client.get("/api/v1/robot-lab/devices")
        assert response.status_code == 502 and "redirect" in response.text
        assert TOKEN not in response.text and len(calls) == 1
    asyncio.run(companion.close())


def test_camera_returns_only_bounded_images_and_never_cache(tmp_path, monkeypatch):
    connection(tmp_path)
    mode = ["valid"]

    def handler(request):
        assert request.url.path == "/v1/camera.jpg" and request.url.params["index"] == "1"
        return httpx.Response(
            200,
            content=b"\xff\xd8camera",
            headers={"content-type": "image/jpeg" if mode[0] == "valid" else "text/html"},
        )

    companion = bridge(tmp_path, handler)
    with browser(tmp_path, companion) as client:
        response = client.get("/api/v1/robot-lab/camera.jpg?index=1")
        assert response.status_code == 200 and response.content == b"\xff\xd8camera"
        assert response.headers["cache-control"] == "no-store"
        mode[0] = "wrong-type"
        assert client.get("/api/v1/robot-lab/camera.jpg?index=1").status_code == 502
        mode[0] = "valid"
        monkeypatch.setattr("vla_platform.robot_lab_bridge.MAX_IMAGE", 3)
        assert client.get("/api/v1/robot-lab/camera.jpg?index=1").status_code == 502
    asyncio.run(companion.close())


def fixture_files():
    root = Path(__file__).parent / "fixtures" / "lerobot_v3_preview"
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"README.md", "manifest.json", "generate.py"}
    }


def manifest(files):
    return {
        "id": "pick-block",
        "name": "Pick block",
        "files": [
            {"path": path, "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            for path, raw in files.items()
        ],
        "total_bytes": sum(map(len, files.values())),
    }


def test_project_import_keeps_original_bytes_and_reuses_a_finalized_dataset(tmp_path):
    connection(tmp_path)
    files, transfers = fixture_files(), []

    def handler(request):
        if request.url.path.endswith("/manifest"):
            return httpx.Response(200, json=manifest(files))
        assert request.url.path == "/v1/datasets/pick-block/files"
        path = request.url.params["path"]
        transfers.append(path)
        return httpx.Response(200, content=files[path])

    companion = bridge(tmp_path, handler)
    with browser(tmp_path, companion) as client:
        route = "/api/v1/projects/project-one/robot-lab/datasets/pick-block/import"
        response = client.post(route)
        assert response.status_code == 201, response.text
        accepted = response.json()
        assert accepted["status"] == "ready" and accepted["project_id"] == "project-one"
        assert accepted["detection"]["frames"] == 6
        saved = client.app.state.dataset_library.resolve(accepted["id"], "project-one")
        assert {
            p.relative_to(saved).as_posix(): p.read_bytes() for p in saved.rglob("*") if p.is_file()
        } == files
        assert client.post(route).json()["id"] == accepted["id"]
        assert len(transfers) == len(files)
        assert client.post(route.replace("project-one", "missing")).status_code == 404
    asyncio.run(companion.close())


def test_interrupted_file_is_not_published_and_retry_resumes_saved_upload(tmp_path):
    connection(tmp_path)
    files, broken = fixture_files(), [True]
    first_path = sorted(files)[0]

    def handler(request):
        if request.url.path.endswith("/manifest"):
            return httpx.Response(200, json=manifest(files))
        path = request.url.params["path"]
        raw = files[path]
        return httpx.Response(200, content=raw[:-1] if broken[0] and path == first_path else raw)

    companion = bridge(tmp_path, handler)
    with browser(tmp_path, companion) as client:
        route = "/api/v1/projects/project-one/robot-lab/datasets/pick-block/import"
        failed = client.post(route)
        assert failed.status_code == 502 and "changed during transfer" in failed.text
        library = client.app.state.dataset_library
        records = list(library.records())
        assert len(records) == 1 and records[0]["status"] == "uploading"
        assert not (library.directory(records[0]["id"]) / "source" / first_path).exists()
        broken[0] = False
        resumed = client.post(route)
        assert resumed.status_code == 201 and resumed.json()["id"] == records[0]["id"]
        assert len(list(library.records())) == 1
    asyncio.run(companion.close())


@pytest.mark.parametrize("unsafe", ["../escape", "/absolute", "a/../meta/info.json", "a\\file"])
def test_import_rejects_unsafe_manifest_before_creating_upload(tmp_path, unsafe):
    connection(tmp_path)
    files = {unsafe: b"data"}
    companion = bridge(tmp_path, lambda _: httpx.Response(200, json=manifest(files)))
    with browser(tmp_path, companion) as client:
        response = client.post("/api/v1/projects/project-one/robot-lab/datasets/pick-block/import")
        assert response.status_code == 502
        assert not list(client.app.state.dataset_library.records())
    asyncio.run(companion.close())


def test_recording_save_allows_compression_time_within_browser_deadline(tmp_path):
    connection(tmp_path)
    observed = []

    def handler(request):
        observed.append((request.url.path, request.extensions["timeout"]["read"]))
        return httpx.Response(200, json={"ok": True})

    companion = bridge(tmp_path, handler)

    async def run():
        try:
            await companion.request("POST", "recording/episode/save", payload={})
            await companion.request("POST", "recording/finish", payload={})
            await companion.request("GET", "devices")
        finally:
            await companion.close()

    asyncio.run(run())
    assert observed == [
        ("/v1/recording/episode/save", 80),
        ("/v1/recording/finish", 80),
        ("/v1/devices", 30),
    ]


@pytest.mark.parametrize(
    "action,payload",
    [
        ("start", {"port": "/dev/cu.test", "supported": True, "confirm": "START ASSEMBLED SETUP"}),
        ("observe", {"joint": "elbow_flex"}),
        ("wiggle", {"motor_id": 12, "supported": True, "confirm": "WIGGLE MOTOR"}),
        ("accept", {"joint": "elbow_flex", "motor_id": 0}),
        ("finish", {"confirm": "SAVE MOTOR IDS"}),
        ("cancel", {}),
    ],
)
def test_assembled_setup_protocol_routes_preserve_body_and_do_not_expose_generic_proxy(
    tmp_path, action, payload
):
    connection(tmp_path)
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"ok": True})

    companion = bridge(tmp_path, handler)
    with browser(tmp_path, companion) as client:
        route = "/api/v1/robot-lab/setup/assembled/" + action
        assert client.post(route, json=payload).status_code == 200
        assert client.get(route).status_code == 405
        assert (
            client.post("/api/v1/robot-lab/setup/assembled/arbitrary", json={}).status_code == 404
        )
    assert calls == [("POST", "/v1/setup/assembled/" + action, payload)]
    asyncio.run(companion.close())


@pytest.mark.parametrize(
    "payload",
    [{}, {"abandon_empty": False}, {"abandon_empty": True, "confirm": "ABANDON EMPTY RECORDING"}],
)
def test_recording_finish_preserves_explicit_close_and_legacy_finalize_payloads(tmp_path, payload):
    connection(tmp_path)
    calls = []

    def handler(request):
        calls.append((request.url.path, json.loads(request.content)))
        return httpx.Response(
            200,
            json={
                "recording": {"active": False},
                "datasets": [
                    {"id": "empty-demo", "finalized": False, "abandoned": True, "pending_frames": 3}
                ],
            },
        )

    companion = bridge(tmp_path, handler)
    with browser(tmp_path, companion) as client:
        response = client.post("/api/v1/robot-lab/recording/finish", json=payload)
        assert response.status_code == 200
        assert response.json()["datasets"][0]["pending_frames"] == 3
        assert response.json()["datasets"][0]["abandoned"] is True
    assert calls == [("/v1/recording/finish", payload)]
    asyncio.run(companion.close())


@pytest.mark.parametrize(
    "payload",
    [
        {"abandon_empty": True},
        {"abandon_empty": True, "confirm": None},
        {"abandon_empty": True, "confirm": "yes"},
        {"abandon_empty": "true"},
        {"abandon_empty": 1, "confirm": "ABANDON EMPTY RECORDING"},
        {"abandon_empty": True, "confirm": "ABANDON EMPTY RECORDING", "delete_files": True},
    ],
)
def test_recording_close_requires_strict_explicit_confirmation_before_contacting_companion(
    tmp_path, payload
):
    connection(tmp_path)
    companion = bridge(tmp_path, lambda _: pytest.fail("Invalid close request reached hardware"))
    with browser(tmp_path, companion) as client:
        assert client.post("/api/v1/robot-lab/recording/finish", json=payload).status_code == 422
    asyncio.run(companion.close())


def test_mapping_observation_route_preserves_run_identity_without_exposing_credentials(tmp_path):
    connection(tmp_path)
    observed = []

    def handler(request):
        assert request.headers["authorization"] == "Bearer " + TOKEN
        observed.append((request.url.path, json.loads(request.content)))
        return httpx.Response(
            200, json={"mapping_checks": {"checks": {"gripper": {"correct": True}}}}
        )

    companion = bridge(tmp_path, handler)
    payload = {
        "port": "/dev/cu.usbmodem123",
        "joint": "gripper",
        "run_id": "completed-run",
        "correct": True,
        "confirm": "CONFIRM PHYSICAL JOINT",
    }
    with browser(tmp_path, companion) as client:
        response = client.post("/api/v1/robot-lab/mapping/confirm", json=payload)
        assert response.status_code == 200
        assert observed == [("/v1/mapping/confirm", payload)]
        assert TOKEN not in response.text
        assert response.headers["cache-control"] == "no-store"
    asyncio.run(companion.close())


@pytest.mark.parametrize(
    "action,confirm",
    [
        ("save", "SAVE REST POSITION"),
        ("return", "RETURN TO REST"),
    ],
)
def test_rest_position_routes_forward_only_the_explicit_action(tmp_path, action, confirm):
    connection(tmp_path)
    seen = []

    def handler(request):
        seen.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"ok": True})

    companion = bridge(tmp_path, handler)
    with browser(tmp_path, companion) as client:
        response = client.post(
            "/api/v1/robot-lab/control/rest/" + action, json={"confirm": confirm}
        )
    assert response.status_code == 200
    assert seen == [("/v1/control/rest/" + action, {"confirm": confirm})]
    asyncio.run(companion.close())


def test_diagnostics_proxy_is_read_only_and_redacts_private_token(tmp_path):
    connection(tmp_path)
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path))
        return httpx.Response(200, json={"enabled": True, "events": [], "message": TOKEN})

    companion = bridge(tmp_path, handler)
    with browser(tmp_path, companion) as client:
        response = client.get("/api/v1/robot-lab/diagnostics")
        assert response.status_code == 200
        assert response.json()["enabled"]
        assert TOKEN not in response.text
    assert calls == [("GET", "/v1/diagnostics")]
    asyncio.run(companion.close())


def test_calibration_transfer_proxy_preserves_profile_destination_and_export_query(tmp_path):
    connection(tmp_path)
    calls = []

    def handler(request):
        calls.append(
            (
                request.method,
                request.url.path,
                str(request.url.query),
                json.loads(request.content) if request.content else None,
            )
        )
        return httpx.Response(200, json={"ok": True})

    companion = bridge(tmp_path, handler)
    payload = {
        "robot_id": "renamed-arm",
        "role": "follower",
        "document": {"example": {}},
        "replace_existing": False,
        "confirm": "IMPORT CALIBRATION",
    }
    with browser(tmp_path, companion) as client:
        assert (
            client.get(
                "/api/v1/robot-lab/calibration/export?robot_id=source&role=follower"
            ).status_code
            == 200
        )
        assert client.post("/api/v1/robot-lab/calibration/import", json=payload).status_code == 200
        assert (
            client.post(
                "/api/v1/robot-lab/calibration/import", json={**payload, "confirm": "yes"}
            ).status_code
            == 422
        )
    assert calls[0][0:2] == ("GET", "/v1/calibration/export")
    assert "robot_id=source" in calls[0][2]
    assert calls[1][0:2] == ("POST", "/v1/calibration/import")
    assert calls[1][3] == payload
    assert len(calls) == 2
    asyncio.run(companion.close())


def test_device_configuration_is_a_validated_finite_proxy_operation(tmp_path):
    connection(tmp_path)
    commands = []

    def handler(request):
        commands.append((request.method, request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"saved_devices": [{"id": "arm-a", "name": "Desk arm"}]})

    companion = bridge(tmp_path, handler)
    with browser(tmp_path, companion) as client:
        payload = {"id": "arm-a", "port": "/dev/ttyACM0", "name": "Desk arm", "role": "leader"}
        response = client.post("/api/v1/robot-lab/devices/configure", json=payload)
        assert response.status_code == 200
        assert response.json()["saved_devices"][0]["name"] == "Desk arm"
        assert commands == [("POST", "/v1/devices/configure", payload)]
        for changes in ({"role": "unknown"}, {"name": "x" * 65}, {"unexpected": True}):
            assert (
                client.post(
                    "/api/v1/robot-lab/devices/configure", json=payload | changes
                ).status_code
                == 422
            )
        assert len(commands) == 1
    asyncio.run(companion.close())


def test_device_removal_and_restore_validate_identity_and_proxy_only_registry_operations(tmp_path):
    connection(tmp_path)
    commands = []

    def handler(request):
        commands.append((request.method, request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"saved_devices": [], "removed_devices": []})

    companion = bridge(tmp_path, handler)
    with browser(tmp_path, companion) as client:
        for action in ("remove", "restore"):
            route = "/api/v1/robot-lab/devices/" + action
            assert client.post(route, json={"id": "arm-a"}).status_code == 200
            for payload in (
                {},
                {"id": ""},
                {"id": "x" * 101},
                {"id": "arm-a", "delete_calibration": True},
            ):
                assert client.post(route, json=payload).status_code == 422
    assert commands == [
        ("POST", "/v1/devices/" + action, {"id": "arm-a"}) for action in ("remove", "restore")
    ]
    asyncio.run(companion.close())


@pytest.mark.parametrize("action", ["start", "pause", "resume", "next", "rerecord"])
def test_guided_recording_actions_are_finite_proxy_routes(tmp_path, action):
    connection(tmp_path)
    calls = []

    def handler(request):
        calls.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"ok": True})

    companion = bridge(tmp_path, handler)
    with browser(tmp_path, companion) as client:
        route = "/api/v1/robot-lab/recording/collection/" + action
        assert client.post(route, json={}).status_code == 200
        assert client.post(route, json={"move_arm": True}).status_code == 422
        assert (
            client.post("/api/v1/robot-lab/recording/collection/arbitrary", json={}).status_code
            == 404
        )
    assert calls == [("/v1/recording/collection/" + action, {})]
    asyncio.run(companion.close())


@pytest.mark.parametrize("fps", [None, 1, 40])
def test_recording_auto_rate_and_guided_configuration_reach_companion(tmp_path, fps):
    connection(tmp_path)
    payload = {
        "dataset_id": "pick-up-pen",
        "task": "Pick up the pen.",
        "fps": fps,
        "camera_index": 0,
        "collection": {
            "episode_time_s": 30,
            "reset_time_s": 10,
            "countdown_s": 3,
            "num_episodes": 10,
        },
    }
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": True})

    companion = bridge(tmp_path, handler)
    with browser(tmp_path, companion) as client:
        assert client.post("/api/v1/robot-lab/recording/start", json=payload).status_code == 200
    assert calls == [payload]
    asyncio.run(companion.close())


@pytest.mark.parametrize(
    "changes",
    [
        {"fps": True},
        {"fps": 41},
        {"fps": 0},
        {"collection": {"episode_time_s": 301}},
        {"collection": {"reset_time_s": -1}},
        {"collection": {"num_episodes": 0}},
        {"collection": {"countdown_s": True}},
    ],
)
def test_invalid_guided_recording_settings_never_contact_hardware(tmp_path, changes):
    connection(tmp_path)
    companion = bridge(tmp_path, lambda _: pytest.fail("Invalid settings reached hardware"))
    with browser(tmp_path, companion) as client:
        assert (
            client.post(
                "/api/v1/robot-lab/recording/start",
                json={"dataset_id": "test", "task": "Pick up the pen", **changes},
            ).status_code
            == 422
        )
    asyncio.run(companion.close())
