"""Local planners share physical admission and cannot grant themselves execution authority."""

import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from test_agent_physical_service import FakeRobotTarget, admission, create, wait
from vla_platform.agent_mcp import LocalAgentClient
from vla_platform.api import create_app
from vla_platform.settings import Settings


@pytest.fixture
def local(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        robot = FakeRobotTarget()
        client.app.state.agent_control.robot_target = robot

        async def forbidden(*_args):
            raise AssertionError("A local planner must never invoke the OpenAI worker")

        client.app.state.agent_control.planner = forbidden
        session = create(client, provider_id="local-mcp")
        route = "/api/v1/agent-sessions/" + session["id"]
        access = client.post(route + "/local/access")
        assert access.status_code == 200, access.text
        token = access.json()["access_token"]
        yield client, robot, session, route, token


def headers(token):
    return {"Authorization": "Bearer " + token}


def context(client, route, token, *, pending=False):
    for n in range(100):
        result = client.get(route + "/local/context", headers=headers(token))
        assert result.status_code == 200, result.text
        value = result.json()
        if not pending or value["planning_request"]:
            return value
        if n % 10 == 0:
            client.post(
                route + "/heartbeat",
                json={
                    "authority_epoch": value["session"]["authority_epoch"],
                    "operator_present": True,
                },
            )
        time.sleep(0.01)
    raise AssertionError(value)


def begin(client, route, token):
    assert client.post(route + "/observe").status_code == 200
    assert client.post(route + "/proposals", json={"request_id": "local-turn"}).status_code == 202
    value = context(client, route, token, pending=True)
    request = {k: v for k, v in value["planning_request"].items() if k != "deadline_at"}
    return value, dict(
        request,
        agent_name="Codex local client",
        arguments={"targets": {"gripper": 48.0}, "duration_seconds": 2.0},
        purpose="Small gripper adjustment through the existing position controller",
    )


def test_local_access_is_scoped_private_and_rotation_invalidates_old_credentials(local):
    client, robot, session, route, token = local
    value = context(client, route, token)
    assert value["action_space"]["joint_units"]["gripper"] == "percent"
    assert value["execution_policy"]["human_approval_required"]
    assert token not in json.dumps(value)
    assert token not in client.get(route + "/events").text
    assert token not in client.get(route).text
    assert client.get(route + "/local/context").status_code == 401
    assert client.get(route + "/local/context", headers=headers("B" * 43)).status_code == 401
    replacement = client.post(route + "/local/access").json()["access_token"]
    assert client.get(route + "/local/context", headers=headers(token)).status_code == 401
    assert client.get(route + "/local/context", headers=headers(replacement)).status_code == 200
    assert not robot.executions


def test_local_segment_waits_for_the_same_physical_admission_and_returns_gripper_receipt(local):
    client, robot, session, route, token = local
    value, action = begin(client, route, token)
    image = client.get(
        route + "/local/image",
        params={"observation_id": session["observation"]["id"]},
        headers=headers(token),
    )
    assert image.status_code == 200 and image.headers["content-type"] == "image/jpeg"
    assert image.content.startswith(b"\xff\xd8")
    result = client.post(route + "/local/result", headers=headers(token), json=action)
    assert result.status_code == 202, result.text
    current = wait(client, session, {"waiting_for_acceptance", "paused", "error"})
    assert current["status"] == "waiting_for_acceptance", current
    assert not robot.executions
    proposal = current["pending_proposal"]
    assert proposal["model"] == "Codex local client"
    assert proposal["arguments"]["targets"]["gripper"] == 48.0
    assert current["reserved_tokens"] == 0
    assert (
        client.post(
            route + "/approve",
            json={
                "proposal_id": proposal["id"],
                "authority_epoch": current["authority_epoch"],
            },
        ).status_code
        == 409
    )
    current, approval = admission(client, current)
    assert client.post(route + "/approve", json=approval).status_code == 202
    finished = wait(client, session, {"ready", "paused", "error"})
    assert len(robot.executions) == 1, finished
    receipt = finished["receipts"][-1]
    assert receipt["status"] == "completed", finished
    assert receipt["joint_units"]["gripper"] == "percent"
    assert receipt["applied_targets"]["gripper"] == 48.0
    assert not receipt["task_success_verified"]
    duplicate = client.post(route + "/local/result", headers=headers(token), json=action)
    assert duplicate.status_code == 202
    assert len(robot.executions) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("authority_epoch", 999),
        ("observation_id", "old-observation"),
        ("observation_sha256", "f" * 64),
    ],
)
def test_stale_local_context_never_creates_a_proposal(local, field, value):
    client, robot, session, route, token = local
    _, action = begin(client, route, token)
    action[field] = value
    assert (
        client.post(route + "/local/result", headers=headers(token), json=action).status_code == 409
    )
    assert client.get(route).json()["pending_proposal"] is None
    assert not robot.executions


def test_revocation_cancels_planning_and_stop_revokes_access(local):
    client, robot, session, route, token = local
    _, action = begin(client, route, token)
    revoked = client.delete(route + "/local/access")
    assert revoked.status_code == 200, revoked.text
    assert revoked.json()["status"] == "paused" and revoked.json()["control_armed"]
    assert (
        client.post(route + "/local/result", headers=headers(token), json=action).status_code == 401
    )
    new_token = client.post(route + "/local/access").json()["access_token"]
    stopped = client.post(route + "/local/stop", headers=headers(new_token), json={})
    assert stopped.status_code == 200, stopped.text
    assert stopped.json()["status"] == "stopped"
    assert robot.releases
    assert client.get(route + "/local/context", headers=headers(new_token)).status_code == 403
    assert not robot.executions


def test_local_assessment_is_not_a_verified_task_result(local):
    client, robot, session, route, token = local
    _, action = begin(client, route, token)
    action.pop("arguments")
    action["assessment"] = {"outcome": "achieved", "summary": "The pen appears lifted."}
    assert (
        client.post(route + "/local/result", headers=headers(token), json=action).status_code == 202
    )
    current = wait(client, session, {"paused", "error"})
    assert current["model_assessment"]["outcome"] == "achieved"
    assert not current["task_success_verified"]
    assert not robot.executions


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "http://127.0.0.1.evil.example",
        "http://user:secret@localhost:8000",
        "http://localhost:8000/path",
        "http://localhost:8000?token=secret",
    ],
)
def test_stdio_client_cannot_send_session_credentials_to_external_urls(url):
    with pytest.raises(ValueError):
        LocalAgentClient(url, "fixture", "A" * 43)


def test_stdio_transport_exposes_only_scoped_tools():
    async def run():
        source = str(Path(__file__).resolve().parents[1] / "packages/core/src")
        server = StdioServerParameters(
            command=sys.executable,
            args=["-m", "vla_platform.agent_mcp", "--session-id", "fixture"],
            env={"OPENJENSEN_AGENT_TOKEN": "A" * 43, "PYTHONPATH": source},
        )
        async with stdio_client(server) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                assert {tool.name for tool in result.tools} == {
                    "get_context",
                    "get_camera_image",
                    "propose_joint_segment",
                    "propose_gripper_position",
                    "report_assessment",
                    "stop_robot",
                }
                assert all(
                    "api_url" not in tool.inputSchema.get("properties", {}) for tool in result.tools
                )

    asyncio.run(run())


def test_stdio_mutation_is_sent_once_and_redirects_do_not_receive_credentials():
    calls = []

    def reply(request):
        calls.append(request)
        return httpx.Response(302, headers={"location": "https://example.com"})

    client = LocalAgentClient(
        "http://127.0.0.1:8000", "fixture", "A" * 43, transport=httpx.MockTransport(reply)
    )
    with pytest.raises(ValueError) as failure:
        asyncio.run(client.request("POST", "stop", {}))
    assert len(calls) == 1
    assert "A" * 43 not in str(failure.value)


def test_mcp_stdio_round_trip_uses_the_existing_supervised_gripper_executor(local):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    client, robot, session, route, token = local
    done = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def relay(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            response = client.request(
                self.command,
                self.path,
                content=body,
                headers={
                    "Authorization": self.headers.get("Authorization", ""),
                    "Content-Type": self.headers.get("Content-Type", "application/json"),
                },
            )
            self.send_response(response.status_code)
            self.send_header(
                "Content-Type", response.headers.get("content-type", "application/json")
            )
            self.end_headers()
            self.wfile.write(response.content)

        do_GET = relay
        do_POST = relay

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def keep_operator_present():
        while not done.wait(0.2):
            response = client.post(
                route + "/heartbeat",
                json={
                    "authority_epoch": session["authority_epoch"],
                    "operator_present": True,
                },
            )
            if response.status_code != 200:
                return

    heartbeat = threading.Thread(target=keep_operator_present, daemon=True)
    heartbeat.start()

    def content(result):
        assert not result.isError, result
        return result.structuredContent or json.loads(result.content[0].text)

    async def run():
        source = str(Path(__file__).resolve().parents[1] / "packages/core/src")
        parameters = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "vla_platform.cli",
                "agent-mcp",
                "--session-id",
                session["id"],
                "--api-url",
                f"http://127.0.0.1:{server.server_port}",
            ],
            env={"OPENJENSEN_AGENT_TOKEN": token, "PYTHONPATH": source},
        )
        async with stdio_client(parameters) as (read, write):
            async with ClientSession(read, write) as mcp:
                await mcp.initialize()
                current = content(await mcp.call_tool("get_context", {}))
                assert current["session"]["provider_id"] == "local-mcp"
                assert current["action_space"]["joint_units"]["gripper"] == "percent"
                image = await mcp.call_tool(
                    "get_camera_image",
                    {
                        "observation_id": current["session"]["observation"]["id"],
                    },
                )
                assert not image.isError
                assert any(block.type == "image" for block in image.content)
                assert client.post(route + "/observe").status_code == 200
                assert (
                    client.post(route + "/proposals", json={"request_id": "stdio-turn"}).status_code
                    == 202
                )
                for _ in range(100):
                    current = content(await mcp.call_tool("get_context", {}))
                    if current["planning_request"]:
                        break
                    await asyncio.sleep(0.01)
                action = {
                    k: v for k, v in current["planning_request"].items() if k != "deadline_at"
                }
                action.update(
                    targets={"gripper": 48.0},
                    duration_seconds=2.0,
                    purpose="Bounded gripper adjustment",
                    agent_name="Claude local client",
                )
                submitted = content(await mcp.call_tool("propose_joint_segment", action))
                assert submitted["received_for_validation"]
                proposed = wait(client, session, {"waiting_for_acceptance", "paused", "error"})
                assert proposed["status"] == "waiting_for_acceptance", proposed
                assert not robot.executions
                _, approval = admission(client, proposed)
                assert client.post(route + "/approve", json=approval).status_code == 202
                finished = wait(client, session, {"ready", "paused", "error"})
                assert finished["receipts"][-1]["applied_targets"]["gripper"] == 48.0
                assert len(robot.executions) == 1
                stopped = content(await mcp.call_tool("stop_robot", {}))
                assert stopped["status"] == "stopped"

    try:
        asyncio.run(run())
    finally:
        done.set()
        heartbeat.join(timeout=3)
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_external_camera_reads_do_not_renew_operator_presence(local):
    client, robot, session, route, token = local
    deadline = time.monotonic() + 5
    response = None
    while time.monotonic() < deadline:
        response = client.get(route + "/local/context", headers=headers(token))
        if response.status_code == 403:
            break
        assert response.status_code == 200, response.text
        time.sleep(0.1)
    assert response.status_code == 403
    assert not client.get(route).json()["control_armed"]
    # Revocation denies access before asynchronous target release completes.
    # Reading the session immediately can still observe that paused transition.
    current = wait(client, session, {"interrupted", "error"}, presence=False)
    assert current["status"] == "interrupted" and not current["control_armed"]
    assert robot.releases and not robot.executions


def test_operator_expiry_revokes_authority_before_target_release_finishes(local, monkeypatch):
    client, robot, session, route, _token = local
    entered, gate = asyncio.Event(), asyncio.Event()
    release = robot.release

    async def delayed_release(*args):
        entered.set()
        await gate.wait()
        await release(*args)

    async def wait_for_release():
        await asyncio.wait_for(entered.wait(), timeout=5)

    monkeypatch.setattr(robot, "release", delayed_release)
    try:
        # Only the operator page can renew presence; no heartbeat is sent here.
        client.portal.call(wait_for_release)
        current = client.get(route).json()
        assert current["status"] == "paused" and not current["control_armed"]
        assert robot.owner == session["id"]
        assert not robot.releases and not robot.executions
    finally:
        client.portal.call(gate.set)
    current = wait(client, session, {"interrupted", "error"}, presence=False)
    assert current["status"] == "interrupted" and not current["control_armed"]
    assert robot.owner is None and robot.releases and not robot.executions


def test_foreign_sites_cannot_create_local_agent_credentials(local):
    client, robot, session, route, token = local
    response = client.post(route + "/local/access", headers={"Origin": "https://example.com"})
    assert response.status_code == 403
    assert token not in response.text
    assert not robot.executions


def test_local_result_rejects_unknown_joints_and_conflicting_retries(local):
    client, robot, session, route, token = local
    _, action = begin(client, route, token)
    wrong = dict(action, arguments={"targets": {"base_motor_packet": 1.0}, "duration_seconds": 2.0})
    assert (
        client.post(route + "/local/result", headers=headers(token), json=wrong).status_code == 422
    )
    assert (
        client.post(route + "/local/result", headers=headers(token), json=action).status_code == 202
    )
    assert (
        client.post(route + "/local/result", headers=headers(token), json=action).status_code == 202
    )
    different = dict(action, arguments={"targets": {"gripper": 49.0}, "duration_seconds": 2.0})
    assert (
        client.post(route + "/local/result", headers=headers(token), json=different).status_code
        == 409
    )
    assert not robot.executions


def test_local_task_start_refreshes_the_observation_after_agent_startup(local):
    client, robot, session, route, token = local
    original = session["observation"]["id"]
    for _ in range(12):
        client.post(
            route + "/heartbeat",
            json={
                "authority_epoch": session["authority_epoch"],
                "operator_present": True,
            },
        )
        time.sleep(0.1)
    assert client.post(route + "/run").status_code == 202
    current = context(client, route, token, pending=True)
    assert current["session"]["observation"]["id"] != original
    assert current["planning_request"]["observation_id"] == current["session"]["observation"]["id"]
    assert not robot.executions


def test_local_launcher_pins_the_running_application_revision_without_new_discovery(
    local, monkeypatch
):
    from vla_platform import application_updates

    client, _, _, route, _ = local

    def unexpected_probe(*_args):
        raise AssertionError(
            "Local connection setup must not rediscover the application's revision"
        )

    monkeypatch.setattr(application_updates, "installed_commit", unexpected_probe)
    access = client.post(route + "/local/access")
    assert access.status_code == 200, access.text
    assert access.json()["source_commit"] == client.app.state.application_updates.installed.commit
