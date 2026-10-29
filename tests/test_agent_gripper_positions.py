"""Metric position proposals use existing physical admission and joint execution."""

import math

import pytest
from fastapi.testclient import TestClient
from test_agent_physical_service import FakeRobotTarget, admission, create, wait
from test_local_agent_teleoperation import context, headers
from vla_platform.agent_control import kinematics as k
from vla_platform.api import create_app
from vla_platform.settings import Settings


@pytest.fixture
def physical_position(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        robot = FakeRobotTarget()
        client.app.state.agent_control.robot_target = robot
        session = create(
            client,
            provider_id="local-mcp",
            gripper_position_control=True,
            max_steps=40,
            max_calls=40,
            max_seconds=120,
        )
        route = "/api/v1/agent-sessions/" + session["id"]
        token = client.post(route + "/local/access").json()["access_token"]
        yield client, robot, session, route, token


def request_position(client, route, token, target, nonce):
    assert client.post(route + "/observe").status_code == 200
    assert client.post(route + "/proposals", json={"request_id": nonce}).status_code == 202
    pending = context(client, route, token, pending=True)["planning_request"]
    return client.post(
        route + "/local/result",
        headers=headers(token),
        json={
            **{key: value for key, value in pending.items() if key != "deadline_at"},
            "agent_name": "position-fixture",
            "position": {
                "frame": "base_link",
                "x_m": target[0],
                "y_m": target[1],
                "z_m": target[2],
                "duration_seconds": 2.0,
            },
            "purpose": "Move the gripper to the specified base-frame endpoint",
        },
    )


def test_metric_target_reaches_through_existing_reviewed_joint_segments(physical_position):
    client, robot, session, route, token = physical_position
    target = k.forward(dict.fromkeys(k.JOINTS, 0.0) | {"shoulder_lift": 8.0, "elbow_flex": 4.0})
    assert "gripper_position" in session["supported_skills"]
    assert session["observation"]["gripper_position_estimate"]["frame"] == "base_link"
    for step in range(20):
        result = request_position(client, route, token, target, "position-turn-" + str(step))
        assert result.status_code == 202, result.text
        current = wait(client, session, {"waiting_for_acceptance", "paused", "error"})
        assert current["status"] == "waiting_for_acceptance", current
        proposal = current["pending_proposal"]
        assert proposal["skill"] == "joint_segment"
        assert proposal["gripper_position_plan"]["target_m"] == target
        assert "gripper" not in proposal["arguments"]["targets"]
        executions = len(robot.executions)
        assert (
            client.post(
                route + "/approve",
                json={"proposal_id": proposal["id"], "authority_epoch": current["authority_epoch"]},
            ).status_code
            == 409
        )
        assert len(robot.executions) == executions
        current, approval = admission(client, current)
        assert client.post(route + "/approve", json=approval).status_code == 202
        current = wait(client, session, {"ready", "paused", "error"})
        assert current["status"] == "ready", current
        estimate = current["observation"]["gripper_position_estimate"]
        assert estimate["position_m"] == pytest.approx(k.forward(robot.joints))
        assert robot.joints["gripper"] == 50
        if math.dist(estimate["position_m"], target) <= k.TOLERANCE_M:
            break
    assert math.dist(k.forward(robot.joints), target) <= k.TOLERANCE_M
    assert robot.executions
    assert not current["task_success_verified"]


def test_unreachable_target_preserves_the_pending_request_and_sends_no_movement(physical_position):
    client, robot, session, route, token = physical_position
    response = request_position(client, route, token, [0.6, 0.6, 0.6], "unreachable-position")
    assert response.status_code == 422, response.text
    assert "reachable" in response.json()["detail"]
    assert (
        context(client, route, token, pending=True)["planning_request"]["request_id"]
        == "unreachable-position"
    )
    assert not robot.executions


def test_position_targets_require_explicit_session_configuration(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        robot = FakeRobotTarget()
        client.app.state.agent_control.robot_target = robot
        session = create(client, provider_id="local-mcp")
        route = "/api/v1/agent-sessions/" + session["id"]
        token = client.post(route + "/local/access").json()["access_token"]
        assert "gripper_position" not in context(client, route, token)["action_space"]
        response = request_position(client, route, token, [0.3, 0.0, 0.2], "not-enabled")
        assert response.status_code == 422
        assert not robot.executions


def test_provider_position_result_uses_the_same_host_kinematics_and_ignores_forged_plan(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        robot = FakeRobotTarget()
        client.app.state.agent_control.robot_target = robot
        target = k.forward(dict.fromkeys(k.JOINTS, 0.0) | {"shoulder_lift": 4.0})

        async def planner(_profile, payload):
            assert payload["action_space"]["gripper_position"]["frame"] == "base_link"
            return {
                "status": "proposed",
                "skill": "gripper_position",
                "arguments": {
                    "frame": "base_link",
                    "x_m": target[0],
                    "y_m": target[1],
                    "z_m": target[2],
                    "duration_seconds": 2.0,
                },
                "gripper_position_plan": {"target_m": [0, 0, 0]},
                "purpose": "Known endpoint",
                "model": "provider-position-fixture",
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            }

        client.app.state.agent_control.planner = planner
        session = create(client, gripper_position_control=True)
        route = "/api/v1/agent-sessions/" + session["id"]
        assert (
            client.post(route + "/proposals", json={"request_id": "provider-position"}).status_code
            == 202
        )
        current = wait(client, session, {"waiting_for_acceptance", "paused", "error"})
        assert current["status"] == "waiting_for_acceptance", current
        assert current["pending_proposal"]["gripper_position_plan"]["target_m"] == target
        assert not robot.executions


def test_mcp_position_tool_preserves_the_scoped_request_envelope():
    import asyncio

    from vla_platform.agent_mcp import make_server

    class Client:
        def __init__(self):
            self.calls = []

        async def request(self, method, operation, body):
            self.calls.append((method, operation, body))
            return {"id": "position-session"}

    async def invoke():
        client = Client()
        server = make_server(client)
        await server.call_tool(
            "propose_gripper_position",
            {
                "request_id": "metric-mcp",
                "authority_epoch": 3,
                "observation_id": "admitted-image",
                "observation_sha256": "a" * 64,
                "x_m": 0.3,
                "y_m": 0.0,
                "z_m": 0.2,
                "purpose": "Move to the known endpoint",
            },
        )
        assert len(client.calls) == 1
        method, operation, body = client.calls[0]
        assert (method, operation) == ("POST", "result")
        assert body["position"] == {
            "frame": "base_link",
            "x_m": 0.3,
            "y_m": 0.0,
            "z_m": 0.2,
            "duration_seconds": 2.0,
        }
        assert body["authority_epoch"] == 3
        assert body["observation_id"] == "admitted-image"
        assert body["arguments"] is None

    asyncio.run(invoke())


def test_position_estimate_changes_without_changing_the_action_policy(physical_position):
    client, robot, session, route, token = physical_position
    before = context(client, route, token)
    robot.joints["shoulder_lift"] = 2.0
    assert client.post(route + "/observe").status_code == 200
    after = context(client, route, token)
    assert after["action_space"] == before["action_space"]
    assert (
        after["session"]["observation"]["gripper_position_estimate"]["position_m"]
        != before["session"]["observation"]["gripper_position_estimate"]["position_m"]
    )
    assert not robot.executions
