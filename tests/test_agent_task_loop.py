"""Abstract goals produce bounded measured actions; assessments never verify tasks."""

import time

import pytest
from fastapi.testclient import TestClient
from test_agent_simulation_service import FakeTeaching
from vla_platform.api import create_app
from vla_platform.settings import Settings


@pytest.fixture
def target(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        executor = FakeTeaching()
        client.app.state.agent_control.execution.teaching = executor
        yield client, executor


def create(client, **kwargs):
    project = client.post("/api/v1/projects", json={"name": "abstract goal protocol"}).json()["id"]
    response = client.post(
        f"/api/v1/projects/{project}/agent-sessions",
        json={
            "task": "Pick up the blue object",
            "target_kind": "teaching",
            "teaching_job_id": "job_fixture",
            "clock_policy": "stepped",
            "planning_mode": "task",
            "max_calls": 8,
            "max_total_tokens": 100000,
            **kwargs,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def wait(client, s, predicate):
    for _ in range(400):
        value = client.get("/api/v1/agent-sessions/" + s["id"]).json()
        if predicate(value):
            return value
        time.sleep(0.01)
    raise AssertionError(value)


def planner(records):
    async def decide(profile, payload):
        records.append(payload)
        if payload["completed_steps"] >= 2:
            return {
                "status": "assessed",
                "assessment": {
                    "outcome": "achieved",
                    "summary": "The object appears lifted in the supplied evidence.",
                },
                "model": "deterministic-test",
                "usage": {"input_tokens": 12, "output_tokens": 8},
                "continuation": None,
            }
        return {
            "status": "proposed",
            "skill": "joint_correction",
            "arguments": {"joint": "shoulder_pan", "delta_rad": 0.03},
            "purpose": "Approach the grounded target by one bounded correction.",
            "model": "deterministic-test",
            "usage": {"input_tokens": 12, "output_tokens": 8},
            "continuation": None,
        }

    return decide


def test_bounded_abstract_goal_loop_observes_after_each_action_and_keeps_model_success_unverified(
    target,
):
    client, executor = target
    records = []
    client.app.state.agent_control.planner = planner(records)
    s = create(client, mode="bounded")
    path = "/api/v1/agent-sessions/" + s["id"]
    assert client.post(path + "/run").status_code == 202
    done = wait(client, s, lambda x: x["status"] == "complete")
    assert done["step_count"] == 2 and done["calls_used"] == 3
    assert done["model_assessment"]["outcome"] == "achieved"
    assert done["task_success_verified"] is False and not done["run_enabled"]
    assert len(executor.executions) == 2 and executor.release_count == 1
    assert [p["completed_steps"] for p in records] == [0, 1, 2]
    assert [p["observation"]["joints"]["shoulder_pan"] for p in records] == [0.0, 0.01, 0.02]
    assert all(
        p["task"] == "Pick up the blue object" and p["planning_mode"] == "task" for p in records
    )
    assert len({p["observation"]["id"] for p in records}) == 3
    assert all(r["status"] == "completed" for r in done["receipts"])
    assert done["input_tokens"] == 36 and done["output_tokens"] == 24


def test_supervised_task_loop_waits_for_each_exact_approval_and_stop_rejects_later_turns(target):
    client, executor = target
    records = []
    client.app.state.agent_control.planner = planner(records)
    s = create(client, mode="supervised")
    path = "/api/v1/agent-sessions/" + s["id"]
    client.post(path + "/run")
    first = wait(client, s, lambda x: x["status"] == "waiting_for_acceptance")
    time.sleep(0.15)
    assert len(executor.executions) == 0 and len(records) == 1
    assert (
        client.post(
            path + "/approve",
            json={
                "proposal_id": first["pending_proposal"]["id"],
                "authority_epoch": first["authority_epoch"],
            },
        ).status_code
        == 202
    )
    second = wait(
        client, s, lambda x: x["step_count"] == 1 and x["status"] == "waiting_for_acceptance"
    )
    assert len(executor.executions) == 1 and len(records) == 2
    assert second["pending_proposal"]["id"] != first["pending_proposal"]["id"]
    stopped = client.post(path + "/stop").json()
    assert stopped["status"] == "stopped" and not stopped["run_enabled"]
    time.sleep(0.1)
    assert len(executor.executions) == 1 and len(records) == 2
    assert (
        client.post(
            path + "/approve",
            json={
                "proposal_id": second["pending_proposal"]["id"],
                "authority_epoch": second["authority_epoch"],
            },
        ).status_code
        == 409
    )


def test_step_budget_cannot_be_bypassed_by_manual_plan_endpoint(target):
    client, executor = target
    client.app.state.agent_control.planner = planner([])
    s = create(client, mode="bounded", max_steps=1)
    path = "/api/v1/agent-sessions/" + s["id"]
    client.post(path + "/run")
    paused = wait(client, s, lambda x: x["status"] == "paused" and x["step_count"] == 1)
    assert not paused["run_enabled"] and len(executor.executions) == 1
    assert client.post(path + "/proposals", json={"request_id": "bypass"}).status_code == 409
    assert client.get(path).json()["step_count"] == 1


def test_needs_help_does_not_dispatch_or_mark_task_success(target):
    client, executor = target

    async def help_needed(profile, payload):
        return {
            "status": "assessed",
            "assessment": {
                "outcome": "needs_help",
                "summary": "The target object is not grounded in this view.",
            },
            "model": "test",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "continuation": None,
        }

    client.app.state.agent_control.planner = help_needed
    s = create(client, mode="bounded")
    path = "/api/v1/agent-sessions/" + s["id"]
    client.post(path + "/run")
    value = wait(client, s, lambda x: x["model_assessment"] is not None)
    assert (
        value["status"] == "paused"
        and not value["run_enabled"]
        and not value["task_success_verified"]
    )
    assert value["receipts"] == [] and executor.executions == []
