"""Task budgets and provider accounting survive final actions, failure and Stop."""

import asyncio
import threading
import time

import pytest
from test_agent_task_loop import create, wait
from test_agent_task_loop import target as task_target

target = task_target


def proposal(**changes):
    return {
        "status": "proposed",
        "skill": "joint_correction",
        "arguments": {"joint": "shoulder_pan", "delta_rad": 0.03},
        "purpose": "One locally bounded approach",
        "model": "deterministic-test",
        "usage": {"input_tokens": 12, "output_tokens": 8},
        "continuation": None,
        **changes,
    }


def assessment(outcome="needs_help", **changes):
    return {
        "status": "assessed",
        "assessment": {"outcome": outcome, "summary": "Needs a better grounded view"},
        "model": "deterministic-test",
        "usage": {"input_tokens": 7, "output_tokens": 3},
        "continuation": None,
        **changes,
    }


def event(client, path, kind):
    for _ in range(300):
        events = client.get(path + "/events").json()
        found = [value for value in events if value["kind"] == kind]
        if found:
            return found
        time.sleep(0.01)
    raise AssertionError(events)


def test_final_reserved_call_completes_actual_action_and_post_observation(target):
    client, executor = target
    records = []

    async def planner(_profile, payload):
        records.append(payload)
        return proposal()

    client.app.state.agent_control.planner = planner
    session = create(client, mode="bounded", max_calls=1)
    path = "/api/v1/agent-sessions/" + session["id"]
    assert client.post(path + "/run").status_code == 202
    done = wait(
        client, session, lambda value: value["status"] == "paused" and value["step_count"] == 1
    )
    assert len(records) == 1 and len(executor.executions) == 1
    assert done["calls_used"] == 1 and not done["run_enabled"]
    receipt = done["receipts"][0]
    assert receipt["status"] == "completed" and receipt["post_observation_id"]
    assert receipt["post_observation_id"] != receipt["pre_observation_id"]
    assert done["observation"]["id"] == receipt["post_observation_id"]
    assert done["observation"]["joints"]["shoulder_pan"] == pytest.approx(0.01)
    assert done["input_tokens"] == 12 and done["output_tokens"] == 8
    assert done["reserved_tokens"] == 0 and not done["task_success_verified"]


def test_next_turn_keeps_exact_continuation_actual_receipt_and_new_observation(target):
    client, executor = target
    records = []
    context = {"pending_call_id": "original_provider_call", "private_context": ["opaque"]}

    async def planner(_profile, payload):
        records.append(payload)
        return proposal(continuation=context) if len(records) == 1 else assessment()

    client.app.state.agent_control.planner = planner
    session = create(client, mode="bounded")
    path = "/api/v1/agent-sessions/" + session["id"]
    client.post(path + "/run")
    done = wait(client, session, lambda value: value["model_assessment"] is not None)
    assert len(records) == 2 and len(executor.executions) == 1
    assert records[1]["operation"] == "continue" and records[1]["continuation"] == context
    assert records[1]["receipt"] == done["receipts"][0]
    assert records[1]["receipt"]["applied_targets"]["shoulder_pan"] == pytest.approx(0.01)
    assert records[1]["observation"]["id"] != records[0]["observation"]["id"]
    assert done["input_tokens"] == 19 and done["output_tokens"] == 11


def test_stop_returns_before_uncooperative_provider_and_late_known_usage_counts_once(target):
    client, executor = target
    started, cancelled, deliver = threading.Event(), threading.Event(), threading.Event()
    records = []

    async def planner(_profile, payload):
        records.append(payload)
        started.set()
        try:
            while not deliver.is_set():
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            cancelled.set()
        while not deliver.is_set():
            await asyncio.sleep(0.01)
        return proposal()

    client.app.state.agent_control.planner = planner
    session = create(client, mode="bounded")
    path = "/api/v1/agent-sessions/" + session["id"]
    try:
        assert client.post(path + "/proposals", json={"request_id": "one_call"}).status_code == 202
        assert started.wait(1)
        stopped = client.post(path + "/stop").json()
        assert not deliver.is_set() and cancelled.is_set()
        assert (
            stopped["status"] == "stopped"
            and stopped["authority_epoch"] > session["authority_epoch"]
        )
        assert executor.owner is None and executor.executions == []
        assert stopped["reserved_tokens"] > 0 and stopped["input_tokens"] == 0
    finally:
        deliver.set()
    event(client, path, "late_result_rejected")
    done = client.get(path).json()
    assert done["status"] == "stopped" and done["pending_proposal"] is None
    assert (
        done["input_tokens"] == 12 and done["output_tokens"] == 8 and done["reserved_tokens"] == 0
    )
    assert len(event(client, path, "provider_usage")) == 1
    repeated = client.post(path + "/proposals", json={"request_id": "one_call"}).json()
    assert repeated["calls_used"] == 1 and len(records) == 1
    assert repeated["input_tokens"] == 12 and repeated["output_tokens"] == 8
    assert executor.executions == []


@pytest.mark.parametrize(
    "result",
    [
        {"status": "error", "usage": {"input_tokens": 19, "output_tokens": 11}},
        proposal(
            arguments={"joint": "not_a_joint", "delta_rad": 0.03},
            usage={"input_tokens": 19, "output_tokens": 11},
        ),
    ],
)
def test_error_or_malformed_proposal_keeps_known_usage_and_releases_reservation(target, result):
    client, executor = target

    async def planner(_profile, _payload):
        return result

    client.app.state.agent_control.planner = planner
    session = create(client, mode="bounded")
    path = "/api/v1/agent-sessions/" + session["id"]
    client.post(path + "/run")
    done = wait(client, session, lambda value: value["status"] == "error")
    assert done["input_tokens"] == 19 and done["output_tokens"] == 11
    assert done["reserved_tokens"] == 0 and done["calls_used"] == 1
    assert done["pending_proposal"] is None and done["receipts"] == []
    assert executor.executions == [] and len(event(client, path, "provider_usage")) == 1


@pytest.mark.parametrize(
    "usage",
    [None, {"input_tokens": True, "output_tokens": 3}, {"input_tokens": -1, "output_tokens": 3}],
)
def test_missing_or_invalid_usage_keeps_outstanding_billing_reservation(target, usage):
    client, executor = target

    async def planner(_profile, _payload):
        return assessment(usage=usage)

    client.app.state.agent_control.planner = planner
    session = create(client, mode="bounded")
    path = "/api/v1/agent-sessions/" + session["id"]
    client.post(path + "/run")
    done = wait(client, session, lambda value: value["status"] in {"paused", "error"})
    assert done["reserved_tokens"] > 0 and done["calls_used"] == 1
    assert done["input_tokens"] == done["output_tokens"] == 0
    assert executor.executions == [] and done["pending_proposal"] is None
    assert not any(
        value["kind"] == "provider_usage" for value in client.get(path + "/events").json()
    )


def test_help_assessment_resumes_with_new_plan_without_finish_tool_continuation(target):
    client, executor = target
    records = []

    async def planner(_profile, payload):
        records.append(payload)
        return assessment(continuation={"pending_call_id": "finish_call_that_must_be_cleared"})

    client.app.state.agent_control.planner = planner
    session = create(client, mode="bounded")
    path = "/api/v1/agent-sessions/" + session["id"]
    for calls in (1, 2):
        assert client.post(path + "/run").status_code == 202
        done = wait(
            client,
            session,
            lambda value: value["status"] == "paused" and value["calls_used"] == calls,
        )
        assert not done["run_enabled"] and not done["task_success_verified"]
    assert [record["operation"] for record in records] == ["plan", "plan"]
    assert all("continuation" not in record and "receipt" not in record for record in records)
    assert (
        done["input_tokens"] == 14 and done["output_tokens"] == 6 and done["reserved_tokens"] == 0
    )
    assert executor.executions == [] and done["receipts"] == []
