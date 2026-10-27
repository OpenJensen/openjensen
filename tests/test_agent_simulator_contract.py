"""The task supervisor obeys the real simulator admission and duration contracts."""

import hashlib

import pytest
from test_agent_task_loop import create, wait
from test_agent_task_loop import target as task_target
from test_teaching_agent_protocol import canonical, observation, rehash_observation
from vla_platform.teaching_api import public_agent_observation, validate_agent_request

target = task_target


@pytest.mark.parametrize("clock", ["stepped", "continuous"])
def test_acquire_passes_real_relay_contract_and_only_extends_frozen_clock(target, clock):
    client, executor = target
    relay = executor.agent_relay

    async def validated(project, job, path, payload=None):
        if payload is not None:
            validate_agent_request(path, payload)
        return await relay(project, job, path, payload)

    executor.agent_relay = validated
    session = create(client, clock_policy=clock)
    assert session["status"] == "ready"
    acquire = next(
        body["arguments"] for operation, body in executor.commands if operation == "acquire"
    )
    assert ("planning_snapshot_seconds" in acquire) is (clock == "stepped")
    if clock == "stepped":
        assert acquire["planning_snapshot_seconds"] == min(120, session["max_seconds"])
    assert client.post("/api/v1/agent-sessions/" + session["id"] + "/stop").status_code == 200
    assert executor.owner is None


@pytest.mark.parametrize("maximum", [None, 1.0, 1.5])
def test_model_manifest_uses_hash_bound_local_duration_limit(target, maximum):
    client, executor = target
    native_observe = executor.observation
    records = []

    def observed():
        raw = native_observe()
        if maximum is not None:
            raw["action_limits"] = {
                "min_duration_seconds": 0.1,
                "max_duration_seconds": maximum,
                "max_ticks": 60,
            }
            del raw["observation_sha256"]
            raw["observation_sha256"] = hashlib.sha256(canonical(raw)).hexdigest()
        public_agent_observation(raw, session_id="executor_fixture")
        return raw

    async def planner(_profile, payload):
        records.append(payload)
        return {
            "status": "assessed",
            "assessment": {"outcome": "needs_help", "summary": "No action"},
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "continuation": None,
        }

    executor.observation = observed
    client.app.state.agent_control.planner = planner
    session = create(client, mode="bounded")
    client.post("/api/v1/agent-sessions/" + session["id"] + "/run")
    wait(client, session, lambda value: value["model_assessment"] is not None)
    space = records[0]["action_space"]
    assert space["max_duration_seconds"] == (2 if maximum is None else maximum)
    assert 0.2 <= space["min_duration_seconds"] <= space["max_duration_seconds"]
    assert executor.executions == []


def test_native_duration_limits_are_strict_and_original_hash_bound():
    value = rehash_observation(
        observation()
        | {
            "action_limits": {
                "min_duration_seconds": 0.1,
                "max_duration_seconds": 1.0,
                "max_ticks": 60,
            }
        }
    )
    assert public_agent_observation(value, session_id="executor_fixture") == value
    for changed in (
        {"max_ticks": 61},
        {"max_ticks": True},
        {"max_duration_seconds": 2.1},
        {"min_duration_seconds": 1.1},
        {"min_duration_seconds": 0.01},
    ):
        body = rehash_observation(value | {"action_limits": value["action_limits"] | changed})
        with pytest.raises(ValueError, match="duration limits"):
            public_agent_observation(body, session_id="executor_fixture")
    with pytest.raises(ValueError, match="fingerprint"):
        public_agent_observation(
            value | {"action_limits": value["action_limits"] | {"max_duration_seconds": 1.5}},
            session_id="executor_fixture",
        )
