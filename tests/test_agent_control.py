"""Authority and durable evidence tests without model calls or robot hardware."""

import asyncio
import time
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from vla_platform.api import create_app
from vla_platform.settings import Settings


def wait(client, identity, status):
    for _ in range(200):
        value = client.get("/api/v1/agent-sessions/" + identity).json()
        if value["status"] in status:
            return value
        time.sleep(0.01)
    raise AssertionError(value)


@pytest.fixture
def app(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        yield client


def create(client, **extra):
    project = client.post("/api/v1/projects", json={"name": "agent test"}).json()["id"]
    response = client.post(
        f"/api/v1/projects/{project}/agent-sessions",
        json={"task": "shoulder_pan +0.03 rad", **extra},
    )
    assert response.status_code == 201, response.text
    return response.json()


def propose(client, s, request_id="request1"):
    response = client.post(
        f"/api/v1/agent-sessions/{s['id']}/proposals", json={"request_id": request_id}
    )
    assert response.status_code == 202, response.text
    return wait(client, s["id"], {"waiting_for_acceptance", "error", "complete"})


def test_replay_records_no_applied_training_label_and_duplicate_request_executes_once(app):
    s = create(app)
    s = propose(app, s)
    assert s["status"] == "waiting_for_acceptance", s
    proposal = s["pending_proposal"]
    assert proposal["observation_sha256"] == s["observation"]["sha256"]
    path = f"/api/v1/agent-sessions/{s['id']}"
    approval = {"proposal_id": proposal["id"], "authority_epoch": s["authority_epoch"]}
    assert app.post(path + "/approve", json=approval).status_code == 202
    s = wait(app, s["id"], {"ready", "error"})
    assert s["receipts"][0]["status"] == "replayed"
    assert s["receipts"][0]["applied_targets"] == {}
    assert s["receipts"][0]["task_success_verified"] is False
    assert app.post(path + "/approve", json=approval).status_code == 202
    duplicate = app.post(path + "/proposals", json={"request_id": "request1"}).json()
    assert duplicate["calls_used"] == 1 and len(duplicate["receipts"]) == 1
    events = app.get(path + "/events").json()
    assert [e["id"] for e in events] == sorted(set(e["id"] for e in events))
    assert all(
        e["id"] > events[0]["id"]
        for e in app.get(path + "/events", params={"after": events[0]["id"]}).json()
    )


def test_wrong_epoch_unknown_proposal_and_advisory_never_dispatch(app):
    for mode in ("supervised", "shadow", "advisory"):
        s = propose(app, create(app, mode=mode))
        path = f"/api/v1/agent-sessions/{s['id']}/approve"
        assert (
            app.post(
                path,
                json={
                    "proposal_id": s["pending_proposal"]["id"],
                    "authority_epoch": s["authority_epoch"] + 1,
                },
            ).status_code
            == 409
        )
        assert (
            app.post(
                path, json={"proposal_id": "other", "authority_epoch": s["authority_epoch"]}
            ).status_code
            == 409
        )
        if mode != "supervised":
            assert (
                app.post(
                    path,
                    json={
                        "proposal_id": s["pending_proposal"]["id"],
                        "authority_epoch": s["authority_epoch"],
                    },
                ).status_code
                == 409
            )
        assert app.get(f"/api/v1/agent-sessions/{s['id']}").json()["receipts"] == []


def test_fresh_observation_invalidates_exact_old_proposal_and_continuation(app):
    s = propose(app, create(app))
    path = f"/api/v1/agent-sessions/{s['id']}"
    proposal = s["pending_proposal"]
    fresh = app.post(path + "/observe", json={}).json()
    assert fresh["observation"]["id"] != s["observation"]["id"]
    assert fresh["pending_proposal"] is None
    assert fresh["receipts"][-1]["status"] == "stopped"
    assert (
        app.post(
            path + "/approve",
            json={"proposal_id": proposal["id"], "authority_epoch": s["authority_epoch"]},
        ).status_code
        == 409
    )
    assert propose(app, fresh, "new-request")["status"] == "waiting_for_acceptance"


def test_stop_blocks_delayed_provider_even_if_it_swallows_cancellation(app):
    agent = app.app.state.agent_control
    original = agent.planner

    async def delayed(profile, payload):
        try:
            await asyncio.sleep(2)
        except asyncio.CancelledError:
            pass
        return await original(profile, payload)

    agent.planner = delayed
    s = create(app)
    path = f"/api/v1/agent-sessions/{s['id']}"
    app.post(path + "/proposals", json={"request_id": "late"})
    assert app.get("/api/v1/updates/readiness").status_code == 409
    stopped = app.post(path + "/stop", json={}).json()
    assert stopped["status"] == "stopped"
    assert stopped["authority_epoch"] == s["authority_epoch"] + 1
    assert stopped["pending_proposal"] is None and stopped["receipts"] == []
    assert app.post(path + "/proposals", json={"request_id": "late"}).json()["calls_used"] == 1
    assert app.post(path + "/proposals", json={"request_id": "new"}).status_code == 409
    for _ in range(200):
        events = app.get(path + "/events").json()
        if any(e["kind"] == "late_result_rejected" for e in events):
            break
        time.sleep(0.01)
    assert any(e["kind"] == "late_result_rejected" for e in events)


def test_private_settings_and_validation_never_echo_credentials(app, tmp_path):
    key = "secret_agent_key_123456789"
    response = app.put(
        "/api/v1/agent/providers/astra",
        json={
            "provider": "openai_responses",
            "api_key": key,
            "model": "gpt-6-astra",
            "reasoning_effort": "low",
        },
    )
    assert response.status_code == 200 and key not in response.text
    assert response.json()["checked"] is False
    assert (tmp_path / "agent-providers.json").stat().st_mode & 0o777 == 0o600
    response = app.put(
        "/api/v1/agent/providers/astra",
        json={"provider": "openai_responses", "api_key": key, "model": key, "bad": key},
    )
    assert response.status_code == 422 and key not in response.text
    assert key not in app.get("/api/v1/agent/providers").text
    project = app.post("/api/v1/projects", json={"name": "private"}).json()["id"]
    assert (
        app.post(
            f"/api/v1/projects/{project}/agent-sessions",
            json={"provider_id": "astra", "task": "move", "cloud_consent": True},
        ).status_code
        == 409
    )


def test_budget_exhaustion_and_expired_proposals_are_rejected(app):
    s = create(app, max_total_tokens=1024)
    path = f"/api/v1/agent-sessions/{s['id']}"
    assert app.post(path + "/proposals", json={"request_id": "budget"}).status_code == 409
    assert app.get(path).json()["calls_used"] == 0
    s = propose(app, create(app))

    async def expire():
        service = app.app.state.agent_control
        current, p = await service.load(s["id"])
        current.pending_proposal.expires_at = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
        await service.save(current, p, "test_expired")

    app.portal.call(expire)
    assert (
        app.post(
            f"/api/v1/agent-sessions/{s['id']}/approve",
            json={
                "proposal_id": s["pending_proposal"]["id"],
                "authority_epoch": s["authority_epoch"],
            },
        ).status_code
        == 409
    )


def test_restart_revokes_epoch_and_preserves_pending_identity(app):
    s = propose(app, create(app))
    app.portal.call(app.app.state.agent_control.reconcile)
    value = app.get(f"/api/v1/agent-sessions/{s['id']}").json()
    assert value["status"] == "interrupted" and value["authority_epoch"] == s["authority_epoch"] + 1
    assert value["receipts"][-1]["status"] == "stopped"
    assert value["receipts"][-1]["proposal_id"] == s["pending_proposal"]["id"]
    assert (
        app.post(
            f"/api/v1/agent-sessions/{s['id']}/approve",
            json={
                "proposal_id": s["pending_proposal"]["id"],
                "authority_epoch": s["authority_epoch"],
            },
        ).status_code
        == 409
    )
    assert value["calls_used"] == 1


def test_prior_observation_replay_is_project_scoped(app):
    s = create(app)
    same = app.post(
        f"/api/v1/projects/{s['project_id']}/agent-sessions",
        json={"task": "inspect", "replay_observation_id": s["observation"]["id"]},
    )
    assert same.status_code == 201
    other = app.post("/api/v1/projects", json={"name": "other"}).json()["id"]
    assert (
        app.post(
            f"/api/v1/projects/{other}/agent-sessions",
            json={"task": "inspect", "replay_observation_id": s["observation"]["id"]},
        ).status_code
        == 503
    )
