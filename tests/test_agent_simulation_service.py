"""Core sessions exercise the simulator mailbox contract with no physics or hardware."""

import asyncio
import base64
import hashlib
import json
import struct
import time
import zlib
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from vla_platform.api import create_app
from vla_platform.settings import Settings


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def png():
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\0\xff\xff\xff"))
        + chunk(b"IEND", b"")
    )


class FakeTeaching:
    """Exact queued/executing/acknowledged protocol, including nested execution."""

    def __init__(self):
        self.owner = None
        self.epoch = 0
        self.step = 0
        self.revision = 2
        self.measured = 0.0
        self.commands = []
        self.receipts = {}
        self.executions = []
        self.observation_count = 0
        self.release_count = 0
        self.age_ms = 0
        self.lose_post_ack = False
        self.lose_read_ack = False

    def state(self):
        return {
            "session_id": "executor_fixture",
            "episode_id": "episode_fixture",
            "authority_epoch": self.epoch,
            "agent_owner": self.owner,
            "joints": ["shoulder_pan"],
            "state_rad": [self.measured],
            "revision": self.revision,
            "mode": "running",
            "agent_clock_policy": "stepped",
        }

    async def relay(self, project, job, path):
        assert path == "/state"
        return self.state()

    def observation(self):
        self.observation_count += 1
        image = png()
        value = {
            "schema_version": 1,
            "observation_id": f"observation_{self.observation_count}",
            "target_session_id": "executor_fixture",
            "agent_session_id": self.owner,
            "episode_id": "episode_fixture",
            "authority_epoch": self.epoch,
            "revision": self.revision,
            "embodiment_manifest_id": "a" * 64,
            "scene_sha256": "b" * 64,
            "calibration_id": None,
            "clock_policy": "stepped",
            "captured_monotonic_ns": 1,
            "valid_until_monotonic_ns": 30_000_000_001,
            "source_age_ms": 0,
            "step": self.step,
            "sim_time": self.step / 60,
            "joints": ["shoulder_pan"],
            "state_rad": [self.measured],
            "units": "rad",
            "image_data_url": "data:image/png;base64," + base64.b64encode(image).decode(),
            "image_sha256": hashlib.sha256(image).hexdigest(),
        }
        value["observation_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
        return value

    async def agent_relay(self, project, job, path, payload=None):
        if payload is None:
            identity = path.rsplit("/", 1)[1]
            value = self.receipts[identity]
            if value.get("execution") and self.lose_read_ack:
                raise OSError("Acknowledgement unavailable")
            return value
        operation = path.rsplit("/", 1)[1]
        self.commands.append((operation, payload))
        identity = payload["command_id"]
        assert identity not in self.receipts, "A dispatch identity must never be posted twice"
        if operation == "acquire":
            assert self.owner is None and payload["authority_epoch"] == 0
            self.owner = payload["agent_session_id"]
            self.epoch += 1
        else:
            assert payload["agent_session_id"] == self.owner
            assert payload["authority_epoch"] == self.epoch
        value = {"command_id": identity, "status": "acknowledged", "state": self.state()}
        if operation == "observe":
            value.update(observation=self.observation(), observation_age_ms=self.age_ms)
        elif operation == "execute":
            args = payload["arguments"]
            source = next(
                receipt["observation"]
                for receipt in self.receipts.values()
                if receipt.get("observation", {}).get("observation_id") == args["observation_id"]
            )
            assert args["observation_sha256"] == source["observation_sha256"]
            self.executions.append(identity)
            self.measured += 0.01  # Local speed limiting applies less than requested +0.03.
            self.step += 1
            self.revision += 1
            value["state"] = self.state()
            value["execution"] = {
                "status": "completed",
                "execution_id": identity,
                "observation_id": args["observation_id"],
                "started_monotonic_ns": 2,
                "completed_monotonic_ns": 3,
                "applied_target_rad": [self.measured],
                "requested_target_rad": [source["state_rad"][0] + args["delta_rad"]],
                "next_state_rad": [self.measured],
                "joint_limit_clipped": [False],
                "speed_limit_clipped": [True],
                "task_success_verified": False,
            }
        elif operation == "release":
            self.release_count += 1
            self.owner = None
            self.epoch += 1
            value["state"] = self.state()
        self.receipts[identity] = value
        if operation == "execute" and self.lose_post_ack:
            raise OSError("Dispatch accepted but acknowledgement lost")
        return {
            "command_id": identity,
            "status": "executing" if operation == "execute" else "queued",
        }


@pytest.fixture
def app(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        teaching = FakeTeaching()
        client.app.state.agent_control.execution.teaching = teaching
        yield client, teaching


def wait(client, identity, statuses):
    for _ in range(200):
        value = client.get(f"/api/v1/agent-sessions/{identity}").json()
        if value["status"] in statuses:
            return value
        time.sleep(0.01)
    raise AssertionError(value)


def create(client):
    project = client.post("/api/v1/projects", json={"name": "simulation protocol"}).json()["id"]
    response = client.post(
        f"/api/v1/projects/{project}/agent-sessions",
        json={
            "task": "shoulder_pan +0.03 rad",
            "target_kind": "teaching",
            "teaching_job_id": "job_fixture",
            "clock_policy": "stepped",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def propose(client, session, request_id="request1"):
    response = client.post(
        f"/api/v1/agent-sessions/{session['id']}/proposals", json={"request_id": request_id}
    )
    assert response.status_code == 202, response.text
    result = wait(client, session["id"], {"waiting_for_acceptance", "complete", "error"})
    return result


def approve(client, session):
    return client.post(
        f"/api/v1/agent-sessions/{session['id']}/approve",
        json={
            "proposal_id": session["pending_proposal"]["id"],
            "authority_epoch": session["authority_epoch"],
        },
    )


def test_full_simulator_loop_records_applied_action_and_stop_releases_owner(app):
    client, target = app
    session = create(client)
    assert target.owner == session["id"]
    assert session["observation"]["environment_kind"] == "simulation"
    assert session["observation"]["joints"] == {"shoulder_pan": 0.0}
    session = propose(client, session)
    assert session["status"] == "waiting_for_acceptance", session
    assert approve(client, session).status_code == 202
    complete = wait(client, session["id"], {"ready", "error"})
    assert complete["status"] == "ready", complete
    receipt = complete["receipts"][-1]
    assert receipt["status"] == "completed"
    assert receipt["arguments"]["delta_rad"] == 0.03
    assert receipt["applied_targets"] == {"shoulder_pan": 0.01}
    assert receipt["task_success_verified"] is False
    assert receipt["post_observation_id"] != receipt["pre_observation_id"]
    assert complete["observation"]["joints"] == {"shoulder_pan": 0.01}
    assert len(target.executions) == 1
    assert approve(client, session).status_code == 202
    assert len(target.executions) == 1
    stopped = client.post(f"/api/v1/agent-sessions/{session['id']}/stop", json={}).json()
    assert stopped["status"] == "stopped"
    assert target.release_count == 1 and target.owner is None


@pytest.mark.parametrize("lose_read", [False, True])
def test_lost_ack_reconciles_exact_identity_or_preserves_unknown_without_retry(app, lose_read):
    client, target = app
    target.lose_post_ack, target.lose_read_ack = True, lose_read
    session = propose(client, create(client))
    assert approve(client, session).status_code == 202
    outcome = wait(client, session["id"], {"ready", "error"})
    assert len(target.executions) == 1
    receipt = outcome["receipts"][-1]
    assert receipt["id"] == target.executions[0]
    assert receipt["status"] == ("unknown" if lose_read else "completed")
    assert outcome["status"] == ("error" if lose_read else "ready")
    if lose_read:
        assert approve(client, session).status_code == 409
        for _ in range(100):
            if target.release_count:
                break
            time.sleep(0.01)
        assert target.release_count == 1 and target.owner is None
    assert len(target.executions) == 1


def test_provider_final_message_releases_simulator_owner(app):
    client, target = app

    async def complete(profile, payload):
        return {
            "status": "complete",
            "model": "deterministic-mock-v1",
            "continuation": None,
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "summary": "No action.",
        }

    client.app.state.agent_control.planner = complete
    session = propose(client, create(client))
    assert session["status"] == "complete"
    for _ in range(100):
        if target.release_count:
            break
        time.sleep(0.01)
    assert target.release_count == 1 and target.owner is None
    assert target.executions == []


def test_receipt_age_reduces_admission_lifetime_without_mutating_hashed_source(app):
    client, target = app
    target.age_ms = 29000
    session = create(client)
    observation = session["observation"]
    remaining = (
        datetime.fromisoformat(observation["expires_at"]) - datetime.now(UTC)
    ).total_seconds()
    assert 0 < remaining < 2
    current, private = client.portal.call(client.app.state.agent_control.load, session["id"])
    source = private["raw_observation"]
    assert "observation_age_ms" not in source
    assert hashlib.sha256(canonical(source)).hexdigest() == current.observation.sha256
    target.age_ms = 31000
    refreshed = client.post(f"/api/v1/agent-sessions/{session['id']}/observe", json={}).json()
    response = client.post(
        f"/api/v1/agent-sessions/{session['id']}/proposals", json={"request_id": "expired"}
    )
    assert response.status_code == 409, refreshed
    assert target.executions == []


def test_stopped_owner_rejects_provider_result_that_swallows_cancellation(app):
    client, target = app
    original = client.app.state.agent_control.planner

    async def delayed(profile, payload):
        try:
            await asyncio.sleep(3)
        except asyncio.CancelledError:
            pass
        return await original(profile, payload)

    client.app.state.agent_control.planner = delayed
    session = create(client)
    path = f"/api/v1/agent-sessions/{session['id']}"
    assert client.post(path + "/proposals", json={"request_id": "late"}).status_code == 202
    stopped = client.post(path + "/stop", json={}).json()
    assert stopped["status"] == "stopped" and stopped["pending_proposal"] is None
    assert target.executions == [] and target.owner is None
    assert target.release_count == 1
    events = client.get(path + "/events").json()
    assert any(event["kind"] == "late_result_rejected" for event in events)
