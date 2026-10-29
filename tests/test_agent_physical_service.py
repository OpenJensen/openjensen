"""Core physical supervision and re-admission with a fake companion target only."""

import asyncio
import base64
import copy
import hashlib
import json
import time
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from vla_platform.api import create_app
from vla_platform.robot_lab_bridge import RobotLabError
from vla_platform.settings import Settings

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class FakeRobotTarget:
    def __init__(self):
        self.owner = None
        self.epoch = 7
        self.counter = 0
        self.joints = dict.fromkeys(JOINTS, 0.0)
        self.joints["gripper"] = 50.0
        self.calibration = "c" * 64
        self.units = {j: "percent" if j == "gripper" else "degrees" for j in JOINTS}
        self.observation_age_ms = {}
        self.heartbeat_calls = 0
        self.releases = []
        self.executions = []
        self.outcome = "completed"
        self.fail_post = False
        self.block_post = None
        self.entered_post = None
        self.block_execute = None
        self.entered_execute = None

    def snapshot(self, session):
        self.counter += 1
        stamp = time.monotonic_ns()
        image = b"\xff\xd8synthetic-physical-service-fixture\xff\xd9"
        raw = {
            "schema_version": 1,
            "observation_id": f"physical_{self.counter}",
            "target_session_id": "so101_fixture",
            "agent_session_id": session,
            "authority_epoch": self.epoch,
            "revision": len(self.executions),
            "environment_kind": "physical",
            "clock_policy": "physical_continuous",
            "embodiment_manifest_id": "so101-position-segment-v1",
            "robot_id": "fixture",
            "calibration_sha256": self.calibration,
            "joint_names": JOINTS,
            "joints": dict(self.joints),
            "units": dict(self.units),
            "joint_limits": {
                j: {
                    "min": 0.0 if j == "gripper" else -90.0,
                    "max": 100.0 if j == "gripper" else 90.0,
                }
                for j in JOINTS
            },
            "captured_monotonic_ns": stamp,
            "valid_until_monotonic_ns": stamp + 1_000_000_000,
            "source_age_ms": 0.0,
            "image_sensor_timestamp_verified": False,
            "image_data_url": "data:image/jpeg;base64," + base64.b64encode(image).decode(),
            "image_sha256": hashlib.sha256(image).hexdigest(),
        }
        raw["observation_sha256"] = fingerprint(raw)
        return raw

    async def acquire(self, session, arming, command_id=None):
        if self.owner is not None:
            raise RobotLabError("Controller already owned", 409)
        assert (
            arming.confirm == "ARM AI CONTROL"
            and arming.operator_present
            and arming.workspace_clear
        )
        assert command_id
        self.owner = session
        raw = self.snapshot(session)
        self.observation_age_ms[(session, self.epoch)] = 0.0
        return {"authority_epoch": self.epoch, "observation": raw}

    async def observe(self, session, epoch):
        assert self.owner == session and epoch == self.epoch
        if self.executions and self.fail_post:
            raise RobotLabError("Post-execution camera unavailable")
        if self.executions and self.block_post:
            self.entered_post.set()
            try:
                await self.block_post.wait()
            except asyncio.CancelledError:
                await self.block_post.wait()
        self.observation_age_ms[(session, epoch)] = 0.0
        return self.snapshot(session)

    async def heartbeat(self, session, epoch):
        assert self.owner == session and epoch == self.epoch
        self.heartbeat_calls += 1

    async def release(self, session, epoch):
        assert self.owner == session and epoch == self.epoch
        self.releases.append((session, epoch))
        self.owner = None

    async def execute(
        self, session, epoch, raw, proposal_id, args, model, execution_id, *, model_observation=None
    ):
        assert self.owner == session and epoch == self.epoch
        assert raw["observation_id"] != model_observation["observation_id"]
        assert (
            fingerprint({k: v for k, v in raw.items() if k != "observation_sha256"})
            == raw["observation_sha256"]
        )
        self.executions.append(
            {
                "id": execution_id,
                "admission": copy.deepcopy(raw),
                "model": copy.deepcopy(model_observation),
                "proposal_id": proposal_id,
            }
        )
        if self.block_execute:
            self.entered_execute.set()
            try:
                await self.block_execute.wait()
            except asyncio.CancelledError:
                await self.block_execute.wait()
        requested = dict(self.joints, **args.targets)
        if self.outcome == "raise":
            raise RobotLabError("Unknown segment outcome")
        if self.outcome == "completed":
            self.joints = requested
        return {
            "command_id": execution_id,
            "status": self.outcome,
            "requested_targets": requested,
            "applied_targets": dict(self.joints),
            "final_measured_state": dict(self.joints),
            "units": dict(self.units),
            "elapsed_seconds": 0.5,
            "reason": "Synthetic companion receipt",
            "model_observation_id": model_observation["observation_id"],
            "observation_id": raw["observation_id"],
        }


async def planner(_profile, payload):
    return {
        "status": "proposed",
        "skill": "joint_segment",
        "arguments": {"targets": {"shoulder_pan": 1.0}, "duration_seconds": 2.0},
        "model": "deterministic-physical-fixture",
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        "purpose": "Bounded fixture action; not an object grasp",
    }


@pytest.fixture
def app(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        robot = FakeRobotTarget()
        client.app.state.agent_control.robot_target = robot
        client.app.state.agent_control.planner = planner
        yield client, robot


def create(client, **changes):
    project = client.post("/api/v1/projects", json={"name": "physical protocol fixture"}).json()[
        "id"
    ]
    value = {
        "task": "Pick up the object",
        "target_kind": "robot_lab",
        "clock_policy": "physical",
        "mode": "supervised",
        "planning_mode": "task",
        "max_seconds": 30,
        "physical": {
            "confirm": "ARM AI CONTROL",
            "expected_generation": 3,
            "camera_index": 0,
            "operator_present": True,
            "workspace_clear": True,
        },
        **changes,
    }
    response = client.post(f"/api/v1/projects/{project}/agent-sessions", json=value)
    assert response.status_code == 201, response.text
    return response.json()


def wait(client, session, statuses, *, presence=True):
    route = "/api/v1/agent-sessions/" + session["id"]
    for n in range(300):
        current = client.get(route).json()
        if current["status"] in statuses:
            return current
        if presence and n % 20 == 0:
            heartbeat = client.post(
                route + "/heartbeat",
                json={"authority_epoch": session["authority_epoch"], "operator_present": True},
            )
            if heartbeat.status_code != 200:
                # Completion can release the lease after GET and before POST.
                # Accept only the same terminal outcome this waiter requires.
                current = client.get(route).json()
                if current["status"] in statuses:
                    return current
            assert heartbeat.status_code == 200, heartbeat.text
        time.sleep(0.01)
    raise AssertionError(current)


def propose(client, session, request="physical-plan"):
    response = client.post(
        "/api/v1/agent-sessions/" + session["id"] + "/proposals", json={"request_id": request}
    )
    assert response.status_code == 202, response.text
    return wait(client, session, {"waiting_for_acceptance", "paused", "error"})


def admission(client, session):
    route = "/api/v1/agent-sessions/" + session["id"]
    response = client.post(route + "/execution-observation")
    assert response.status_code == 200, response.text
    current = response.json()
    observed = current["execution_observation"]
    return current, {
        "proposal_id": current["pending_proposal"]["id"],
        "authority_epoch": current["authority_epoch"],
        "execution_observation_id": observed["id"],
        "execution_observation_sha256": observed["sha256"],
        "scene_unchanged_confirmed": True,
    }


def test_arming_is_explicit_and_physical_autonomy_is_not_silently_enabled(app):
    client, robot = app
    project = client.post("/api/v1/projects", json={"name": "not armed"}).json()["id"]
    for fields in (
        {},
        {
            "physical": {
                "confirm": "ARM AI CONTROL",
                "expected_generation": 3,
                "camera_index": 0,
                "operator_present": False,
                "workspace_clear": True,
            }
        },
        {
            "mode": "bounded",
            "physical": {
                "confirm": "ARM AI CONTROL",
                "expected_generation": 3,
                "camera_index": 0,
                "operator_present": True,
                "workspace_clear": True,
            },
        },
    ):
        response = client.post(
            f"/api/v1/projects/{project}/agent-sessions",
            json={
                "task": "pick up an object",
                "target_kind": "robot_lab",
                "clock_policy": "physical",
                **fields,
            },
        )
        assert response.status_code == 422, response.text
    assert robot.owner is None and not robot.executions


def test_exact_fresh_re_admission_keeps_original_model_context_and_receipt(app):
    client, robot = app
    session = propose(client, create(client))
    model_id = session["observation"]["id"]
    route = "/api/v1/agent-sessions/" + session["id"]
    assert (
        client.post(
            route + "/approve",
            json={
                "proposal_id": session["pending_proposal"]["id"],
                "authority_epoch": session["authority_epoch"],
            },
        ).status_code
        == 409
    )
    current, approval = admission(client, session)
    image = client.get(route + "/observations/" + current["execution_observation"]["id"] + "/image")
    assert image.status_code == 200 and image.content.startswith(b"\xff\xd8")
    assert client.post(route + "/approve", json=approval).status_code == 202
    finished = wait(client, session, {"ready", "error"})
    assert finished["status"] == "ready", finished
    receipt = finished["receipts"][0]
    assert receipt["status"] == "completed"
    assert receipt["original_model_observation_id"] == model_id
    assert receipt["admission_observation_id"] == current["execution_observation"]["id"]
    assert len(robot.executions) == 1
    assert (
        robot.executions[0]["model"]["observation_id"]
        != robot.executions[0]["admission"]["observation_id"]
    )
    assert client.post(route + "/approve", json=approval).status_code == 202
    assert len(robot.executions) == 1
    assert robot.heartbeat_calls >= 1


@pytest.mark.parametrize(
    "fault", ["hash", "epoch", "scene", "calibration", "units", "drift", "expired", "tamper"]
)
def test_physical_re_admission_refuses_wrong_or_stale_context(app, fault):
    client, robot = app
    session = propose(client, create(client))
    if fault == "calibration":
        robot.calibration = "d" * 64
    if fault == "units":
        robot.units["shoulder_pan"] = "rad"
    if fault == "drift":
        robot.joints["shoulder_pan"] = 0.75
    current, approval = admission(client, session)
    if fault == "hash":
        approval["execution_observation_sha256"] = "a" * 64
    if fault == "epoch":
        approval["authority_epoch"] += 1
    if fault == "scene":
        approval["scene_unchanged_confirmed"] = False
    if fault in {"expired", "tamper"}:

        async def mutate():
            service = client.app.state.agent_control
            s, p = await service.load(session["id"])
            if fault == "expired":
                s.execution_observation.expires_at = (
                    datetime.now(UTC) - timedelta(seconds=1)
                ).isoformat()
            else:
                p["execution_raw_observation"]["image_sha256"] = "b" * 64
            await service.save(s, p, "fixture-context-change")

        client.portal.call(mutate)
    route = "/api/v1/agent-sessions/" + session["id"] + "/approve"
    assert client.post(route, json=approval).status_code == 409
    assert robot.executions == []


def test_slow_model_requires_new_physical_snapshot_and_can_be_supervised(app):
    client, robot = app

    async def slow(profile, payload):
        await asyncio.sleep(1.1)
        return await planner(profile, payload)

    client.app.state.agent_control.planner = slow
    session = propose(client, create(client))
    assert session["status"] == "waiting_for_acceptance", session
    assert datetime.fromisoformat(session["observation"]["expires_at"]) < datetime.now(UTC)
    current, approval = admission(client, session)
    assert (
        client.post(
            "/api/v1/agent-sessions/" + session["id"] + "/approve", json=approval
        ).status_code
        == 202
    )
    result = wait(client, session, {"ready", "error"})
    assert result["receipts"][0]["status"] == "completed"
    assert len(robot.executions) == 1


@pytest.mark.parametrize("outcome", ["rejected", "stopped", "unknown", "raise"])
def test_physical_receipt_outcomes_never_repost_a_motion(app, outcome):
    client, robot = app
    session = propose(client, create(client))
    robot.outcome = outcome
    current, approval = admission(client, session)
    route = "/api/v1/agent-sessions/" + session["id"]
    assert client.post(route + "/approve", json=approval).status_code == 202
    result = wait(client, session, {"paused", "error"})
    assert result["receipts"][0]["status"] == ("unknown" if outcome == "raise" else outcome)
    assert len(robot.executions) == 1
    assert client.post(route + "/approve", json=approval).status_code == 409
    assert len(robot.executions) == 1


def test_operator_lease_loss_stops_while_provider_ignores_cancellation(app):
    client, robot = app

    async def delayed(profile, payload):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            pass
        return await planner(profile, payload)

    client.app.state.agent_control.planner = delayed
    session = create(client)
    route = "/api/v1/agent-sessions/" + session["id"]
    assert client.post(route + "/proposals", json={"request_id": "late"}).status_code == 202
    result = wait(client, session, {"interrupted", "stopped"}, presence=False)
    assert result["authority_epoch"] > session["authority_epoch"]
    assert not result["control_armed"] and robot.owner is None
    assert len(robot.releases) == 1 and robot.executions == []
    assert result["pending_proposal"] is None


def test_known_completion_survives_post_observation_failure(app):
    client, robot = app
    session = propose(client, create(client))
    current, approval = admission(client, session)
    robot.fail_post = True
    route = "/api/v1/agent-sessions/" + session["id"]
    assert client.post(route + "/approve", json=approval).status_code == 202
    result = wait(client, session, {"paused"})
    assert robot.owner == session["id"] and robot.releases == []
    assert result["control_armed"]
    assert result["receipts"][0]["status"] == "completed"
    assert result["receipts"][0]["post_observation_id"] is None
    assert result["receipts"][0]["original_model_observation_id"] == session["observation"]["id"]
    assert len(robot.executions) == 1


def test_stale_dispatch_records_rejection_without_motion(app):
    client, robot = app
    session = propose(client, create(client))
    current, approval = admission(client, session)
    service = client.app.state.agent_control
    original = service.execute
    gate = asyncio.Event()

    async def delayed(identity, approved):
        await gate.wait()
        await original(identity, approved)

    service.execute = delayed
    route = "/api/v1/agent-sessions/" + session["id"]
    assert client.post(route + "/approve", json=approval).status_code == 202

    async def expire():
        s, p = await service.load(session["id"])
        p["approved_admission"]["public"]["expires_at"] = (
            datetime.now(UTC) - timedelta(seconds=1)
        ).isoformat()
        await service.save(s, p, "fixture-admission-expired")
        gate.set()

    client.portal.call(expire)
    result = wait(client, session, {"paused", "error"})
    assert result["receipts"][-1]["status"] == "rejected"
    assert robot.executions == []


def wait_released(client, robot, session):
    deadline = time.monotonic() + 2.8
    route = "/api/v1/agent-sessions/" + session["id"]
    while time.monotonic() < deadline:
        state = client.get(route).json()
        if robot.owner is None:
            return state
        time.sleep(0.02)
    raise AssertionError(
        "Revoked physical owner remained active while a coroutine ignored cancellation"
    )


def test_lease_loss_releases_before_uncooperative_provider_returns(app):
    client, robot = app
    entered, gate = asyncio.Event(), asyncio.Event()

    async def uncooperative(profile, payload):
        entered.set()
        try:
            await gate.wait()
        except asyncio.CancelledError:
            await gate.wait()
        return await planner(profile, payload)

    client.app.state.agent_control.planner = uncooperative
    session = create(client)
    route = "/api/v1/agent-sessions/" + session["id"]
    try:
        assert (
            client.post(route + "/proposals", json={"request_id": "uncooperative"}).status_code
            == 202
        )
        client.portal.call(entered.wait)
        result = wait_released(client, robot, session)
        assert not result["control_armed"]
        assert result["authority_epoch"] > session["authority_epoch"]
        assert result["pending_proposal"] is None and robot.executions == []
    finally:
        client.portal.call(gate.set)


@pytest.mark.parametrize("stage", ["execute", "post_observation"])
def test_revocation_fences_late_segment_or_camera_completion(app, stage):
    client, robot = app
    session = propose(client, create(client))
    current, approval = admission(client, session)
    route = "/api/v1/agent-sessions/" + session["id"]
    gate, entered = asyncio.Event(), asyncio.Event()
    if stage == "execute":
        robot.block_execute, robot.entered_execute = gate, entered
    else:
        robot.block_post, robot.entered_post = gate, entered
    try:
        assert client.post(route + "/approve", json=approval).status_code == 202
        client.portal.call(entered.wait)
        interrupted = wait_released(client, robot, session)
        assert not interrupted["control_armed"]
        receipt = interrupted["receipts"][-1]
        assert receipt["status"] == ("unknown" if stage == "execute" else "completed")
        assert receipt["post_observation_id"] is None
        prior_observation = interrupted["observation"]["id"]
        client.portal.call(gate.set)
        time.sleep(0.1)
        late = client.get(route).json()
        assert late["authority_epoch"] == interrupted["authority_epoch"]
        assert late["observation"]["id"] == prior_observation
        assert late["receipts"] == interrupted["receipts"]
        assert len(robot.executions) == 1
    finally:
        client.portal.call(gate.set)


@pytest.mark.parametrize("operation", ["pause", "cancel-inference"])
def test_pause_or_cancel_preserves_local_hold_and_fences_late_provider(app, operation):
    client, robot = app
    entered, gate = asyncio.Event(), asyncio.Event()

    async def uncooperative(profile, payload):
        entered.set()
        try:
            await gate.wait()
        except asyncio.CancelledError:
            await gate.wait()
        return await planner(profile, payload)

    client.app.state.agent_control.planner = uncooperative
    session = create(client)
    route = "/api/v1/agent-sessions/" + session["id"]
    try:
        assert (
            client.post(route + "/proposals", json={"request_id": "pause-race"}).status_code == 202
        )
        client.portal.call(entered.wait)
        started = time.monotonic()
        response = client.post(route + "/" + operation)
        assert response.status_code == 200, response.text
        paused = response.json()
        assert time.monotonic() - started < 0.8
        assert paused["status"] == "paused" and paused["control_armed"]
        assert paused["authority_epoch"] == session["authority_epoch"]
        assert paused["pending_proposal"] is None
        assert robot.owner == session["id"] and robot.releases == []
        assert (
            client.post(
                route + "/heartbeat",
                json={"authority_epoch": session["authority_epoch"], "operator_present": True},
            ).status_code
            == 200
        )
        client.portal.call(gate.set)
        time.sleep(0.1)
        late = client.get(route).json()
        assert late["pending_proposal"] is None and late["status"] == "paused"
        assert not robot.executions
        stopped = client.post(route + "/stop")
        assert stopped.status_code == 200 and not stopped.json()["control_armed"]
        assert robot.owner is None
    finally:
        client.portal.call(gate.set)


def test_provider_failure_preserves_local_hold_until_explicit_stop(app):
    client, robot = app

    async def failed(_profile, _payload):
        raise RuntimeError("Synthetic unavailable model")

    client.app.state.agent_control.planner = failed
    session = propose(client, create(client))
    session = wait(client, session, {"paused"})
    assert session["control_armed"] and robot.owner == session["id"]
    assert robot.releases == [] and robot.executions == []
    route = "/api/v1/agent-sessions/" + session["id"]
    stopped = client.post(route + "/stop")
    assert stopped.status_code == 200 and robot.owner is None
