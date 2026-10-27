"""Private SO101 adapter binding and reconciliation with no serial devices."""

import asyncio
import base64
import hashlib
import json

import pytest
from vla_platform.agent_control.robot_target import RobotTarget
from vla_platform.robot_lab_bridge import RobotLabError

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def observation(owner="session", epoch=5):
    image = b"\xff\xd8synthetic-wire-fixture\xff\xd9"
    raw = {
        "observation_id": "observed",
        "agent_session_id": owner,
        "authority_epoch": epoch,
        "joint_names": JOINTS,
        "joints": dict.fromkeys(JOINTS, 0.0),
        "units": {j: "percent" if j == "gripper" else "degrees" for j in JOINTS},
        "environment_kind": "physical",
        "embodiment_manifest_id": "so101-position-segment-v1",
        "calibration_sha256": "c" * 64,
        "source_age_ms": 10.0,
        "image_data_url": "data:image/jpeg;base64," + base64.b64encode(image).decode(),
        "image_sha256": hashlib.sha256(image).hexdigest(),
    }
    raw["observation_sha256"] = digest(raw)
    return raw


class FakeBridge:
    def __init__(self):
        self.records = {}
        self.calls = []
        self.lose_execute_ack = False
        self.corrupt = False
        self.block_acquire = None
        self.acquired = None
        self.releases = 0

    async def request(self, method, path, *, payload=None):
        self.calls.append((method, path, payload))
        if method == "GET":
            record = self.records[path.rsplit("/", 1)[-1]]
            if record["operation"] == "execute":
                record.update(
                    status="completed",
                    applied_targets={"shoulder_pan": 1.0},
                    final_measured_state={"shoulder_pan": 0.98},
                    units={"shoulder_pan": "degrees"},
                    elapsed_seconds=0.5,
                )
            return record
        operation = path.rsplit("/", 1)[-1]
        identity = payload["command_id"]
        fingerprint = digest({"operation": operation, "payload": payload})
        if identity in self.records:
            if self.records[identity]["request_sha256"] != fingerprint:
                raise RobotLabError("Changed command", 422)
            return self.records[identity]
        record = {
            "command_id": identity,
            "request_sha256": fingerprint,
            "operation": operation,
            "status": "completed",
            "state": {"owner": "session", "authority_epoch": 5},
        }
        if operation == "observe":
            record["observation"] = observation()
            if self.corrupt:
                record["observation"]["units"]["shoulder_pan"] = "radians"
        elif operation == "execute":
            record["status"] = "started"
        elif operation == "release":
            self.releases += 1
        self.records[identity] = record
        if operation == "execute" and self.lose_execute_ack:
            raise RobotLabError("Acknowledgement lost")
        if operation == "acquire" and self.block_acquire:
            self.acquired.set()
            await self.block_acquire.wait()
        return record


def test_arm_observe_uses_exact_hash_units_and_conservative_transport_age():
    async def check():
        bridge = FakeBridge()
        target = RobotTarget(bridge)
        value = await target.acquire("session", {}, command_id="durable-acquire")
        assert value["authority_epoch"] == 5
        assert target.observation_age_ms[("session", 5)] >= 10
        assert target.age_ms(value["observation"]) >= 10
        raw = dict(value["observation"])
        claimed = raw.pop("observation_sha256")
        assert digest(raw) == claimed
        assert bridge.calls[0][2]["command_id"] == "durable-acquire"
        await target.release("session", 5)
        assert bridge.releases == 1

    asyncio.run(check())


def test_changed_snapshot_never_reaches_a_model_and_releases_owner():
    async def check():
        bridge = FakeBridge()
        bridge.corrupt = True
        target = RobotTarget(bridge)
        with pytest.raises(RobotLabError, match="observation"):
            await target.acquire("session", {})
        assert bridge.releases == 1

    asyncio.run(check())


def test_lost_segment_ack_is_reconciled_once_without_a_second_post():
    async def check():
        bridge = FakeBridge()
        bridge.lose_execute_ack = True
        target = RobotTarget(bridge)
        raw = observation()
        receipt = await target.execute(
            "session",
            5,
            raw,
            "proposal",
            {"targets": {"shoulder_pan": 1.0}, "duration_seconds": 0.5},
            "model",
            "execution",
            model_observation=raw,
            prompt_provenance={"prompt_version": "agent-goal-v1", "prompt_sha256": "d" * 64},
        )
        assert receipt["status"] == "completed"
        assert len([c for c in bridge.calls if c[0] == "POST" and c[1] == "agent/execute"]) == 1
        sent = bridge.calls[0][2]["arguments"]
        assert sent["prompt_version"] == "agent-goal-v1" and sent["prompt_sha256"] == "d" * 64
        assert sent["units"] == {"shoulder_pan": "degrees"}
        assert sent["model_observation_id"] == raw["observation_id"]
        assert sent["observation_sha256"] == raw["observation_sha256"]

    asyncio.run(check())


def test_changed_payload_with_same_id_cannot_reuse_an_old_receipt():
    async def check():
        bridge = FakeBridge()
        target = RobotTarget(bridge)
        raw = observation()
        await target.execute(
            "session",
            5,
            raw,
            "proposal",
            {"targets": {"shoulder_pan": 1.0}, "duration_seconds": 0.5},
            "model",
            "execution",
        )
        with pytest.raises(RobotLabError, match="original request"):
            await target.execute(
                "session",
                5,
                raw,
                "proposal",
                {"targets": {"shoulder_pan": 2.0}, "duration_seconds": 0.5},
                "model",
                "execution",
            )
        assert len(bridge.records) == 1

    asyncio.run(check())


def test_cancelled_acquire_reconciles_original_identity_then_releases():
    async def check():
        bridge = FakeBridge()
        bridge.block_acquire = asyncio.Event()
        bridge.acquired = asyncio.Event()
        target = RobotTarget(bridge)
        task = asyncio.create_task(target.acquire("session", {}, command_id="known-acquire"))
        await bridge.acquired.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert bridge.releases == 1
        posts = [c for c in bridge.calls if c[0] == "POST" and c[1] == "agent/acquire"]
        assert len(posts) == 1 and posts[0][2]["command_id"] == "known-acquire"

    asyncio.run(check())
