"""Fixed private SO101 transport; no companion credential reaches a model."""

import asyncio
import base64
import hashlib
import json
import math
import re
import time
from uuid import uuid4

from vla_platform.robot_lab_bridge import RobotLabError


class RobotTarget:
    def __init__(self, bridge):
        self.bridge = bridge
        self.nonces = {}
        self.observation_receipts = {}
        self.observation_age_ms = {}
        self.command_hashes = {}

    async def command(self, operation, session_id, epoch, arguments, command_id=None):
        identity = command_id or uuid4().hex
        body = {
            "command_id": identity,
            "agent_session_id": session_id,
            "authority_epoch": epoch,
            "arguments": arguments,
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                {"operation": operation, "payload": body},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        self.command_hashes[identity] = fingerprint
        # A lost POST response is reconciled by identity, never reposted.
        try:
            value = await self.bridge.request("POST", "agent/" + operation, payload=body)
        except RobotLabError, OSError, TimeoutError:
            value = await self.bridge.request("GET", "agent/executions/" + identity)
        self.validate_receipt(value, identity)
        return value

    def validate_receipt(self, value, identity):
        if (
            not isinstance(value, dict)
            or value.get("command_id") != identity
            or value.get("request_sha256") != self.command_hashes[identity]
        ):
            raise RobotLabError(
                "Physical execution receipt does not match the original request", 502
            )

    def validate_observation(self, raw, session_id, epoch):
        try:
            value = dict(raw)
            expected_hash = value.pop("observation_sha256")
            actual_hash = hashlib.sha256(
                json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest()
            joints = [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            ]
            if (
                expected_hash != actual_hash
                or value["agent_session_id"] != session_id
                or type(value["authority_epoch"]) is not int
                or value["authority_epoch"] != epoch
                or value["joint_names"] != joints
                or value["units"] != {j: "percent" if j == "gripper" else "degrees" for j in joints}
            ):
                raise ValueError
            if (
                value["environment_kind"] != "physical"
                or value["embodiment_manifest_id"] != "so101-position-segment-v1"
                or not re.fullmatch(r"[a-f0-9]{64}", value["calibration_sha256"])
            ):
                raise ValueError
            if set(value["joints"]) != set(joints) or any(
                isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n)
                for n in value["joints"].values()
            ):
                raise ValueError
            image_url = value["image_data_url"]
            if not isinstance(image_url, str) or not image_url.startswith(
                "data:image/jpeg;base64,"
            ):
                raise ValueError
            image = base64.b64decode(image_url.split(",", 1)[1], validate=True)
            if (
                not image.startswith(b"\xff\xd8")
                or len(image) > 2 * 1024**2
                or hashlib.sha256(image).hexdigest() != value["image_sha256"]
            ):
                raise ValueError
            age = value["source_age_ms"]
            if (
                isinstance(age, bool)
                or not isinstance(age, (int, float))
                or not math.isfinite(age)
                or not 0 <= age <= 1000
            ):
                raise ValueError
        except ValueError, KeyError, TypeError:
            raise RobotLabError(
                "Physical observation identity, units or image binding is invalid", 502
            ) from None

    async def acquire(self, session_id, arming, command_id=None):
        payload = arming.model_dump() if hasattr(arming, "model_dump") else dict(arming)
        identity = command_id or uuid4().hex
        try:
            value = await self.command("acquire", session_id, 0, payload, identity)
        except asyncio.CancelledError:
            # The POST may have transferred an already powered manual controller.
            # Reconcile the original ID without reacquiring or replaying motion.
            try:
                async with asyncio.timeout(1.5):
                    value = await self.bridge.request("GET", "agent/executions/" + identity)
                    self.validate_receipt(value, identity)
                    state = value.get("state", {})
                    if value.get("status") == "completed" and state.get("owner") == session_id:
                        await self.release(session_id, state["authority_epoch"])
            except Exception:
                pass  # Independent companion lease still expires after two seconds.
            raise
        if value.get("status") != "completed":
            raise RobotLabError("Physical controller did not acknowledge arming", 409)
        epoch = value["state"]["authority_epoch"]
        self.nonces[(session_id, epoch)] = 0
        try:
            observed = await self.observe(session_id, epoch)
        except BaseException:
            await self.release(session_id, epoch)
            raise
        return {"authority_epoch": epoch, "observation": observed}

    async def observe(self, session_id, epoch):
        started = time.monotonic()
        value = await self.command("observe", session_id, epoch, {})
        if value.get("status") != "completed" or "observation" not in value:
            raise RobotLabError("Fresh physical observation was not admitted", 409)
        raw = value["observation"]
        self.validate_observation(raw, session_id, epoch)
        self.observation_age_ms[(session_id, epoch)] = (
            raw.get("source_age_ms", 0) + (time.monotonic() - started) * 1000
        )
        self.observation_receipts[raw["observation_id"]] = (
            time.monotonic(),
            raw.get("source_age_ms", 0) + (time.monotonic() - started) * 1000,
        )
        while len(self.observation_receipts) > 64:
            self.observation_receipts.pop(next(iter(self.observation_receipts)))
        return raw

    def age_ms(self, raw):
        """Conservative receipt age without altering the immutable source hash."""
        received, age = self.observation_receipts[raw["observation_id"]]
        return age + (time.monotonic() - received) * 1000

    async def execute(
        self,
        session_id,
        epoch,
        raw,
        proposal_id,
        args,
        model,
        execution_id,
        *,
        model_observation=None,
    ):
        args = args.model_dump() if hasattr(args, "model_dump") else dict(args)
        targets = args["targets"]
        body = {
            "observation_id": raw["observation_id"],
            "observation_sha256": raw["observation_sha256"],
            "targets": targets,
            "units": {j: raw["units"][j] for j in targets},
            "duration_seconds": args["duration_seconds"],
            "proposal_id": proposal_id,
            "provider_model": model,
        }
        if model_observation is not None:
            body.update(
                model_observation_id=model_observation["observation_id"],
                model_observation_sha256=model_observation["observation_sha256"],
            )
        value = await self.command("execute", session_id, epoch, body, execution_id)
        deadline = time.monotonic() + args["duration_seconds"] + 4
        while value.get("status") in {"accepted", "started"}:
            if time.monotonic() > deadline:
                raise RobotLabError("Physical segment receipt is unknown; no retry was made", 409)
            await asyncio.sleep(0.05)
            value = await self.bridge.request("GET", "agent/executions/" + execution_id)
            self.validate_receipt(value, execution_id)
        return value

    async def heartbeat(self, session_id, epoch):
        key = (session_id, epoch)
        self.nonces[key] = self.nonces.get(key, 0) + 1
        value = await self.command(
            "heartbeat", session_id, epoch, {"nonce": self.nonces[key], "operator_present": True}
        )
        if value.get("status") != "completed":
            raise RobotLabError("AI operator lease was not renewed", 409)

    async def release(self, session_id, epoch):
        value = await self.command("release", session_id, epoch, {})
        self.nonces.pop((session_id, epoch), None)
        self.observation_age_ms.pop((session_id, epoch), None)
        if value.get("status") != "completed":
            raise RobotLabError("AI release was not acknowledged; use physical Stop", 409)
