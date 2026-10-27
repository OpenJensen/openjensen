"""Lease-bound SO101 position segments owned and timed by the local companion.

This is joint-position control, not Cartesian planning, collision avoidance or
contact-force control. No model, network client or browser writes motor packets.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

from hardware import JOINTS, HardwareError, denormalized, normalized
from motion import control_limits

IDENTITY = re.compile(r"[A-Za-z0-9_-]{1,96}\Z")
LEASE_SECONDS = 2.0
MAX_AGE_MS = 1000.0
MAX_SKEW_MS = 150.0
MAX_DRIFT = {j: 2.0 if j == "gripper" else 1.0 for j in JOINTS}
UNITS = {j: "percent" if j == "gripper" else "degrees" for j in JOINTS}
LIMITS = {
    "arm_delta_degrees": 5.0,
    "gripper_delta_percent": 10.0,
    "max_speed_units_per_second": 10.0,
    "max_acceleration_units_per_second2": 20.0,
    "duration_min_seconds": 0.5,
    "duration_max_seconds": 5.0,
    "tracking_timeout_seconds": 2.0,
    "arm_tracking_error_degrees": 3.0,
    "gripper_tracking_error_percent": 6.0,
    "arm_completion_tolerance_degrees": 1.0,
    "gripper_completion_tolerance_percent": 2.0,
    "max_observation_age_ms": MAX_AGE_MS,
    "max_sensor_read_skew_ms": MAX_SKEW_MS,
    "max_local_health_age_seconds": 0.5,
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate agent JSON property")
            result[key] = value
        return result

    def constant(_value):
        raise ValueError("Non-finite agent JSON number")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def identity(value):
    if not isinstance(value, str) or not IDENTITY.fullmatch(value):
        raise ValueError("Invalid agent command identity")
    return value


def number(value, low, high, label):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError(f"Invalid {label}")
    return float(value)


def durable_json(path, value):
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("Agent evidence cannot use symbolic links")
    descriptor, pending = tempfile.mkstemp(prefix=".agent-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(value, stream, allow_nan=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        Path(pending).unlink(missing_ok=True)


class Segment:
    def __init__(self, initial, targets, duration, now):
        self.initial = dict(initial)
        self.targets = dict(targets)
        self.duration = duration
        self.started = now
        self.last = now
        self.progress_time = 0.0
        self.deadline = now + duration + LIMITS["tracking_timeout_seconds"] + 0.5
        self.last_command = dict(initial)

    def step(self, measured, now):
        dt = min(0.075, max(0.0, now - self.last))
        self.last = now
        self.progress_time = min(self.duration, self.progress_time + dt)
        u = self.progress_time / self.duration
        fraction = u**3 * (10 - 15 * u + 6 * u * u)
        targets = {
            j: self.initial[j] + (self.targets[j] - self.initial[j]) * fraction for j in JOINTS
        }
        for j in JOINTS:
            tracking = 6.0 if j == "gripper" else 3.0
            if abs(measured[j] - self.last_command[j]) > tracking:
                raise HardwareError("Agent segment tracking tolerance exceeded")
        self.last_command = targets
        arrived = self.progress_time >= self.duration and all(
            abs(measured[j] - self.targets[j]) <= (2.0 if j == "gripper" else 1.0) for j in JOINTS
        )
        if now > self.deadline and not arrived:
            raise HardwareError(
                "Agent segment did not reach its measured target before its deadline"
            )
        return targets, arrived


class AgentExecutor:
    def __init__(self, lab, *, clock=time.monotonic):
        self.lab = lab
        self.clock = clock
        self.root = lab.data_dir / "agent-control"
        self.root.mkdir(mode=0o700, exist_ok=True)
        self.receipts_dir = self.root / "executions"
        self.observations_dir = self.root / "observations"
        for directory in (self.root, self.receipts_dir, self.observations_dir):
            if directory.is_symlink():
                raise ValueError("Agent evidence directories cannot use symbolic links")
            directory.mkdir(mode=0o700, exist_ok=True)
        self.state_path = self.root / "authority.json"
        if self.state_path.is_symlink():
            raise ValueError("Agent authority cannot use symbolic links")
        self.session_observation_bytes = 0
        self.observation_bytes = sum(p.stat().st_size for p in self.observations_dir.glob("*.json"))
        self.epoch = 0
        if self.state_path.exists():
            value = json.loads(self.state_path.read_text())
            if type(value.get("epoch")) is not int or value["epoch"] < 0:
                raise ValueError("Agent authority store is invalid")
            self.epoch = value["epoch"]
        self.epoch += 1
        durable_json(self.state_path, {"epoch": self.epoch})
        self.cache = {}
        self.evidence_lock = threading.RLock()
        self.command_lock = threading.Lock()
        self.pending_revocations = []
        self.owner = None
        self.generation = None
        self.lease_deadline = 0.0
        self.operator_deadline = 0.0
        self.camera_index = None
        self.calibration_sha256 = None
        self.target_session_id = None
        self.observations = {}
        self.active_id = None
        self.segment = None
        self.last_applied = None
        self.last_health = 0.0
        self.heartbeat_nonce = -1
        self.command_revision = 0
        # Never resume a motion after process restart. Keep its original identity.
        for path in self.receipts_dir.glob("*.json"):
            record = json.loads(path.read_text())
            self.cache[record["command_id"]] = record
            if record.get("status") in {"started", "accepted"}:
                record.update(
                    status="unknown", reason="Companion restarted; no execution was replayed"
                )
                durable_json(path, record)

    def _next_epoch(self):
        self.epoch += 1
        durable_json(self.state_path, {"epoch": self.epoch})

    def status(self):
        return {
            "schema_version": 1,
            "available": True,
            "owner": self.owner,
            "authority_epoch": self.epoch,
            "target_session_id": self.target_session_id,
            "active_execution_id": self.active_id,
            "units": dict(UNITS),
            "limits": dict(LIMITS),
            "lease_seconds": LEASE_SECONDS,
            "cartesian_control": False,
            "collision_avoidance": False,
            "contact_force_control": False,
            "image_sensor_timestamp_verified": False,
            "profile_semantics": "commanded_position_interpolation",
            "tracking_profile_hardware_verified": False,
            "firmware_watchdog_verified": False,
        }

    def calibration_hash(self):
        return digest(self.lab.calibration)

    def validate_owner(self, owner, epoch):
        if (
            self.owner != owner
            or type(epoch) is not int
            or epoch != self.epoch
            or self.generation != self.lab.generation
            or self.lab.mode != "agent"
            or not self.lab.torque
            or self.lab.controller is None
            or self.lab.bus is None
        ):
            raise ValueError("Agent controller ownership or generation is stale")
        if self.clock() >= min(self.lease_deadline, self.operator_deadline):
            raise HardwareError("Agent operator or execution lease expired")
        if self.calibration_hash() != self.calibration_sha256:
            raise HardwareError("Agent calibration changed")

    def expired(self):
        return self.owner is not None and (
            self.clock() >= min(self.lease_deadline, self.operator_deadline)
            or self.clock() - self.last_health > LIMITS["max_local_health_age_seconds"]
        )

    def receipt(self, ident):
        ident = identity(ident)
        if ident in self.cache:
            return json.loads(json.dumps(self.cache[ident]))
        path = self.receipts_dir / (ident + ".json")
        if not path.exists():
            raise FileNotFoundError("Agent execution identity not found")
        return json.loads(path.read_text())

    def _save(self, record):
        # Stop revokes in memory without waiting for disk. Serialize persistence
        # afterwards, and never let a delayed "started" write replace a terminal
        # receipt that has already been published by revocation/completion.
        with self.evidence_lock:
            existing = self.cache.get(record["command_id"])
            if existing and existing["status"] in {"completed", "stopped", "rejected", "unknown"}:
                record = existing
            else:
                self.cache[record["command_id"]] = dict(record)
            durable_json(self.receipts_dir / (record["command_id"] + ".json"), record)

    def revoke(self, reason="Stopped by operator"):
        # This path performs no file or USB I/O. Stop first invalidates ownership
        # and then disables torque; evidence is flushed after the port is closed.
        if self.active_id:
            record = dict(self.cache[self.active_id])
            if record["status"] == "started":
                record.update(
                    status="stopped",
                    reason=reason,
                    final_measured_state=dict(self.lab.positions),
                    completed_monotonic_ns=time.monotonic_ns(),
                )
                self.cache[self.active_id] = record
                self.pending_revocations.append(record)
        if self.owner is not None:
            self.epoch += 1
        self.owner = None
        self.segment = None
        self.active_id = None
        self.observations.clear()
        self.lease_deadline = self.operator_deadline = 0.0

    def flush_revocations(self):
        records, self.pending_revocations = self.pending_revocations, []
        durable_json(self.state_path, {"epoch": self.epoch})
        for record in records:
            self._save(record)

    def command(self, operation, payload):
        if (
            operation not in {"acquire", "observe", "execute", "heartbeat", "release"}
            or not isinstance(payload, dict)
            or set(payload) != {"command_id", "agent_session_id", "authority_epoch", "arguments"}
        ):
            raise ValueError("Invalid agent command envelope")
        command_id, owner = identity(payload["command_id"]), identity(payload["agent_session_id"])
        args, epoch = payload["arguments"], payload["authority_epoch"]
        if not isinstance(args, dict) or type(epoch) is not int or epoch < 0:
            raise ValueError("Invalid agent arguments or epoch")
        fingerprint = digest({"operation": operation, "payload": payload})
        with self.command_lock:
            existing = self.receipts_dir / (command_id + ".json")
            if existing.exists():
                record = self.receipt(command_id)
                if record["request_sha256"] != fingerprint:
                    raise ValueError("Command identity already belongs to another payload")
                return record  # Duplicate heartbeat never renews a lease.
            if len(list(self.receipts_dir.glob("*.json"))) >= 20000 and operation != "release":
                raise ValueError("Agent evidence store needs archival before another session")
            record = {
                "schema_version": 1,
                "command_id": command_id,
                "agent_session_id": owner,
                "authority_epoch": epoch,
                "operation": operation,
                "request_sha256": fingerprint,
                "status": "accepted",
                "created_monotonic_ns": time.monotonic_ns(),
                "task_success_verified": False,
            }
            # Admission failures are durable too; the same ID can never dispatch later.
            self._save(record)
        try:
            with self.lab.lock:
                if operation == "acquire":
                    if time.monotonic_ns() - record["created_monotonic_ns"] > 500_000_000:
                        raise ValueError(
                            "Arming request expired while waiting for the local controller"
                        )
                    result = self.acquire(owner, epoch, args)
                else:
                    self.validate_owner(owner, epoch)
                    if operation == "observe":
                        if args:
                            raise ValueError("Observe takes no model arguments")
                        result = {"observation": self.observe()}
                    elif operation == "execute":
                        result = self.execute(command_id, args)
                    elif operation == "heartbeat":
                        with self.lab.lifecycle_lock:
                            self.validate_owner(owner, epoch)
                            result = self.heartbeat(args)
                    else:
                        if args:
                            raise ValueError("Release takes no model arguments")
                        self.lab.stop(
                            "Agent session released; torque disabled",
                            expected_generation=self.generation,
                            expected_mode="agent",
                        )
                        result = {"state": self.status()}
                with self.lab.lifecycle_lock:
                    if (
                        operation != "release"
                        and operation != "heartbeat"
                        and self.generation != self.lab.generation
                    ):
                        raise ValueError("Controller generation changed before command completion")
                    record.update(
                        result,
                        status="started" if operation == "execute" else "completed",
                        authority_epoch=self.epoch,
                    )
                    self.cache[command_id] = dict(record)
            self._save(record)
            return record
        except Exception as error:
            record.update(status="rejected", reason=str(error))
            self._save(record)
            raise

    def acquire(self, owner, epoch, args):
        expected = {
            "confirm",
            "expected_generation",
            "camera_index",
            "operator_present",
            "workspace_clear",
        }
        if (
            set(args) != expected
            or args["confirm"] != "ARM AI CONTROL"
            or args["operator_present"] is not True
            or args["workspace_clear"] is not True
            or epoch != 0
        ):
            raise ValueError("Explicit operator arming and clear workspace are required")
        if (
            type(args["expected_generation"]) is not int
            or args["expected_generation"] != self.lab.generation
        ):
            raise ValueError("Manual controller generation changed before arming")
        if type(args["camera_index"]) is not int or not 0 <= args["camera_index"] <= 63:
            raise ValueError("Choose an admitted camera")
        if (
            self.owner
            or self.lab.mode != "manual"
            or set(self.lab.calibration) != set(JOINTS)
            or not self.lab.torque
            or self.lab.controller is None
            or self.lab.bus is None
        ):
            raise ValueError("Start full-arm calibrated joint control before arming AI control")
        if (
            self.lab.middle_motion
            or self.lab.rest_motion
            or self.lab.writer
            and self.lab.writer.episode_active
        ):
            raise ValueError("Finish movement and recording before transferring control")
        if self.clock() - self.lab.heartbeat > LEASE_SECONDS:
            raise ValueError("Manual operator heartbeat expired")
        if not callable(getattr(self.lab.camera, "capture_timed", None)):
            raise ValueError("A fresh timestamped camera capture is required")
        self._next_epoch()
        with self.lab.stop_lock, self.lab.lifecycle_lock:
            if (
                args["expected_generation"] != self.lab.generation
                or self.lab.mode != "manual"
                or not self.lab.torque
            ):
                raise ValueError("Manual session was stopped during acquisition")
            self.owner, self.generation = owner, self.lab.generation
            self.camera_index = args["camera_index"]
            self.calibration_sha256 = self.calibration_hash()
            self.target_session_id = (
                "so101_"
                + digest(
                    {
                        "robot_id": self.lab.robot_id,
                        "generation": self.generation,
                        "epoch": self.epoch,
                    }
                )[:24]
            )
            self.lease_deadline = self.operator_deadline = self.clock() + LEASE_SECONDS
            self.heartbeat_nonce = -1
            self.command_revision = 0
            self.session_observation_bytes = 0
            self.lab.mode = "agent"
            self.last_applied = dict(self.lab.action)
            self.last_health = self.clock()
            self.observations.clear()
        return {"state": self.status()}

    def heartbeat(self, args):
        if self.clock() - self.last_health > LIMITS["max_local_health_age_seconds"]:
            raise HardwareError("Local encoder/controller health expired")
        if (
            set(args) != {"nonce", "operator_present"}
            or type(args["nonce"]) is not int
            or args["nonce"] <= self.heartbeat_nonce
            or args["operator_present"] is not True
        ):
            raise ValueError("Heartbeat needs a fresh owner nonce and present operator")
        self.heartbeat_nonce = args["nonce"]
        self.lease_deadline = self.operator_deadline = self.clock() + LEASE_SECONDS
        return {"state": self.status()}

    def measured(self, raw):
        if not isinstance(raw, dict) or set(raw) != set(JOINTS):
            raise HardwareError("Incomplete agent encoder sample")
        for joint, value in raw.items():
            entry = self.lab.calibration[joint]
            if type(value) is not int or not entry["range_min"] <= value <= entry["range_max"]:
                raise HardwareError("Agent encoder is outside its admitted calibration")
        return {j: normalized(j, raw[j], self.lab.calibration) for j in JOINTS}

    def observe(self):
        if self.active_id:
            raise ValueError("Wait for the segment receipt before observing")
        if (
            self.session_observation_bytes > 128 * 1024**2 - 2 * 1024**2
            or self.observation_bytes > 512 * 1024**2 - 2 * 1024**2
            or shutil.disk_usage(self.root).free < 1024**3
        ):
            raise ValueError("Agent observation evidence needs archival or free disk space")
        lab, generation = self.lab, self.lab.generation
        bus = lab.bus
        begin_ns = time.monotonic_ns()
        with bus.lock:
            before = bus.positions()
            if set(before) != set(JOINTS):
                raise HardwareError("Incomplete encoder sample")
            self.measured(before)
            before_ns = time.monotonic_ns()
        capture = lab.camera.capture_timed(self.camera_index)
        with bus.lock:
            self.validate_owner(self.owner, self.epoch)
            after = bus.positions()
            end_ns = time.monotonic_ns()
            self.validate_owner(self.owner, self.epoch)
            if generation != lab.generation:
                raise ValueError("Controller changed during capture")
            for joint in JOINTS:
                lo, hi = lab.calibration[joint]["range_min"], lab.calibration[joint]["range_max"]
                if type(after[joint]) is not int or not lo <= after[joint] <= hi:
                    raise HardwareError("Encoder is outside the admitted calibration")
            measured = {j: normalized(j, after[j], lab.calibration) for j in JOINTS}
            prior = {j: normalized(j, before[j], lab.calibration) for j in JOINTS}
            if any(abs(measured[j] - prior[j]) > MAX_DRIFT[j] for j in JOINTS):
                raise ValueError("Robot moved during camera/encoder acquisition")
            skew_ms = (end_ns - begin_ns) / 1e6
            if skew_ms > MAX_SKEW_MS:
                raise ValueError("Camera/encoder read interval exceeds the skew bound")
            image = capture["jpeg"]
            value = {
                "schema_version": 1,
                "observation_id": uuid.uuid4().hex,
                "target_session_id": self.target_session_id,
                "agent_session_id": self.owner,
                "authority_epoch": self.epoch,
                "controller_generation": self.generation,
                "episode_id": None,
                "revision": self.command_revision,
                "clock_policy": "physical_continuous",
                "environment_kind": "physical",
                "embodiment_manifest_id": "so101-position-segment-v1",
                "calibration_sha256": self.calibration_sha256,
                "robot_id": lab.robot_id,
                "joint_names": list(JOINTS),
                "joints": measured,
                "targets": dict(lab.action),
                "units": dict(UNITS),
                "joint_limits": control_limits(lab.calibration),
                "limits": dict(LIMITS),
                "captured_monotonic_ns": begin_ns,
                "valid_until_monotonic_ns": begin_ns + int(MAX_AGE_MS * 1e6),
                "joint_before_monotonic_ns": before_ns,
                "joint_after_monotonic_ns": end_ns,
                "camera_read_start_monotonic_ns": capture["read_start_monotonic_ns"],
                "camera_read_end_monotonic_ns": capture["read_end_monotonic_ns"],
                "image_sensor_timestamp_verified": False,
                "sensor_time_policy": "host_read_interval",
                "sensor_read_skew_ms": skew_ms,
                "image_data_url": "data:image/jpeg;base64," + base64.b64encode(image).decode(),
                "image_sha256": hashlib.sha256(image).hexdigest(),
                "source_age_ms": (time.monotonic_ns() - begin_ns) / 1e6,
                "skills": [
                    {
                        "name": "joint_position_segment",
                        "units": dict(UNITS),
                        "limits": dict(LIMITS),
                        "cartesian": False,
                        "contact_force": False,
                    }
                ],
                "fault": lab.error,
            }
            value["observation_sha256"] = digest(value)
            self.observations[value["observation_id"]] = value
            while len(self.observations) > 16:
                self.observations.pop(next(iter(self.observations)))
            encoded_bytes = len(canonical(value))
            if encoded_bytes > 2 * 1024**2:
                raise ValueError("Admitted camera observation exceeds the image bound")
            durable_json(self.observations_dir / (value["observation_id"] + ".json"), value)
            self.observation_bytes += encoded_bytes
            self.session_observation_bytes += encoded_bytes
            return value

    def execute(self, command_id, args):
        required = {"observation_id", "observation_sha256", "targets", "units", "duration_seconds"}
        optional = {
            "proposal_id",
            "provider_model",
            "model_observation_id",
            "model_observation_sha256",
            "prompt_version",
            "prompt_sha256",
        }
        if not required <= set(args) or set(args) - required - optional:
            raise ValueError(
                "A segment needs its original observation, named targets, units and duration"
            )
        prompt_keys = {"prompt_version", "prompt_sha256"}
        if prompt_keys & set(args) and (
            not prompt_keys <= set(args)
            or not isinstance(args["prompt_version"], str)
            or re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", args["prompt_version"]) is None
            or not isinstance(args["prompt_sha256"], str)
            or re.fullmatch(r"[a-f0-9]{64}", args["prompt_sha256"]) is None
        ):
            raise ValueError("Invalid prompt provenance")
        if "proposal_id" in args:
            identity(args["proposal_id"])
        if "provider_model" in args and (
            not isinstance(args["provider_model"], str)
            or not 1 <= len(args["provider_model"]) <= 192
        ):
            raise ValueError("Invalid provider model provenance")
        if "model_observation_id" in args:
            identity(args["model_observation_id"])
        if "model_observation_sha256" in args and (
            not isinstance(args["model_observation_sha256"], str)
            or not re.fullmatch(r"[a-f0-9]{64}", args["model_observation_sha256"])
        ):
            raise ValueError("Invalid model observation provenance")
        if self.active_id or self.segment:
            raise ValueError("Another segment is active")
        observed = self.observations.get(identity(args["observation_id"]))
        if observed is None or args["observation_sha256"] != observed["observation_sha256"]:
            raise ValueError("Original observation is unknown or changed")
        if time.monotonic_ns() >= observed["valid_until_monotonic_ns"]:
            raise ValueError("Original physical observation expired")
        targets, units = args["targets"], args["units"]
        if (
            not isinstance(targets, dict)
            or not targets
            or not set(targets) <= set(JOINTS)
            or not isinstance(units, dict)
            or units != {j: UNITS[j] for j in targets}
        ):
            raise ValueError("Segment targets need exact embodiment joints and units")
        duration = number(args["duration_seconds"], 0.5, 5, "segment duration")
        owner, epoch = self.owner, self.epoch
        with self.lab.bus.lock:
            raw = self.lab.bus.positions()
            initial = self.measured(raw)
            if any(abs(initial[j] - observed["joints"][j]) > MAX_DRIFT[j] for j in JOINTS):
                raise ValueError("Measured robot drift invalidated the proposal")
            admitted = dict(initial)
            ranges = control_limits(self.lab.calibration)
            for j, value in targets.items():
                target = number(value, ranges[j]["min"], ranges[j]["max"], "joint target")
                distance = abs(target - initial[j])
                if distance > (10 if j == "gripper" else 5):
                    raise ValueError("Segment delta exceeds the admitted envelope")
                if 1.875 * distance / duration > 10 or 5.774 * distance / duration**2 > 20:
                    raise ValueError("Segment speed or acceleration exceeds its profile")
                admitted[j] = target
            with self.lab.lifecycle_lock:
                self.validate_owner(owner, epoch)
                self.command_revision += 1
                self.segment = Segment(initial, admitted, duration, self.clock())
                self.active_id = command_id
                self.last_applied = dict(initial)
            return {
                "admitted_arguments": dict(args),
                "requested_targets": dict(admitted),
                "initial_measured_state": initial,
                "motor_target_readback_verified": False,
                "measured_completion_verified": False,
                "applied_targets": dict(initial),
                "units": dict(UNITS),
                "calibration_sha256": self.calibration_sha256,
                "observation_id": observed["observation_id"],
                "observation_sha256": observed["observation_sha256"],
                "started_monotonic_ns": time.monotonic_ns(),
            }

    def tick(self, raw):
        owner, epoch = self.owner, self.epoch
        generation, segment, active_id = self.generation, self.segment, self.active_id
        controller = self.lab.controller
        self.validate_owner(owner, epoch)
        measured = self.measured(raw)
        if segment is None:
            if self.last_applied is None or any(
                abs(measured[j] - self.last_applied[j]) > (6.0 if j == "gripper" else 3.0)
                for j in JOINTS
            ):
                raise HardwareError("Agent hold tracking tolerance exceeded")
            controller.repeat()
            with self.lab.lifecycle_lock:
                if (
                    generation == self.lab.generation
                    and self.owner == owner
                    and self.epoch == epoch
                ):
                    self.last_health = self.clock()
            return
        targets, arrived = segment.step(measured, self.clock())
        # Stop may revoke this tick while a USB write is in progress. It must
        # acquire only the short lifecycle publication lock, never this I/O.
        self.validate_owner(owner, epoch)
        sent = controller.command_raw(
            {j: denormalized(j, targets[j], self.lab.calibration) for j in JOINTS}
        )
        with self.lab.lifecycle_lock:
            if (
                generation != self.lab.generation
                or self.owner != owner
                or self.epoch != epoch
                or self.active_id != active_id
                or self.segment is not segment
            ):
                return None
            self.last_health = self.clock()
            self.lab.commanded_raw = dict(sent)
            self.lab.action = {j: normalized(j, sent[j], self.lab.calibration) for j in JOINTS}
            self.last_applied = dict(self.lab.action)
            record = self.receipt(active_id)
            record.update(applied_targets=dict(self.last_applied), final_measured_state=measured)
            if arrived:
                record.update(
                    status="completed",
                    measured_completion_verified=True,
                    completed_monotonic_ns=time.monotonic_ns(),
                    elapsed_seconds=self.clock() - segment.started,
                    reason="Measured segment target reached; task success remains unverified",
                )
                self.segment = None
                self.active_id = None
                self.cache[record["command_id"]] = record
                return record
            self.cache[record["command_id"]] = record
            return None
