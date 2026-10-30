#!/usr/bin/env python3
"""Private, loopback-only OpenJensen SO-101 hardware companion."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import mimetypes
import os
import platform
import secrets
import shutil
import signal
import stat
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from agent_control import AgentExecutor
from agent_control import strict_json as agent_json
from assembled_session import AssembledSession, CalibrationRevisions, empty_assembled_state
from calibration_draft import CalibrationDraft, file_hash
from camera import MAX_CAMERA_INDEX, CameraPool
from dataset import DatasetWriter, atomic_json, dataset_file, list_datasets, manifest, safe_slug
from device_registry import DeviceRegistry
from diagnostics import Diagnostics
from dispatch import MainThreadDispatcher
from hardware import (
    BAUDRATES,
    IDS,
    ISOLATION_WARNING,
    JOINTS,
    HardwareError,
    ServoBus,
    check_selected_motors,
    denormalized,
    normalized,
    probe_bus,
)
from lerobot_control import (
    PROFILE,
    LeRobotPositionControl,
    configure_passive_leader,
    prepare_single_turn_feedback,
)
from motion import MiddleMotion, RestMotion, control_limits, middle_targets, step_target
from rest_position import RestPositions
from verification import (
    MappingChecks,
    MappingVerification,
    empty_verification,
    latest_saved_mapping,
    saved_mapping,
)


class RequestError(ValueError):
    pass


def require(payload, key, expected):
    if payload.get(key) != expected:
        raise RequestError(f"Confirm this operation with {key}={expected!r}")


def integer(value, low, high, name):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise RequestError(f"{name} must be an integer between {low} and {high}")
    return value


def dependencies():
    result = {}
    for module, package in (
        ("lerobot", "lerobot"),
        ("serial", "pyserial"),
        ("cv2", "opencv-python-headless"),
        ("scservo_sdk", "feetech-servo-sdk"),
        ("pyarrow", "pyarrow"),
        ("PIL", "Pillow"),
        ("numpy", "numpy"),
    ):
        installed = importlib.util.find_spec(module) is not None
        try:
            version = importlib.metadata.version(package) if installed else None
        except importlib.metadata.PackageNotFoundError:
            version = "available" if installed else None
        result[module] = {"installed": installed, "version": version}
    return result


def is_usb_serial(port):
    device = port.device
    return getattr(port, "vid", None) is not None or device.startswith(
        ("/dev/cu.usb", "/dev/cu.wch", "/dev/ttyUSB", "/dev/ttyACM")
    )


def serial_ports():
    from serial.tools import list_ports

    return [
        {
            "device": port.device,
            "description": port.description,
            "manufacturer": port.manufacturer,
            "serial_number": port.serial_number,
        }
        for port in list_ports.comports()
        if is_usb_serial(port)
    ]


def valid_calibration(value):
    if not isinstance(value, dict) or set(value) != set(JOINTS):
        raise RequestError("Calibration must contain all six joints")
    for joint, entry in value.items():
        if (
            not isinstance(entry, dict)
            or entry.get("id") != IDS[joint]
            or entry.get("drive_mode") != 0
            or not all(
                isinstance(entry.get(key), int) and not isinstance(entry.get(key), bool)
                for key in ("id", "drive_mode", "homing_offset", "range_min", "range_max")
            )
        ):
            raise RequestError("Calibration has invalid joint data")
        if (
            not 0 <= entry["range_min"] < entry["range_max"] <= 4095
            or not -2047 <= entry["homing_offset"] <= 2047
        ):
            raise RequestError("Calibration range/offset is invalid")
    return value


class EncoderRangeError(HardwareError):
    def __init__(self, joint, value, low, high):
        outside = low - value if value < low else value - high
        direction = "below minimum" if value < low else "above maximum"
        distance = outside * (100 / (high - low) if joint == "gripper" else 360 / 4095)
        unit = "%" if joint == "gripper" else "°"
        self.details = {
            "code": "encoder_out_of_range",
            "joint": joint,
            "motor_id": IDS[joint],
            "measured_ticks": value,
            "min_ticks": low,
            "max_ticks": high,
            "outside_ticks": outside,
            "direction": direction,
            "outside_units": distance,
            "unit": "percent" if joint == "gripper" else "degrees",
        }
        super().__init__(
            f"{joint.replace('_', ' ').capitalize()} is outside its saved range "
            f"({value} ticks; allowed {low}–{high}; {outside} ticks {direction}, "
            f"about {distance:.2f}{unit}). Motion stopped. Support the arm; "
            "check its pose and saved calibration before enabling again"
        )


def validate_raw(raw, calibration=None, *, calibration_bounds=True, range_tolerance=0):
    if set(raw) != set(calibration if calibration is not None else JOINTS) or any(
        isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 4095
        for value in raw.values()
    ):
        raise HardwareError("Invalid encoder reading; inspect cabling and motor setup")
    if calibration is not None and calibration_bounds:
        for joint, value in raw.items():
            low, high = calibration[joint]["range_min"], calibration[joint]["range_max"]
            if not low - range_tolerance <= value <= high + range_tolerance:
                raise EncoderRangeError(joint, value, low, high)
    return raw


class RobotLab:
    def __init__(
        self,
        data_dir: Path,
        bus_factory=ServoBus,
        camera=None,
        port_provider=serial_ports,
        start_threads=True,
        assembled_factory=None,
        movement_detector=None,
        observe_timeout=45,
        verification_factory=None,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.datasets_dir = self.data_dir / "datasets"
        self.datasets_dir.mkdir(exist_ok=True)
        self.calibration_root = Path(
            os.environ.get(
                "HF_LEROBOT_CALIBRATION", Path.home() / ".cache/huggingface/lerobot/calibration"
            )
        )
        self.lock = threading.RLock()
        self.stop_lock = threading.Lock()
        self.lifecycle_lock = threading.RLock()
        self.stopping = False
        self.assembled_factory = assembled_factory
        self.movement_detector = movement_detector
        self.observe_timeout = observe_timeout
        self.device_registry = DeviceRegistry(self.data_dir)
        self.calibration_revisions = CalibrationRevisions(self.data_dir)
        self.assembled_session = None
        self.assembled_state = empty_assembled_state()
        saved = latest_saved_mapping(self.data_dir)
        if saved:
            self.assembled_state.update(saved, phase="complete", message="Saved motor IDs 1–6")
        self.verification_factory = verification_factory
        self.verification_session = None
        self.verification_state = empty_verification()
        self.mapping_checks = MappingChecks(self.data_dir)
        self.verification_revision = None
        self.bus_factory = bus_factory
        self.port_provider = port_provider
        self.camera = camera or CameraPool()
        self.bus = None
        self.leader = None
        self.pending_bus = None
        # Random initial generation fences delayed preflight requests across restarts.
        self.generation = secrets.randbits(48) + 1
        self.mode = "idle"
        self.port = self.robot_id = self.role = None
        self.torque = False
        self.error = None
        self.positions = {}
        self.action = {}
        self.commanded_raw = {}
        self.middle_motion = None
        self.rest_positions = RestPositions(self.data_dir)
        self.rest_targets = None
        self.rest_motion = None
        self.rest_phase = "idle"
        self.middle_tolerance = 2
        self.controller = None
        self.calibration = {}
        self.leader_calibration = {}
        self.original_calibration = None
        self.calibration_state = {
            "phase": "idle",
            "mins": {},
            "maxes": {},
            "offsets": {},
            "path": None,
        }
        self.calibration_draft = CalibrationDraft(self.data_dir)
        self.calibration_profile_hash = None
        try:
            draft = self.calibration_draft.read()
            if draft:
                self.calibration_state = dict(
                    draft["state"],
                    phase="interrupted",
                    interrupted_stage=draft["state"]["phase"],
                    recoverable=True,
                    error=None,
                    robot_id=draft["robot_id"],
                    role=draft["role"],
                    port=draft["port"],
                )
        except (ValueError, KeyError, TypeError, OSError) as error:
            self.calibration_state["error"] = f"Calibration checkpoint needs inspection: {error}"
        self.heartbeat = time.monotonic()
        self.last_jog = 0.0
        self.writer = None
        self.recording_error = None
        self.camera_index = None
        self.next_record = 0.0
        self.finished = threading.Event()
        self.threads = []
        self.agent = AgentExecutor(self)
        self.last_raw = {}
        self.last_fault = None
        self.motor_alarms = {}
        self.range_excursions = {}
        self.diagnostics = Diagnostics(self.data_dir)
        if start_threads:
            for target in (self._loop, self._watchdog):
                thread = threading.Thread(target=target, daemon=True)
                thread.start()
                self.threads.append(thread)

    def _diagnostic_context(self):
        return {
            "generation": self.generation,
            "mode": self.mode,
            "robot_id": self.robot_id,
            "role": self.role,
            "port": self.port,
            "torque_enabled": self.torque,
            "heartbeat_age_seconds": round(max(0, time.monotonic() - self.heartbeat), 3),
            "raw_positions": {key: dict(value) for key, value in dict(self.last_raw).items()},
            "motor_alarms": {
                source: dict(alarms) for source, alarms in dict(self.motor_alarms).items()
            },
            "calibration_excursions": {
                source: dict(value) for source, value in dict(self.range_excursions).items()
            },
            "commanded_raw": dict(self.commanded_raw),
            "targets": dict(self.action),
            "calibration": {joint: dict(entry) for joint, entry in dict(self.calibration).items()},
            "fault": self.last_fault
            if self.last_fault and self.last_fault.get("generation") == self.generation
            else None,
        }

    def _read_positions(self, bus, calibration=None, *, source="follower"):
        try:
            raw = bus.positions()
        except HardwareError as error:
            self.last_fault = {
                "source": source,
                "generation": self.generation,
                "message": str(error),
                "exception": type(error).__name__,
                **getattr(error, "details", {"code": "motor_read_failed"}),
            }
            self.diagnostics.emit("motor_fault", fault=self.last_fault)
            raise
        alarms = {
            str(motor): status
            for motor, status in dict(getattr(bus, "motor_status", {})).items()
            if status
        }
        previous_alarms = self.motor_alarms.get(source, {})
        self.motor_alarms[source] = alarms
        # Capture the offending reading before validation; no additional serial I/O.
        self.last_raw[source] = {
            joint: value for joint, value in raw.items() if joint in JOINTS and type(value) is int
        }
        try:
            # LeRobot normalizes observations without rejecting sampled-range
            # crossings: degree observations remain measured, gripper values
            # saturate at 0/100. Admission and agent execution remain strict.
            result = validate_raw(
                raw,
                calibration,
                calibration_bounds=self.mode not in {"manual", "leader", "handguide"},
            )
        except HardwareError as error:
            self.last_fault = {
                "source": source,
                "generation": self.generation,
                "message": str(error),
                "exception": type(error).__name__,
                **getattr(error, "details", {"code": "invalid_encoder_reading"}),
                "raw_positions": dict(self.last_raw[source]),
                "calibration": {joint: dict(entry) for joint, entry in (calibration or {}).items()},
            }
            self.diagnostics.emit("encoder_fault", fault=self.last_fault)
            raise
        excursions = {
            joint: {
                "measured_ticks": value,
                "min_ticks": calibration[joint]["range_min"],
                "max_ticks": calibration[joint]["range_max"],
            }
            for joint, value in raw.items()
            if calibration is not None
            and not calibration[joint]["range_min"] <= value <= calibration[joint]["range_max"]
        }
        previous_joints = set(self.range_excursions.get(source, {}))
        self.range_excursions[source] = excursions
        if set(excursions) != previous_joints:
            self.diagnostics.emit(
                "calibration_endpoint_observed" if excursions else "calibration_endpoint_cleared",
                reason=(
                    f"{source.capitalize()} {', '.join(j.replace('_', ' ') for j in excursions)} "
                    "crossed a sampled calibration endpoint; bounded control continues"
                    if excursions
                    else f"{source.capitalize()} readings returned within sampled endpoints"
                ),
                source=source,
                context=self._diagnostic_context(),
            )
        if alarms != previous_alarms:
            self.diagnostics.emit(
                "gripper_load_alarm" if alarms else "gripper_load_cleared",
                reason=(
                    f"{source.capitalize()} gripper overload alarm: control continues with "
                    "the motor's configured torque/current protection"
                    if alarms
                    else f"{source.capitalize()} gripper overload alarm cleared"
                ),
                source=source,
                status_errors=alarms,
                context=self._diagnostic_context(),
            )
        if self.mode != "idle":
            self.diagnostics.sample(context=self._diagnostic_context())
        return result

    def calibration_path(self, robot_id, role):
        safe_slug(robot_id)
        if role not in ("follower", "leader"):
            raise RequestError("Role must be follower or leader")
        device = next((d for d in self.device_registry.read() if d["robot_id"] == robot_id), None)
        if device and device.get("profile_role"):
            role = device["profile_role"]
            robot_id = safe_slug(device.get("calibration_id", robot_id))
        return (
            self.calibration_root
            / ("robots/so_follower" if role == "follower" else "teleoperators/so_leader")
            / f"{robot_id}.json"
        )

    def load_calibration(self, robot_id, role, port=None):
        path = self.calibration_path(robot_id, role)
        if not path.is_file():
            raise RequestError(f"No saved calibration for {robot_id}; complete calibration first")
        calibration = valid_calibration(json.loads(path.read_text()))
        if port and not self.calibration_revisions.fresh(port, robot_id, role, path):
            raise RequestError(
                "Motor IDs changed on this port; save a new calibration before control"
            )
        return calibration

    def calibration_export(self, robot_id, role):
        with self.lock:
            info = self.profile_info(robot_id, role)
            if not info["valid"]:
                raise RequestError(info["error"] or "No valid calibration to export")
            return {
                "format": "openjensen.so101.calibration",
                "version": 1,
                "robot_id": robot_id,
                "role": role,
                "calibration": self.load_calibration(robot_id, role),
            }

    def calibration_import(self, payload):
        require(payload, "confirm", "IMPORT CALIBRATION")
        if set(payload) - {"confirm", "robot_id", "role", "document", "port", "replace_existing"}:
            raise RequestError("Invalid calibration import request")
        robot_id, role = payload.get("robot_id"), payload.get("role")
        path = self.calibration_path(robot_id, role)
        replace = payload.get("replace_existing", False)
        if type(replace) is not bool:
            raise RequestError("Choose whether to replace the existing profile")
        document = payload.get("document")
        if not isinstance(document, dict):
            raise RequestError("Choose a calibration JSON file")
        if document.get("format") == "openjensen.so101.calibration":
            if (
                set(document) != {"format", "version", "robot_id", "role", "calibration"}
                or type(document.get("version")) is not int
                or document["version"] != 1
            ):
                raise RequestError("Unsupported calibration export version or fields")
            safe_slug(document.get("robot_id"))
            if document.get("role") != role:
                raise RequestError(
                    "This file is for a different arm role; select its leader or follower role"
                )
            document = document["calibration"]
        calibration = valid_calibration(document)
        fields = ("id", "drive_mode", "homing_offset", "range_min", "range_max")
        if any(set(entry) != set(fields) for entry in calibration.values()):
            raise RequestError("Calibration contains unsupported joint fields")
        calibration = {joint: {key: calibration[joint][key] for key in fields} for joint in JOINTS}
        with self.lock:
            self._idle()
            generation = self.generation
            if path.is_symlink() or (path.exists() and not path.is_file()):
                raise RequestError("Calibration destination must be a regular file")
            if path.exists() and not replace:
                raise RequestError(
                    "Confirm replacing the existing calibration; a backup will be kept"
                )
            verified = payload.get("port") is not None
            port = self._port(payload["port"]) if verified else None
            if verified:
                bus = self.bus_factory(port)
                try:
                    with bus.lock:
                        bus.check_six()
                        actual = bus.read_calibration()
                        different = [
                            joint.replace("_", " ")
                            for joint in JOINTS
                            if actual.get(joint) != calibration[joint]
                        ]
                        if different:
                            raise RequestError(
                                "Calibration file does not match the connected motors: "
                                + ", ".join(different)
                                + ". Choose the file from the computer that last "
                                "calibrated this arm."
                            )
                finally:
                    # Import verification is read only, including on failure.
                    bus.close()
            with self.stop_lock:
                if generation != self.generation or self.stopping:
                    raise RequestError("Calibration import cancelled by Stop")
                backup = None
                if path.exists():
                    backup = path.with_name(path.stem + f".backup-{time.time_ns()}.json")
                    shutil.copy2(path, backup)
                    backup.chmod(0o600)
                try:
                    atomic_json(path, calibration)
                    path.chmod(0o600)
                    if verified:
                        self.calibration_revisions.acknowledge(port, robot_id, role, path)
                except Exception:
                    if backup is not None:
                        shutil.copy2(backup, path)
                    else:
                        path.unlink(missing_ok=True)
                    raise
                if verified and self.error in (
                    "Motor calibration differs from the saved profile; recalibrate before motion",
                    "Leader calibration differs from its saved profile",
                ):
                    self.error = None
                self.diagnostics.emit(
                    "calibration_imported",
                    robot_id=robot_id,
                    role=role,
                    verified=verified,
                    previous_profile_backed_up=backup is not None,
                )
            return {
                **self.status(),
                "calibration_transfer": {
                    "robot_id": robot_id,
                    "role": role,
                    "verified": verified,
                    "backup_created": backup is not None,
                },
            }

    def _port(self, value):
        if not isinstance(value, str) or value not in {
            entry["device"] for entry in self.port_provider()
        }:
            raise RequestError("Select a currently connected serial device")
        return value

    def _idle(self):
        if (
            self.mode != "idle"
            or self.bus is not None
            or self.pending_bus is not None
            or self.assembled_session is not None
            or self.verification_session is not None
            or self.stopping
        ):
            raise RequestError("Stop the current session before changing hardware setup")

    def profile_info(self, robot_id, role, port=None):
        path = self.calibration_path(robot_id, role)
        result = {
            "robot_id": robot_id,
            "role": role,
            "path": str(path),
            "valid": False,
            "setup_port": port,
            "setup_fresh": False,
            "sha256": None,
            "error": None,
        }
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 65536:
                raise ValueError("No valid saved calibration")
            raw = path.read_bytes()
            valid_calibration(json.loads(raw))
            if port is None:
                port = self.calibration_revisions.connection(robot_id, role, path)
            result.update(
                valid=True,
                setup_port=port,
                sha256=hashlib.sha256(raw).hexdigest(),
                setup_fresh=not port
                or self.calibration_revisions.fresh(port, robot_id, role, path),
            )
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            result["error"] = str(error)
        return result

    def status(self):
        # Snapshot only; polling never opens a camera or touches the motors.
        deps = dependencies()
        hardware = all(deps[key]["installed"] for key in ("serial", "scservo_sdk"))
        calibrations = []
        for role, pattern in (
            ("follower", "robots/so_follower/*.json"),
            ("leader", "teleoperators/so_leader/*.json"),
        ):
            for path in self.calibration_root.glob(pattern):
                try:
                    safe_slug(path.stem)
                except ValueError:
                    continue
                calibrations.append(self.profile_info(path.stem, role))
        for device in self.device_registry.read():
            for role in ("leader", "follower"):
                if not any(
                    p["robot_id"] == device["robot_id"] and p["role"] == role for p in calibrations
                ):
                    info = self.profile_info(device["robot_id"], role)
                    if info["valid"]:
                        calibrations.append(info)
        writer = self.writer
        with self.lifecycle_lock:
            assembled = json.loads(json.dumps(self.assembled_state))
            copy_verification = json.loads(json.dumps(self.verification_state))
            middle_motion = self.middle_motion
            rest_motion, rest_phase = self.rest_motion, self.rest_phase
            rest_calibration = dict(self.calibration)
            rest_targets = (
                self.rest_targets
                if self.mode in {"manual", "leader", "handguide"}
                and set(self.calibration) == set(JOINTS)
                else None
            )
        try:
            assembled["calibration_required"] = self.calibration_revisions.required(
                assembled["port"]
            )
        except (ValueError, OSError) as error:
            assembled["calibration_required"] = True
            assembled["error"] = str(error)
        return {
            "ok": True,
            "version": "1",
            "platform": platform.system(),
            "backend": "feetech-sdk",
            "dataset_format": "lerobot-v3",
            "controller": dict(PROFILE),
            "dependencies": deps,
            "capabilities": {
                "probe": hardware,
                "camera": deps["cv2"]["installed"],
                "motor_setup": hardware,
                "assembled_setup": hardware,
                "mapping_verification": hardware,
                "arm_readiness": hardware,
                "calibration": hardware,
                "calibration_persistence": hardware,
                "calibration_transfer": True,
                "device_registry": True,
                "control": hardware,
                "lerobot_position_control": hardware,
                "rest_position": hardware,
                "persistent_diagnostics": True,
                "gripper_load_tolerance": True,
                "lerobot_observation_ranges": True,
                "background_recording_io": True,
                "agent_control": hardware,
                "move_to_middle": hardware,
                "middle_tolerance": hardware,
                "partial_control": hardware,
                "joint_sliders": hardware,
                "large_joint_steps": hardware,
                "recording": hardware
                and all(deps[key]["installed"] for key in ("pyarrow", "PIL", "numpy")),
            },
            "diagnostics": self.diagnostics.snapshot(),
            "agent_control": self.agent.status(),
            "session": {
                "generation": self.generation,
                "mode": self.mode,
                "robot_id": self.robot_id,
                "port": self.port,
                "role": self.role,
                "positions": dict(self.positions),
                "active_joints": list(self.calibration)
                if self.mode in {"manual", "leader", "handguide", "agent"}
                else [],
                "targets": dict(self.action) if self.mode in {"manual", "leader", "agent"} else {},
                "joint_limits": control_limits(self.calibration) if self.mode == "manual" else {},
                "slider_targets": dict(self.action) if self.mode == "manual" else {},
                "torque_limits": dict(self.controller.torque_limits) if self.controller else {},
                "torque_enabled": self.torque,
                "rest_position": {
                    "saved": rest_targets is not None,
                    "phase": rest_phase,
                    "progress": rest_motion.progress
                    if rest_motion
                    else (1.0 if rest_phase == "holding" else 0.0),
                    "targets": {
                        j: normalized(j, v, rest_calibration)
                        for j, v in (rest_targets or {}).items()
                    },
                },
                "centering": {
                    "active": middle_motion is not None,
                    "progress": middle_motion.progress if middle_motion else 1.0,
                    "tolerance": self.middle_tolerance,
                },
                "watchdog_seconds": None
                if self.mode == "calibration"
                else (10.0 if self.mode in {"setup", "verification"} else 2.0),
                "error": self.error,
            },
            "calibration": json.loads(json.dumps(self.calibration_state)),
            "assembled": assembled,
            "verification": copy_verification,
            "mapping_checks": self.mapping_checks.snapshot(assembled.get("port")),
            "recording": {
                "actual_fps": writer.actual_fps if writer else None,
                "active": writer is not None and not writer.finalized,
                "dataset_id": writer.id if writer else None,
                "episode_active": writer.episode_active if writer else False,
                "frames": writer.frames if writer else 0,
                "episodes": len(writer.episodes) if writer else 0,
                "fps": writer.fps if writer else 10,
                "error": self.recording_error,
            },
            "calibrations": calibrations,
            "datasets": list_datasets(self.datasets_dir),
            "joints": [
                {
                    "name": joint,
                    "id": IDS[joint],
                    "label": joint.replace("_", " ").capitalize(),
                    "unit": "percent" if joint == "gripper" else "degrees",
                }
                for joint in JOINTS
            ],
            "setup_order": list(reversed(JOINTS)),
        }

    def devices(self):
        error = None
        try:
            cameras = self.camera.discover()
        except Exception as caught:
            cameras, error = [], str(caught)
        with self.lock:
            ports = self.port_provider()
            catalog = self.device_registry.discover(ports, self.status()["calibrations"])
            for device in catalog:
                try:
                    saved_mapping(self.data_dir, device["port"])
                    device["motor_ids"] = True
                except ValueError, OSError:
                    device["motor_ids"] = False
                profile = self.profile_info(
                    device["robot_id"], device["profile_role"] or "follower"
                )
                device["calibration"] = (
                    "saved"
                    if profile["valid"] and profile["setup_fresh"]
                    else "needs_attention"
                    if profile["valid"]
                    else "missing"
                )
            return {
                "ports": ports,
                "saved_devices": catalog,
                "cameras": cameras,
                "camera_error": error,
            }

    def configure_device(self, payload):
        with self.lock:
            self._idle()
            if self.writer is not None:
                raise RequestError("Finish recording before changing devices")
            self.device_registry.save(payload, self.devices()["saved_devices"])
            return self.devices()

    def probe(self, payload):
        with self.lock:
            self._idle()
            discover = payload.get("discover", False)
            if type(discover) is not bool:
                raise RequestError("Motor discovery must be true or false")
            port = self._port(payload.get("port"))
            baud = payload.get("baudrate", 1_000_000)
            if baud not in BAUDRATES:
                raise RequestError("Unsupported STS3215 baudrate")
            if payload.get("robot_id") is not None:
                self.calibration_path(payload["robot_id"], payload.get("role", "follower"))
            generation = self.generation
            bus = self.bus_factory(port, baud)
            try:
                with bus.lock:
                    result = {
                        "port": port,
                        "baudrate": baud,
                        "checked_at": time.time(),
                        **probe_bus(
                            bus,
                            discover=discover,
                            is_current=lambda: generation == self.generation and not self.stopping,
                        ),
                    }
                    if payload.get("robot_id") is not None:
                        robot_id, role = payload["robot_id"], payload.get("role", "follower")
                        info = self.profile_info(robot_id, role, port)
                        profile = {
                            "robot_id": robot_id,
                            "role": role,
                            "sha256": info["sha256"],
                            "state": "missing"
                            if not self.calibration_path(robot_id, role).exists()
                            else "invalid"
                            if not info["valid"]
                            else "stale"
                            if not info["setup_fresh"]
                            else "unavailable",
                            "joints": {},
                            "error": info["error"],
                        }
                        if result["ready"] and info["valid"] and info["setup_fresh"]:
                            try:
                                saved = self.load_calibration(robot_id, role, port)
                                actual = bus.read_calibration()
                                profile["joints"] = {j: actual.get(j) == saved[j] for j in JOINTS}
                                profile["state"] = (
                                    "matched" if all(profile["joints"].values()) else "mismatch"
                                )
                            except (ValueError, OSError, HardwareError) as error:
                                profile["error"] = str(error)
                        result["profile"] = profile
                    return result
            finally:
                # Probe is read only: closing must never write torque registers.
                bus.close()

    def setup_motor(self, payload):
        require(payload, "isolated", True)
        require(payload, "confirm", "ASSIGN MOTOR ID")
        joint = payload.get("joint")
        if joint not in JOINTS:
            raise RequestError("Select one of the six SO-101 joints")
        with self.lock:
            self._idle()
            baud = payload.get("baudrate", 1_000_000)
            if baud not in BAUDRATES:
                raise RequestError("Unsupported baudrate")
            port = self._port(payload.get("port"))
            generation = self.generation
            bus = self.bus_factory(port, baud)
            with self.stop_lock, self.lifecycle_lock:
                if generation != self.generation or self.stopping:
                    bus.close()
                    raise RequestError("Motor setup cancelled by Stop")
                self.pending_bus = bus
            try:
                with bus.lock:
                    scan = []
                    for motor_id in range(1, 254):
                        if generation != self.generation:
                            raise RequestError("Motor setup cancelled by Stop")
                        scan.extend(probe_bus(bus, [motor_id], repeats=1)["motors"])
                    if len(scan) != 1:
                        raise RequestError(
                            "Expected exactly one responding ID; physically isolate the "
                            "selected motor. " + ISOLATION_WARNING
                        )
                    current = scan[0]["id"]
                    if [bus.ping(current) for _ in range(3)] != [777] * 3:
                        raise RequestError(
                            "Motor replies are intermittent or model is not STS3215; no ID written"
                        )
                    if generation != self.generation:
                        raise RequestError("Motor setup cancelled by Stop")
                    self.calibration_revisions.mark_dirty(port)
                    if generation != self.generation:
                        raise RequestError("Motor setup cancelled by Stop")
                    # Once started, finish the bounded ID/baud/re-lock transaction
                    # under the bus lock before Stop can close this port.
                    unlocked = relocked = False
                    try:
                        bus.write("Torque_Enable", current, 0)
                        bus.write("Lock", current, 0)
                        unlocked = True
                        bus.write("ID", current, IDS[joint])
                        bus.write("Baud_Rate", IDS[joint], 0)
                        bus.port.setBaudRate(1_000_000)
                        if [bus.ping(IDS[joint]) for _ in range(3)] != [777] * 3:
                            raise HardwareError(
                                "ID write needs inspection: motor did not verify at new ID/baudrate"
                            )
                        bus.write("Lock", IDS[joint], 1)
                        if bus.read("Lock", IDS[joint]) != 1:
                            raise HardwareError("Motor EEPROM could not be locked at its new ID")
                        relocked = True
                    finally:
                        if unlocked and not relocked:
                            # An acknowledged ID may be lost on a cable error.
                            # Try only EEPROM re-lock at the two possible IDs.
                            for motor_id in {current, IDS[joint]}:
                                try:
                                    bus.write("Lock", motor_id, 1)
                                except Exception:
                                    pass
            finally:
                with self.lifecycle_lock:
                    if self.pending_bus is bus:
                        self.pending_bus = None
                bus.close()
            return {
                "assigned": {"joint": joint, "id": IDS[joint], "baudrate": 1_000_000},
                "warning": ISOLATION_WARNING,
            }

    def _assembled_changed(self, controller, snapshot, terminal):
        # Engine progress may run under its I/O lock. Never acquire lab.lock here.
        with self.lifecycle_lock:
            if self.assembled_session is not controller:
                return
            self.assembled_state = snapshot
            self.torque = bool(snapshot.get("motion_active"))
            if terminal:
                self.assembled_session = None
                self.mode = "idle"
                self.torque = False
                if snapshot.get("error"):
                    self.error = snapshot["error"]

    def assembled_start(self, payload):
        require(payload, "confirm", "START ASSEMBLED SETUP")
        if payload.get("supported") is not True:
            raise RequestError(
                "Support the SO-101 arm and confirm its six STS3215 motors "
                "before experimental ID reboots"
            )
        with self.lock:
            self._idle()
            port = self._port(payload.get("port"))
            generation = self.generation
            # Preserve old profiles, but revoke their authority before the first
            # possible ID write, including interrupted/partially completed work.
            self.calibration_revisions.mark_dirty(port)
            with self.stop_lock, self.lifecycle_lock:
                if generation != self.generation or self.stopping:
                    raise RequestError("Setup was cancelled before it started")
                controller = AssembledSession(
                    port,
                    self.data_dir,
                    self._assembled_changed,
                    engine_factory=self.assembled_factory,
                    detector=self.movement_detector,
                    observe_timeout=self.observe_timeout,
                    verified_so101=True,
                )
                self.assembled_session = controller
                self.assembled_state = controller.snapshot()
                self.generation += 1
                self.mode, self.port = "setup", port
                self.robot_id = self.role = None
                self.positions = {}
                self.action = {}
                self.torque = False
                self.error = None
                self.heartbeat = time.monotonic()
        controller.start()
        return self.status()

    def assembled_action(self, action, payload):
        with self.lifecycle_lock:
            controller = self.assembled_session
            if controller is None or self.mode != "setup" or self.stopping:
                raise RequestError("Start assembled motor setup first")
        # Reads and background-thread publication never hold the global lab lock.
        if action == "observe":
            controller.observe(payload.get("joint"))
        elif action == "wiggle":
            require(payload, "confirm", "WIGGLE MOTOR")
            if payload.get("supported") is not True:
                raise RequestError("Support the arm and position all joints clear of their stops")
            controller.wiggle(payload.get("motor_id"))
        elif action == "accept":
            if payload.get("method") == "wiggle":
                require(payload, "confirm", "CONFIRM OBSERVED JOINT")
                controller.accept_wiggle(payload.get("joint"), payload.get("motor_id"))
            elif payload.get("method") in (None, "hand"):
                controller.accept(payload.get("joint"), payload.get("motor_id"))
            else:
                raise RequestError("Choose hand or wiggle identification")
        elif action == "finish":
            require(payload, "confirm", "SAVE MOTOR IDS")
            controller.finish()
        else:
            raise RequestError("Unknown assembled setup operation")
        return self.status()

    def _verification_changed(self, controller, snapshot, terminal):
        with self.lifecycle_lock:
            if self.verification_session is not controller:
                return
            self.verification_state = snapshot
            self.torque = bool(snapshot["motion_active"])
            if terminal:
                self.verification_session = None
                self.mode = "idle"
                self.error = snapshot["error"]

    def mapping_wiggle(self, payload):
        require(payload, "confirm", "WIGGLE SAVED MOTOR")
        if payload.get("supported") is not True:
            raise RequestError("Support the arm with all joints clear of their stops")
        joint = payload.get("joint")
        if joint not in JOINTS:
            raise RequestError("Choose one of the six SO-101 joints")
        with self.lock:
            self._idle()
            port = self._port(payload.get("port"))
            generation = self.generation
            if self.torque:
                raise RequestError("Release motor torque before checking the mapping")
            saved = saved_mapping(self.data_dir, port)
            revision = self.mapping_checks.revision(port)
            with self.stop_lock, self.lifecycle_lock:
                if generation != self.generation or self.stopping:
                    raise RequestError("Motor verification was cancelled before it started")
                controller = MappingVerification(
                    port,
                    joint,
                    self.data_dir,
                    self._verification_changed,
                    engine_factory=self.verification_factory,
                )
                self.verification_session = controller
                self.verification_state = controller.snapshot()
                self.verification_revision = revision
                self.assembled_state.update(saved, phase="complete")
                self.generation += 1
                self.mode, self.port = "verification", port
                self.robot_id = self.role = None
                self.torque, self.error = False, None
                self.heartbeat = time.monotonic()
                controller.start()
        return self.status()

    def mapping_confirm(self, payload):
        require(payload, "confirm", "CONFIRM PHYSICAL JOINT")
        if type(payload.get("correct")) is not bool:
            raise RequestError("Choose correct joint or wrong joint")
        with self.lock, self.lifecycle_lock:
            self._idle()
            observed = self.verification_state
            if (
                observed["phase"] != "done"
                or observed["motion_active"]
                or observed["error"]
                or any(payload.get(key) != observed[key] for key in ("port", "joint", "run_id"))
            ):
                raise RequestError("Confirm a completed motion check for this joint")
            self.mapping_checks.record(observed, self.verification_revision, payload["correct"])
        return self.status()

    def _checkpoint_calibration(self, *, force=False):
        if self.original_calibration is None or self.calibration_state["phase"] not in {
            "center",
            "ranges",
        }:
            return
        self.calibration_draft.write(
            {
                "version": 1,
                "robot_id": self.robot_id,
                "role": self.role,
                "port": self.port,
                "profile_hash": self.calibration_profile_hash,
                "original": self.original_calibration,
                "state": json.loads(json.dumps(self.calibration_state)),
            },
            force=force,
        )

    def calibration_cancel(self):
        self.stop()
        with self.lock:
            self.calibration_draft.archive("cancelled")
            self.calibration_state = {
                "phase": "idle",
                "mins": {},
                "maxes": {},
                "offsets": {},
                "path": None,
            }
            self.error = None
        return self.status()

    def _calibration_resume(self):
        with self.lock:
            self._idle()
            draft = self.calibration_draft.read()
            if draft is None:
                raise RequestError("No captured calibration to resume")
            robot_id, role = draft["robot_id"], draft["role"]
            path = self.calibration_path(robot_id, role)
            if file_hash(path) != draft["profile_hash"]:
                raise RequestError(
                    "Saved calibration changed after capture; inspect before resuming"
                )
            port = self._port(draft["port"])
            state = dict(draft["state"], path=str(path))
            expected = draft["original"]
            temporary = (
                {
                    j: dict(
                        expected[j], homing_offset=state["offsets"][j], range_min=0, range_max=4095
                    )
                    for j in JOINTS
                }
                if state["phase"] == "ranges"
                else expected
            )
            generation = self.generation
            bus = self.bus_factory(port)
            with self.stop_lock, self.lifecycle_lock:
                if generation != self.generation:
                    bus.close()
                    raise RequestError("Resume cancelled by Stop")
                self.pending_bus = bus
            known_configuration = False
            try:
                with bus.lock:
                    bus.check_six()
                    current = bus.read_calibration()
                    if current not in (expected, temporary):
                        raise RequestError(
                            "Motor configuration changed; captured calibration cannot be resumed"
                        )
                    known_configuration = True
                    with self.lifecycle_lock:
                        if generation != self.generation:
                            raise RequestError("Resume cancelled by Stop")
                        self.original_calibration = expected
                    bus.disable()
                    prepare_single_turn_feedback(bus, lambda: generation == self.generation)
                    if generation != self.generation:
                        raise RequestError("Resume cancelled by Stop")
                    if current != temporary:
                        bus.write_calibration(temporary)
                    if bus.read_calibration() != temporary:
                        raise HardwareError(
                            "Captured calibration settings did not verify; motion remains off"
                        )
                    for motor_id in IDS.values():
                        if generation != self.generation:
                            raise RequestError("Resume cancelled by Stop")
                        bus.write("Lock", motor_id, 0)
                        bus.write("Operating_Mode", motor_id, 0)
                    raw = self._read_positions(bus)
            except Exception:
                try:
                    with bus.lock:
                        bus.disable()
                        if known_configuration and generation == self.generation:
                            bus.write_calibration(expected)
                finally:
                    bus.close()
                    if self.pending_bus is bus:
                        self.pending_bus = None
                        self.original_calibration = None
                raise
            with self.stop_lock, self.lifecycle_lock:
                if generation != self.generation:
                    bus.close()
                    raise RequestError("Resume cancelled by Stop")
                self.bus, self.pending_bus = bus, None
                self.mode, self.port, self.robot_id, self.role = "calibration", port, robot_id, role
                self.generation += 1
                self.original_calibration = expected
                self.calibration_profile_hash = draft["profile_hash"]
                self.calibration_state = state
                self.positions, self.torque, self.error = raw, False, None
                self.heartbeat = time.monotonic()
        return self.status()

    def calibration_start(self, payload):
        require(payload, "confirm", "START CALIBRATION")
        resume = payload.get("resume", False)
        if type(resume) is not bool:
            raise RequestError("Resume must be a boolean")
        if resume:
            return self._calibration_resume()
        with self.lock:
            self._idle()
            robot_id = safe_slug(payload.get("robot_id"))
            role = payload.get("role", "follower")
            path = self.calibration_path(robot_id, role)
            port = self._port(payload.get("port"))
            generation = self.generation
            bus = self.bus_factory(port)
            with self.stop_lock, self.lifecycle_lock:
                if generation != self.generation:
                    bus.close()
                    raise RequestError("Session cancelled by Stop")
                self.pending_bus = bus
            try:
                with bus.lock:
                    if generation != self.generation:
                        raise RequestError("Session cancelled by Stop")
                    bus.check_six()
                    original = bus.read_calibration()
                    if generation != self.generation:
                        raise RequestError("Session cancelled by Stop")
                    bus.disable()
                    prepare_single_turn_feedback(bus, lambda: generation == self.generation)
                    for motor_id in IDS.values():
                        bus.write("Lock", motor_id, 0)
                        bus.write("Operating_Mode", motor_id, 0)
            except Exception:
                try:
                    with bus.lock:
                        if getattr(bus, "opened", True):
                            bus.disable()
                finally:
                    bus.close()
                    if self.pending_bus is bus:
                        self.pending_bus = None
                raise
            with self.stop_lock, self.lifecycle_lock:
                if generation != self.generation:
                    bus.close()
                    raise RequestError("Session cancelled by Stop")
                self.bus = bus
                self.pending_bus = None
                self.generation += 1
                self.mode, self.port, self.robot_id, self.role = "calibration", port, robot_id, role
                self.original_calibration = original
                self.calibration_profile_hash = file_hash(path)
                self.calibration_draft.archive("restarted")
                self.calibration_state = {
                    "phase": "center",
                    "mins": {},
                    "maxes": {},
                    "offsets": {},
                    "path": str(path),
                }
                self.torque = False
                self.heartbeat, self.error = time.monotonic(), None
                self._checkpoint_calibration(force=True)
        return self.status()

    def calibration_center(self, payload):
        with self.lock:
            if self.mode != "calibration" or self.calibration_state["phase"] != "center":
                raise RequestError(
                    "Start calibration and move the arm to its middle position first"
                )
            bus, generation = self.bus, self.generation
            if bus is None:
                raise RequestError("Calibration has stopped")
            with bus.lock:
                if generation != self.generation:
                    raise RequestError("Calibration has stopped")
                try:
                    for motor_id in IDS.values():
                        bus.write("Homing_Offset", motor_id, 0)
                        bus.write("Min_Position_Limit", motor_id, 0)
                        bus.write("Max_Position_Limit", motor_id, 4095)
                    raw = self._read_positions(bus)
                    # Match upstream sign-magnitude homing: raw 4095 needs
                    # offset 2048, clamped to the register's representable 2047.
                    offsets = {
                        joint: max(-2047, min(2047, position - 2047))
                        for joint, position in raw.items()
                    }
                    if any(abs(value) > 2047 for value in offsets.values()):
                        raise RequestError(
                            "Invalid middle pose encoder reading; "
                            "cancel and inspect the motor setup"
                        )
                    for joint, offset in offsets.items():
                        bus.write("Homing_Offset", IDS[joint], offset)
                    raw = self._read_positions(bus)
                except Exception as error:
                    raise HardwareError(f"Middle-position capture failed: {error}") from error
            # Publish only after a concurrent Stop has fully completed, or
            # before Stop can invalidate this stage. Never resurrect a phase.
            with self.stop_lock, self.lifecycle_lock:
                if generation != self.generation:
                    raise RequestError("Calibration has stopped")
                # Homing changes the encoder coordinate system. Do not compare
                # the first range sample against the pre-offset position.
                self.positions = dict(raw)
                self.calibration_state.update(
                    phase="ranges", offsets=offsets, mins=dict(raw), maxes=dict(raw)
                )
                self._checkpoint_calibration(force=True)
                self.heartbeat = time.monotonic()
        return self.status()

    def calibration_finish(self, payload):
        require(payload, "confirm", "SAVE CALIBRATION")
        with self.lock:
            if self.mode != "calibration" or self.calibration_state["phase"] != "ranges":
                raise RequestError(
                    "Capture the middle pose and move every joint through its range first"
                )
            bus, generation = self.bus, self.generation
            if bus is None:
                raise RequestError("Calibration has stopped")
            # Match Stop's lock order. EEPROM verification and file publication
            # form a single save operation; Stop cannot restore the old values
            # between writing the new motors and committing the matching file.
            with self.stop_lock:
                if generation != self.generation:
                    raise RequestError("Calibration has stopped")
                with bus.lock:
                    state = self.calibration_state
                    for joint in JOINTS:
                        if (
                            joint != "wrist_roll"
                            and state["maxes"][joint] - state["mins"][joint] < 100
                        ):
                            raise RequestError(
                                f"Move {joint.replace('_', ' ')} through more of its range "
                                "before saving"
                            )
                    calibration = {
                        joint: {
                            "id": IDS[joint],
                            "drive_mode": 0,
                            "homing_offset": state["offsets"][joint],
                            "range_min": 0 if joint == "wrist_roll" else state["mins"][joint],
                            "range_max": 4095 if joint == "wrist_roll" else state["maxes"][joint],
                        }
                        for joint in JOINTS
                    }
                    valid_calibration(calibration)
                    path = Path(state["path"])
                    bus.write_calibration(calibration)
                    if bus.read_calibration() != calibration:
                        raise HardwareError(
                            "Calibration read-back did not match; cancel and inspect wiring"
                        )
                    if path.exists():
                        backup = path.with_name(path.stem + f".backup-{time.time_ns()}.json")
                        backup.write_bytes(path.read_bytes())
                    atomic_json(path, calibration)
                    self.calibration_revisions.acknowledge(
                        self.port, self.robot_id, self.role, path
                    )
                    self.original_calibration = None
                    state["phase"] = "saved"
                    self.calibration_draft.archive("saved")
        self.stop()
        return self.status()

    def control_start(self, payload):
        require(payload, "confirm", "ENABLE MOTION")
        mode = payload.get("mode")
        if mode not in ("manual", "leader", "handguide"):
            raise RequestError("Choose manual, leader or handguide mode")
        selection = payload.get("joints", list(JOINTS))
        if (
            not isinstance(selection, list)
            or not selection
            or any(not isinstance(j, str) or j not in JOINTS for j in selection)
            or len(set(selection)) != len(selection)
        ):
            raise RequestError("Select at least one distinct SO-101 joint")
        joints = tuple(j for j in JOINTS if j in selection)
        if mode != "manual" and joints != JOINTS:
            raise RequestError("Selected-motor control is available only in joint control")
        to_middle = payload.get("move_to_middle", False)
        if type(to_middle) is not bool or (to_middle and mode != "manual"):
            raise RequestError("Move to recorded middle is available only for joint control")
        tolerance = payload.get("middle_tolerance", 2)
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(tolerance)
            or not 0.5 <= tolerance <= 5
            or ("middle_tolerance" in payload and mode != "manual")
        ):
            raise RequestError("Middle tolerance must be between 0.5 and 5 degrees / percent")
        with self.lock:
            self._idle()
            port = self._port(payload.get("port"))
            robot_id = safe_slug(payload.get("robot_id"))
            calibration = self.load_calibration(robot_id, "follower", port)
            calibration = {j: calibration[j] for j in joints}
            if to_middle:
                middle_targets(calibration)
            generation = self.generation
            control_generation = [generation]
            self.last_fault = None
            self.last_raw = {}
            self.motor_alarms = {}
            self.range_excursions = {}
            leader, control, bus = None, None, self.bus_factory(port)
            bus.joints = joints
            with self.stop_lock, self.lifecycle_lock:
                if generation != self.generation:
                    bus.close()
                    raise RequestError("Session cancelled by Stop")
                self.pending_bus = bus
            try:
                with bus.lock:
                    if joints == JOINTS:
                        bus.check_six()
                    else:
                        check_selected_motors(bus, joints)
                    if generation != self.generation:
                        raise RequestError("Session cancelled by Stop")
                    # Never rewrite saved EEPROM calibration when starting motion.
                    if bus.read_calibration() != calibration:
                        raise RequestError(
                            "Motor calibration differs from the saved profile; recalibrate "
                            "before motion"
                        )
                    if joints == JOINTS:
                        self.calibration_revisions.acknowledge(
                            port, robot_id, "follower", self.calibration_path(robot_id, "follower")
                        )
                    bus.disable()
                    prepare_single_turn_feedback(bus, lambda: generation == self.generation)
                    raw = self._read_positions(bus, calibration)
                    if any(
                        not calibration[j]["range_min"] <= raw[j] <= calibration[j]["range_max"]
                        for j in joints
                    ):
                        raise RequestError(
                            "A joint is outside its saved calibration range; reposition with "
                            "power off or recalibrate"
                        )
                    positions = {
                        joint: normalized(joint, value, calibration) for joint, value in raw.items()
                    }
                    if mode == "leader":
                        leader_port = self._port(payload.get("leader_port"))
                        if leader_port == port:
                            raise RequestError(
                                "Leader and follower must use separate serial devices"
                            )
                        leader_calibration = self.load_calibration(
                            payload.get("leader_id"), "leader", leader_port
                        )
                        leader = self.bus_factory(leader_port)
                        with leader.lock:
                            leader.check_six()
                            if leader.read_calibration() != leader_calibration:
                                raise RequestError(
                                    "Leader calibration differs from its saved profile"
                                )
                            configure_passive_leader(
                                leader, leader_calibration, lambda: generation == self.generation
                            )
                            self._read_positions(leader, leader_calibration, source="leader")
                        self.leader_calibration = leader_calibration
                    if mode != "handguide":
                        control = LeRobotPositionControl(
                            bus, calibration, lambda: control_generation[0] == self.generation
                        )
                        control.configure()
                        # Sample again after configuration, immediately before holding.
                        raw = self._read_positions(bus, calibration)
                        for joint in joints:
                            motor_id = IDS[joint]
                            if generation != self.generation:
                                raise RequestError("Session cancelled by Stop")
                            bus.write("Goal_Position", motor_id, raw[joint])
                            if bus.read("Goal_Position", motor_id) != raw[joint]:
                                raise HardwareError(f"{joint} hold target read-back did not match")
                        control.hold(raw)
                        for joint in joints:
                            motor_id = IDS[joint]
                            if generation != self.generation:
                                raise RequestError("Session cancelled by Stop")
                            bus.write("Torque_Enable", motor_id, 1)
                            bus.write("Lock", motor_id, 1)
            except Exception:
                try:
                    with bus.lock:
                        if getattr(bus, "opened", True):
                            bus.disable()
                finally:
                    bus.close()
                    if leader:
                        leader.close()
                    if self.pending_bus is bus:
                        self.pending_bus = None
                raise
            with self.stop_lock, self.lifecycle_lock:
                if generation != self.generation:
                    bus.close()
                    if leader:
                        leader.close()
                    raise RequestError("Session cancelled by Stop")
                self.pending_bus = None
                self.bus, self.leader, self.calibration = bus, leader, calibration
                self.generation += 1
                control_generation[0] = self.generation
                self.mode, self.port, self.robot_id, self.role = mode, port, robot_id, "follower"
                positions = {j: normalized(j, value, calibration) for j, value in raw.items()}
                self.positions, self.action = positions, dict(positions)
                self.commanded_raw = dict(raw)
                self.middle_tolerance = tolerance
                self.middle_motion = (
                    MiddleMotion(raw, calibration, tolerance) if to_middle else None
                )
                self.rest_targets = (
                    self.rest_positions.load(robot_id, calibration) if joints == JOINTS else None
                )
                self.rest_motion = None
                self.rest_phase = "idle"
                self.controller = control
                self.torque = mode != "handguide"
                self.error = None
                self.heartbeat = time.monotonic()
                self.diagnostics.emit("control_started", context=self._diagnostic_context())
        return self.status()

    def control_heartbeat(self, payload):
        if not isinstance(payload, dict) or set(payload) - {"expected_generation"}:
            raise RequestError("Invalid manual heartbeat")
        with self.lifecycle_lock:
            if self.mode == "agent":
                raise RequestError("Manual heartbeat cannot renew AI controller authority")
            if "expected_generation" in payload and (
                type(payload["expected_generation"]) is not int
                or payload["expected_generation"] != self.generation
            ):
                raise RequestError("Manual controller generation changed")
            self.heartbeat = time.monotonic()
        return self.status()

    def rest_position(self, payload, *, save=False):
        require(payload, "confirm", "SAVE REST POSITION" if save else "RETURN TO REST")
        with self.lock:
            if self.mode not in {"manual", "leader", "handguide"} or set(self.calibration) != set(
                JOINTS
            ):
                raise RequestError("Rest positions require an active session with all six joints")
            if self.middle_motion is not None or self.rest_motion is not None:
                raise RequestError("Wait for the current arm movement to finish")
            if self.writer and self.writer.episode_active:
                raise RequestError(
                    "Save or discard the recording episode before using rest positions"
                )
            if time.monotonic() - self.heartbeat > 2:
                raise RequestError("Control heartbeat expired; stop and reconnect")
            if not save and (self.mode == "handguide" or not self.torque):
                raise RequestError(
                    "Start joint control or leader teleoperation before returning to rest"
                )
            bus, generation = self.bus, self.generation
            if bus is None:
                raise RequestError("Session has stopped")
            with bus.lock:
                if generation != self.generation:
                    raise RequestError("Session has stopped")
                raw = self._read_positions(bus, self.calibration)
                with self.lifecycle_lock:
                    if generation != self.generation:
                        raise RequestError("Session has stopped")
                    if save:
                        self.rest_targets = self.rest_positions.save(
                            self.robot_id, raw, self.calibration
                        )
                        self.rest_phase = "idle"
                    else:
                        targets = self.rest_positions.load(self.robot_id, self.calibration)
                        if targets is None:
                            raise RequestError(
                                "Save a rest position for this arm and calibration first"
                            )
                        if self.controller is None:
                            raise HardwareError("Position controller is unavailable")
                        # Cancel previous goals and leader tracking without releasing torque.
                        sent = self.controller.command_raw(raw)
                        self.commanded_raw = dict(sent)
                        self.action = {
                            j: normalized(j, v, self.calibration) for j, v in sent.items()
                        }
                        self.mode = "manual"
                        self.rest_targets = targets
                        self.rest_motion = RestMotion(raw, targets, self.calibration)
                        self.rest_phase = "returning"
                        self.error = None
        return self.status()

    def jog(self, payload):
        with self.lock:
            if self.mode != "manual" or not self.torque:
                raise RequestError("Start manual control before jogging a joint")
            if self.rest_motion is not None:
                raise RequestError("Wait for the return to rest to finish")
            if self.middle_motion is not None:
                raise RequestError("Wait for movement to the recorded middle to finish")
            joint = payload.get("joint")
            absolute = "target" in payload
            if absolute == ("delta" in payload):
                raise RequestError("Choose either a slider target or a jog delta")
            value = payload.get("target" if absolute else "delta")
            if (
                not isinstance(joint, str)
                or joint not in self.calibration
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise RequestError("Choose a valid joint and a finite position")
            if absolute:
                limits = control_limits(self.calibration)[joint]
                if not limits["min"] <= value <= limits["max"]:
                    raise RequestError("Slider target is outside the calibrated joint limits")
            elif not 0 < abs(value) <= 10:
                raise RequestError(
                    "Step one valid joint by at most 10 degrees (gripper 10 percent)"
                )
            elif time.monotonic() - self.last_jog < 0.1:
                raise RequestError("Wait at least 100 ms between jog commands")
            if time.monotonic() - self.heartbeat > 2:
                raise RequestError("Control heartbeat expired; stop and reconnect")
            bus, generation = self.bus, self.generation
            if bus is None:
                raise RequestError("Session has stopped")
            with bus.lock:
                if generation != self.generation:
                    raise RequestError("Session has stopped")
                self._read_positions(bus, self.calibration)
                if generation != self.generation:
                    raise RequestError("Session has stopped")
                target = (
                    denormalized(joint, value, self.calibration)
                    if absolute
                    else step_target(joint, value, self.commanded_raw, self.calibration)
                )
                if self.controller is None:
                    raise HardwareError("LeRobot position controller is unavailable")
                sent = self.controller.command_raw({joint: target})
                if generation != self.generation:
                    raise RequestError("Session has stopped")
                self.commanded_raw.update(sent)
                self.action[joint] = normalized(joint, sent[joint], self.calibration)
                self.rest_phase = "idle"
                self.last_jog = time.monotonic()
        return self.status()

    def stop(self, reason=None, *, expected_generation=None, expected_mode=None):
        with self.stop_lock:
            self.stopping = True
            try:
                return self._stop(
                    reason, expected_generation=expected_generation, expected_mode=expected_mode
                )
            finally:
                self.stopping = False

    def _stop(self, reason=None, *, expected_generation=None, expected_mode=None):
        # Invalidate in-flight capture before waiting on any camera/serialization.
        with self.lifecycle_lock:
            if expected_generation is not None and (
                type(expected_generation) is not int or expected_generation != self.generation
            ):
                raise RequestError("Controller generation changed; stale cleanup was rejected")
            if expected_mode is not None and self.mode != expected_mode:
                raise RequestError("Controller mode changed; stale manual cleanup was rejected")
            stop_context = self._diagnostic_context()
            self.generation += 1
            self.agent.revoke(reason or "Stopped by operator")
            verification = self.verification_session
            self.verification_session = None
            if verification is not None:
                verification.stop_event.set()
            assembled = self.assembled_session
            self.assembled_session = None
            if assembled is not None:
                assembled.stop_event.set()
            bus, leader = self.bus or self.pending_bus, self.leader
            self.bus = self.leader = self.pending_bus = None
            self.torque = False
            self.mode = "idle"
            self.middle_motion = None
            self.rest_motion = None
            self.rest_phase = "idle"
            self.controller = None
            if reason:
                self.error = reason
            if self.writer:
                self.writer.episode_active = False
        errors = []
        if verification is not None:
            try:
                verification.cancel()
            except Exception as caught:
                errors.append(str(caught))
            finally:
                with self.lifecycle_lock:
                    self.verification_state = verification.snapshot()
                    self.torque = bool(self.verification_state["motion_active"])
        if assembled is not None:
            try:
                assembled.cancel()
            except Exception as caught:
                errors.append(str(caught))
            finally:
                with self.lifecycle_lock:
                    self.assembled_state = assembled.snapshot()
        for device in (bus, leader):
            if device is None:
                continue
            try:
                with device.lock:
                    if getattr(device, "opened", True):
                        if device is bus:
                            try:
                                self._checkpoint_calibration(force=True)
                            except Exception as caught:
                                errors.append(f"Calibration checkpoint failed: {caught}")
                        device.disable()
                        if device is bus and self.original_calibration is not None:
                            device.write_calibration(self.original_calibration)
            except Exception as caught:
                errors.append(str(caught))
            finally:
                device.close()
        self.original_calibration = None
        try:
            self.agent.flush_revocations()
        except Exception as caught:
            errors.append("Agent evidence flush failed: " + str(caught))
        if errors:
            self.error = (
                ((reason + ". ") if reason else "")
                + "Stop attempted but motor acknowledgement failed. Cut motor power if needed. "
                + "; ".join(errors)
            )
        elif reason is None and self.error in (
            "Robot Lab page was hidden; motor torque disabled",
            "Robot Lab page was closed; motor torque disabled",
        ):
            # An operator Stop acknowledges an earlier page-lifecycle stop.
            # Keep real hardware/cleanup faults visible and in diagnostics.
            self.error = None
        if self.calibration_state["phase"] in ("center", "ranges"):
            self.calibration_state.update(
                recoverable=self.calibration_draft.path.exists(),
                robot_id=self.robot_id,
                role=self.role,
                port=self.port,
            )
            if reason or errors:
                self.calibration_state.update(
                    interrupted_stage=self.calibration_state["phase"],
                    phase="interrupted",
                    error=self.error,
                )
            else:
                self.calibration_state["phase"] = "idle"
        self.diagnostics.emit(
            "session_stopped",
            reason=reason or "Stopped by operator",
            context=stop_context,
            cleanup_errors=list(errors),
            torque_disable_attempted=bus is not None or leader is not None,
            torque_disable_acknowledged=not errors
            if bus is not None or leader is not None
            else None,
            retain_as_last_stop=stop_context["mode"] != "idle"
            or bool(reason and reason != "Companion shutting down")
            or bool(errors),
        )
        return self.status()

    def recording_start(self, payload):
        with self.lock:
            if self.mode not in ("manual", "leader", "handguide"):
                raise RequestError("Connect a calibrated arm in Teleoperate before recording")
            if set(self.calibration) != set(JOINTS):
                raise RequestError("Dataset recording requires all six joints")
            if self.rest_motion is not None:
                raise RequestError("Wait for the return to rest before recording")
            if self.middle_motion is not None:
                raise RequestError("Wait for movement to the recorded middle before recording")
            if self.writer and not self.writer.finalized:
                raise RequestError("Finalize the previous dataset before starting another")
            fps = integer(payload.get("fps", 10), 1, 20, "FPS")
            task = payload.get("task")
            if not isinstance(task, str) or not task.strip() or len(task) > 1000:
                raise RequestError("Describe the task in 1–1000 characters")
            camera_index = payload.get("camera_index", 0)
            image_shape = None
            if camera_index is not None:
                integer(camera_index, 0, MAX_CAMERA_INDEX, "Camera index")
                image_shape = tuple(self.camera.read(camera_index).shape)
                if hasattr(self.camera, "start_reader"):
                    self.camera.start_reader(camera_index)
            if shutil.disk_usage(self.datasets_dir).free < 1024**3:
                raise RequestError(
                    "Less than 1 GiB disk space remains; free space before recording"
                )
            self.writer = DatasetWriter(
                self.datasets_dir, payload.get("dataset_id"), task.strip(), fps, image_shape
            )
            self.camera_index = camera_index
            self.recording_error = None
        return self.status()

    def recording_action(self, action, payload=None):
        payload = {} if payload is None else payload
        with self.lock:
            writer = self.writer
            if writer is None or writer.finalized:
                raise RequestError("Create a dataset first")
            if action == "start":
                if set(self.calibration) != set(JOINTS):
                    raise RequestError("Dataset recording requires all six joints")
                if self.rest_motion is not None:
                    raise RequestError("Wait for the return to rest before recording")
                if self.middle_motion is not None:
                    raise RequestError("Wait for movement to the recorded middle before recording")
                if self.mode not in ("manual", "leader", "handguide"):
                    raise RequestError("Reconnect the calibrated arm before recording")
                writer.start_episode()
                self.next_record = time.monotonic()
                self.recording_error = None
            elif action == "save":
                writer.save_episode()
            elif action == "discard":
                writer.discard_episode()
            elif action == "finish":
                abandon_empty = payload.get("abandon_empty", False)
                if type(abandon_empty) is not bool:
                    raise RequestError("abandon_empty must be true or false")
                if abandon_empty:
                    require(payload, "confirm", "ABANDON EMPTY RECORDING")
                    writer.abandon_empty()
                    # The recording loop rechecks writer identity under this
                    # lock before publishing a frame, so an in-flight capture
                    # cannot append to this retired writer after detachment.
                    self.writer = None
                    self.camera_index = None
                    self.next_record = 0.0
                    self.recording_error = None
                else:
                    writer.finalize()
            else:
                raise RequestError("Unknown recording action")
        return self.status()

    def _tick(self):
        bus, leader, mode, generation = self.bus, self.leader, self.mode, self.generation
        if bus is None or mode == "idle":
            return
        if mode == "agent":
            # Agent admission and tick use lab->bus lock order. Stop invalidates
            # generation without waiting for either lock, before torque-off I/O.
            with self.lock, bus.lock:
                if generation != self.generation:
                    return
                raw = self._read_positions(bus, self.calibration)
                if generation != self.generation or self.mode != "agent":
                    return
                self.positions = {
                    j: normalized(j, value, self.calibration) for j, value in raw.items()
                }
                completed = self.agent.tick(raw)
            if completed is not None:
                # Disk acknowledgement cannot delay the independent Stop path's
                # serial torque-off command.
                self.agent._save(completed)
            return
        with bus.lock:
            if generation != self.generation:
                return
            mode = self.mode
            raw = self._read_positions(bus, None if mode == "calibration" else self.calibration)
            if mode == "calibration":
                if self.calibration_state["phase"] == "ranges" and any(
                    joint != "wrist_roll"
                    and joint in self.positions
                    and abs(value - self.positions[joint]) > 2000
                    for joint, value in raw.items()
                ):
                    raise HardwareError(
                        "Encoder wrap or implausible jump during calibration; restart from a "
                        "supported middle pose"
                    )
                self.positions = raw
                if self.calibration_state["phase"] == "ranges":
                    for joint, position in raw.items():
                        self.calibration_state["mins"][joint] = min(
                            self.calibration_state["mins"][joint], position
                        )
                        self.calibration_state["maxes"][joint] = max(
                            self.calibration_state["maxes"][joint], position
                        )
                    self._checkpoint_calibration()
                return
            positions = {
                joint: normalized(joint, value, self.calibration) for joint, value in raw.items()
            }
            self.positions = positions
            control = self.controller
            if generation != self.generation:
                return
            if mode == "manual":
                if control is None:
                    raise HardwareError("LeRobot position controller is unavailable")
                rest = self.rest_motion
                if rest is not None:
                    goals, arrived, expired = rest.step(raw)
                    if expired and not arrived:
                        goals = raw
                    sent = control.command_raw(goals)
                    if generation != self.generation:
                        return
                    self.commanded_raw.update(sent)
                    self.action.update(
                        {j: normalized(j, v, self.calibration) for j, v in sent.items()}
                    )
                    if arrived or expired:
                        with self.lifecycle_lock:
                            if generation == self.generation:
                                self.rest_motion = None
                                self.rest_phase = "holding" if arrived else "error"
                                if not arrived:
                                    self.error = (
                                        "Return to rest timed out. Holding current position; "
                                        "check the arm before retrying."
                                    )
                    return
                motion = self.middle_motion
                if motion is not None:
                    if motion.arrived(raw):
                        with self.lifecycle_lock:
                            if generation == self.generation:
                                self.middle_motion = None
                    elif not motion.sent:
                        sent = control.command_raw(motion.targets)
                        if generation != self.generation:
                            return
                        self.commanded_raw.update(sent)
                        self.action.update(
                            {j: normalized(j, value, self.calibration) for j, value in sent.items()}
                        )
                        motion.sent = True
                if generation != self.generation:
                    return
                # LeRobot teleoperation sends complete position goals repeatedly.
                # A joint may legitimately lag a full target while lifting; never
                # reinterpret that error as breakage or reduce its requested goal.
                control.repeat()
            if mode == "leader" and leader is not None:
                if control is None:
                    raise HardwareError("LeRobot position controller is unavailable")
                with leader.lock:
                    leader_raw = self._read_positions(
                        leader, self.leader_calibration, source="leader"
                    )
                desired = {
                    joint: normalized(joint, leader_raw[joint], self.leader_calibration)
                    for joint in JOINTS
                }
                if generation != self.generation:
                    return
                sent = control.command(desired)
                if generation != self.generation:
                    return
                self.commanded_raw.update(sent)
                self.action = {
                    j: normalized(j, value, self.calibration) for j, value in sent.items()
                }
            elif mode == "handguide":
                self.action = dict(positions)
        writer = self.writer
        if writer and writer.episode_active and time.monotonic() >= self.next_record:
            try:
                image = None
                if self.camera_index is not None:
                    read_image = getattr(self.camera, "read_latest", self.camera.read)
                    image = read_image(self.camera_index)[:, :, ::-1].copy()
                with self.lock:
                    if (
                        generation == self.generation
                        and writer is self.writer
                        and writer.episode_active
                    ):
                        captured_at = time.monotonic()
                        writer.add_frame(
                            dict(self.positions), dict(self.action), image, captured_at
                        )
                        self.next_record += 1.0 / writer.fps
            except Exception as error:
                # A recorder failure pauses collection, without conflating it
                # with motor feedback or bypassing the independent Stop/watchdog.
                with self.lifecycle_lock:
                    if generation != self.generation or writer is not self.writer:
                        return
                    writer.episode_active = False
                    self.recording_error = "Recording paused: " + str(error)
                self.diagnostics.emit(
                    "recording_paused",
                    reason=self.recording_error,
                    context=self._diagnostic_context(),
                )

    def _loop(self):
        while not self.finished.is_set():
            cycle_started = time.monotonic()
            generation, mode = self.generation, self.mode
            try:
                self._tick()
                self.camera.close_idle()
            except Exception as caught:
                with self.lifecycle_lock:
                    if generation != self.generation or mode != self.mode:
                        continue
                    if self.writer and self.writer.episode_active:
                        self.writer.episode_active = False
                        self.recording_error = str(caught)
                try:
                    self.stop(
                        "Hardware session stopped: " + str(caught),
                        expected_generation=generation,
                        expected_mode=mode,
                    )
                except RequestError:
                    pass  # An intervening takeover owns its independent lifecycle.
            # LeRobot CycleTimer pacing: account for work already done, without
            # bursting missed commands after a slow read or camera operation.
            if self.finished.wait(max(0.0, 0.025 - (time.monotonic() - cycle_started))):
                break

    def _watchdog(self):
        while not self.finished.wait(0.1):
            # A delayed watchdog result cannot stop a later controller, including
            # the manual-to-agent transfer that preserves the serial generation.
            generation, mode = self.generation, self.mode
            reason = None
            if mode == "agent":
                if self.agent.expired():
                    reason = "AI operator or execution lease expired; torque disabled"
            elif mode not in {"idle", "calibration"} and time.monotonic() - self.heartbeat > (
                10 if mode in {"setup", "verification"} else 2
            ):
                reason = (
                    "Setup heartbeat timed out; ID work stopped"
                    if mode == "setup"
                    else "Browser heartbeat timed out; motor torque disabled"
                )
            if reason:
                try:
                    self.stop(reason, expected_generation=generation, expected_mode=mode)
                except RequestError:
                    pass

    def close(self):
        self.finished.set()
        self.stop("Companion shutting down")
        self.camera.close()
        if self.writer and hasattr(self.writer, "close"):
            try:
                self.writer.close()
            except Exception as error:
                self.diagnostics.emit("recording_write_error", reason=str(error))
        for thread in self.threads:
            thread.join(timeout=1)
        self.diagnostics.close()


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class Handler(BaseHTTPRequestHandler):
    server_version = "OpenJensenRobotLab/1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        # No headers, payloads or private token ever logged.
        pass

    def _send(self, code, value, kind="application/json"):
        body = json.dumps(value, allow_nan=False).encode() if kind == "application/json" else value
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _authenticate(self):
        provided = self.headers.get("Authorization", "")
        if not secrets.compare_digest(provided, "Bearer " + self.server.token):
            self._send(
                401, {"error": "Private companion authentication required", "code": "unauthorized"}
            )
            return False
        # Server-to-server only. A browser cannot directly command the hardware.
        if self.headers.get("Origin") or self.headers.get("Sec-Fetch-Site"):
            self._send(403, {"error": "Use the OpenJensen backend proxy", "code": "forbidden"})
            return False
        return True

    def do_GET(self):
        if not self._authenticate():
            return
        try:
            parsed = urlsplit(self.path)
            path, query, lab = unquote(parsed.path), parse_qs(parsed.query), self.server.lab
            if path.startswith("/v1/agent/executions/"):
                return self._send(200, lab.agent.receipt(path.rsplit("/", 1)[-1]))
            if path == "/v1/diagnostics":
                return self._send(200, lab.diagnostics.snapshot(full=True))
            if path == "/v1/status":
                return self._send(200, lab.status())
            if path == "/v1/calibration/export":
                if set(query) != {"robot_id", "role"} or any(len(v) != 1 for v in query.values()):
                    raise RequestError("Choose one calibration profile and arm role")
                return self._send(
                    200, lab.calibration_export(query["robot_id"][0], query["role"][0])
                )
            if path == "/v1/devices":
                return self._send(200, lab.devices())
            if path == "/v1/camera.jpg":
                index = integer(
                    int(query.get("index", ["0"])[0]), 0, MAX_CAMERA_INDEX, "Camera index"
                )
                image = (
                    lab.camera.cached_jpeg(index)
                    if lab.writer and lab.writer.episode_active and lab.camera_index == index
                    else lab.camera.jpeg(index)
                )
                return self._send(200, image, "image/jpeg")
            if path == "/v1/datasets":
                return self._send(200, {"datasets": list_datasets(lab.datasets_dir)})
            parts = path.split("/")
            if len(parts) >= 5 and parts[1:3] == ["v1", "datasets"]:
                dataset_id, action = parts[3:5]
                if (
                    action == "manifest"
                    or action == "files"
                    and len(parts) == 5
                    and "path" not in query
                ):
                    return self._send(200, manifest(lab.datasets_dir, dataset_id))
                if action == "files":
                    relative = "/".join(parts[5:]) if len(parts) > 5 else query.get("path", [""])[0]
                    target = dataset_file(lab.datasets_dir, dataset_id, relative)
                    self.send_response(200)
                    self.send_header(
                        "Content-Type",
                        mimetypes.guess_type(target.name)[0] or "application/octet-stream",
                    )
                    self.send_header("Content-Length", str(target.stat().st_size))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    with target.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            self.wfile.write(chunk)
                    self.close_connection = True
                    return
            return self._send(404, {"error": "Unknown companion route", "code": "not_found"})
        except BrokenPipeError, ConnectionResetError:
            pass
        except Exception as error:
            self._error(error)

    def do_POST(self):
        if not self._authenticate():
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= length <= 16_384:
                raise RequestError("Request body too large")
            raw = self.rfile.read(length) or b"{}"
            payload = (
                agent_json(raw)
                if urlsplit(self.path).path.startswith("/v1/agent/")
                else json.loads(raw)
            )
            if not isinstance(payload, dict):
                raise RequestError("Expected a JSON object")
            lab, path = self.server.lab, urlsplit(self.path).path
            if path.startswith("/v1/agent/") and path.rsplit("/", 1)[-1] in {
                "acquire",
                "observe",
                "execute",
                "heartbeat",
                "release",
            }:
                result = lab.agent.command(path.rsplit("/", 1)[-1], payload)
            elif path == "/v1/probe":
                result = lab.probe(payload)
            elif path == "/v1/setup/motor":
                result = lab.setup_motor(payload)
            elif path == "/v1/mapping/confirm":
                result = lab.mapping_confirm(payload)
            elif path == "/v1/mapping/wiggle":
                result = lab.mapping_wiggle(payload)
            elif path == "/v1/setup/assembled/start":
                result = lab.assembled_start(payload)
            elif path in (
                "/v1/setup/assembled/observe",
                "/v1/setup/assembled/wiggle",
                "/v1/setup/assembled/accept",
                "/v1/setup/assembled/finish",
            ):
                result = lab.assembled_action(path.rsplit("/", 1)[-1], payload)
            elif path == "/v1/setup/assembled/cancel":
                result = lab.stop()
            elif path == "/v1/devices/configure":
                result = lab.configure_device(payload)
            elif path == "/v1/calibration/import":
                result = lab.calibration_import(payload)
            elif path == "/v1/calibration/start":
                result = lab.calibration_start(payload)
            elif path == "/v1/calibration/center":
                result = lab.calibration_center(payload)
            elif path == "/v1/calibration/finish":
                result = lab.calibration_finish(payload)
            elif path == "/v1/calibration/cancel":
                result = lab.calibration_cancel()
            elif path in ("/v1/control/stop", "/v1/stop"):
                if (
                    set(payload) - {"expected_generation", "expected_mode", "cause"}
                    or "expected_generation" in payload
                    and type(payload["expected_generation"]) is not int
                    or "expected_mode" in payload
                    and payload["expected_mode"] != "manual"
                    or "cause" in payload
                    and payload["cause"] not in ("page_hidden", "page_closed")
                ):
                    raise RequestError("Invalid Stop request")
                result = lab.stop(
                    {
                        "page_hidden": "Robot Lab page was hidden; motor torque disabled",
                        "page_closed": "Robot Lab page was closed; motor torque disabled",
                    }.get(payload.get("cause")),
                    expected_generation=payload.get("expected_generation"),
                    expected_mode=payload.get("expected_mode"),
                )
            elif path == "/v1/control/start":
                result = lab.control_start(payload)
            elif path == "/v1/control/rest/save":
                result = lab.rest_position(payload, save=True)
            elif path == "/v1/control/rest/return":
                result = lab.rest_position(payload)
            elif path == "/v1/control/jog":
                result = lab.jog(payload)
            elif path == "/v1/control/heartbeat":
                result = lab.control_heartbeat(payload)
            elif path == "/v1/recording/start":
                result = lab.recording_start(payload)
            elif path.startswith("/v1/recording/episode/") and path.rsplit("/", 1)[-1] in (
                "start",
                "save",
                "discard",
            ):
                result = lab.recording_action(path.rsplit("/", 1)[-1])
            elif path == "/v1/recording/finish":
                result = lab.recording_action("finish", payload)
            else:
                return self._send(404, {"error": "Unknown companion route", "code": "not_found"})
            self._send(200, result)
        except BrokenPipeError, ConnectionResetError:
            pass
        except Exception as error:
            self._error(error)

    def _error(self, error):
        if isinstance(error, HardwareError):
            self.server.lab.diagnostics.emit(
                "hardware_request_failed", exception=type(error).__name__, message=str(error)
            )
            self.server.lab.stop("Motor safety check failed: " + str(error))
        if isinstance(error, (ValueError, KeyError, TypeError)):
            self._send(422, {"error": str(error), "code": "invalid_request"})
        elif isinstance(error, (ImportError, ModuleNotFoundError)):
            self._send(
                503,
                {
                    "error": f"Companion dependency unavailable: {error}",
                    "code": "missing_dependency",
                },
            )
        elif isinstance(error, FileNotFoundError):
            self._send(404, {"error": "Requested local file not found", "code": "not_found"})
        else:
            self._send(409, {"error": str(error), "code": "hardware_error"})


def create_server(lab, token, host="127.0.0.1", port=18101):
    if host != "127.0.0.1":
        raise ValueError("Companion must bind to IPv4 loopback only")
    if not isinstance(token, str) or len(token) < 32:
        raise ValueError("Private companion token must contain at least 32 characters")
    server = Server((host, port), Handler)
    server.lab, server.token = lab, token
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18101)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=(
            Path.home() / "Library/Application Support/OpenJensen/robot-lab"
            if platform.system() == "Darwin"
            else Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
            / "OpenJensen/robot-lab"
        ),
    )
    args = parser.parse_args()
    token_path = args.token_file
    if token_path.is_symlink() or stat.S_IMODE(token_path.stat().st_mode) & 0o077:
        parser.error("Token must be a regular private file with permissions 0600")
    token = token_path.read_text().strip()
    dispatcher = MainThreadDispatcher()
    lab = RobotLab(args.data_dir, camera=CameraPool(dispatcher=dispatcher))
    server = create_server(lab, token, args.host, args.port)
    stopping = threading.Event()

    def stop_signal(signum, frame):
        stopping.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop_signal)
    signal.signal(signal.SIGTERM, stop_signal)
    serve_thread = threading.Thread(
        target=lambda: server.serve_forever(poll_interval=0.1), daemon=True
    )
    serve_thread.start()
    try:
        while not stopping.is_set():
            dispatcher.pump_once(timeout=0.1)
    finally:
        dispatcher.close()
        lab.close()
        server.shutdown()
        server.server_close()
        serve_thread.join(timeout=1)


if __name__ == "__main__":
    main()
