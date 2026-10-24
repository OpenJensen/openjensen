#!/usr/bin/env python3
"""Private, loopback-only OpenJensen SO-101 hardware companion."""

from __future__ import annotations

import argparse
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

from assembled_session import AssembledSession, CalibrationRevisions, empty_assembled_state
from camera import CameraPool
from dataset import DatasetWriter, atomic_json, dataset_file, list_datasets, manifest, safe_slug
from dispatch import MainThreadDispatcher
from hardware import (
    BAUDRATES,
    IDS,
    ISOLATION_WARNING,
    JOINTS,
    HardwareError,
    ServoBus,
    denormalized,
    normalized,
    probe_bus,
)
from verification import (
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
            entry.get("id") != IDS[joint]
            or entry.get("drive_mode") != 0
            or not all(
                isinstance(entry.get(key), int) and not isinstance(entry.get(key), bool)
                for key in ("homing_offset", "range_min", "range_max")
            )
        ):
            raise RequestError("Calibration has invalid joint data")
        if (
            not 0 <= entry["range_min"] < entry["range_max"] <= 4095
            or not -2047 <= entry["homing_offset"] <= 2047
        ):
            raise RequestError("Calibration range/offset is invalid")
    return value


def validate_raw(raw, calibration=None):
    if set(raw) != set(JOINTS) or any(
        isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 4095
        for value in raw.values()
    ):
        raise HardwareError("Invalid encoder reading; inspect cabling and motor setup")
    if calibration is not None and any(
        not calibration[joint]["range_min"] <= value <= calibration[joint]["range_max"]
        for joint, value in raw.items()
    ):
        raise HardwareError(
            "Joint reading outside its calibrated range; stop and recalibrate before motion"
        )
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
        self.calibration_revisions = CalibrationRevisions(self.data_dir)
        self.assembled_session = None
        self.assembled_state = empty_assembled_state()
        saved = latest_saved_mapping(self.data_dir)
        if saved:
            self.assembled_state.update(saved, phase="complete", message="Saved motor IDs 1–6")
        self.verification_factory = verification_factory
        self.verification_session = None
        self.verification_state = empty_verification()
        self.bus_factory = bus_factory
        self.port_provider = port_provider
        self.camera = camera or CameraPool()
        self.bus = None
        self.leader = None
        self.pending_bus = None
        self.generation = 0
        self.mode = "idle"
        self.port = self.robot_id = self.role = None
        self.torque = False
        self.error = None
        self.positions = {}
        self.action = {}
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
        self.heartbeat = time.monotonic()
        self.last_jog = 0.0
        self.writer = None
        self.recording_error = None
        self.camera_index = None
        self.next_record = 0.0
        self.finished = threading.Event()
        self.threads = []
        if start_threads:
            for target in (self._loop, self._watchdog):
                thread = threading.Thread(target=target, daemon=True)
                thread.start()
                self.threads.append(thread)

    def calibration_path(self, robot_id, role):
        safe_slug(robot_id)
        if role not in ("follower", "leader"):
            raise RequestError("Role must be follower or leader")
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
                calibrations.append({"robot_id": path.stem, "role": role, "path": str(path)})
        writer = self.writer
        with self.lifecycle_lock:
            assembled = json.loads(json.dumps(self.assembled_state))
            copy_verification = json.loads(json.dumps(self.verification_state))
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
            "dependencies": deps,
            "capabilities": {
                "probe": hardware,
                "camera": deps["cv2"]["installed"],
                "motor_setup": hardware,
                "assembled_setup": hardware,
                "mapping_verification": hardware,
                "calibration": hardware,
                "control": hardware,
                "recording": hardware
                and all(deps[key]["installed"] for key in ("pyarrow", "PIL", "numpy")),
            },
            "session": {
                "mode": self.mode,
                "robot_id": self.robot_id,
                "port": self.port,
                "role": self.role,
                "positions": dict(self.positions),
                "torque_enabled": self.torque,
                "watchdog_seconds": 10.0
                if self.mode in {"calibration", "setup", "verification"}
                else 2.0,
                "error": self.error,
            },
            "calibration": json.loads(json.dumps(self.calibration_state)),
            "assembled": assembled,
            "verification": copy_verification,
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
                    "label": joint.replace("_", " ").title(),
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
        return {"ports": self.port_provider(), "cameras": cameras, "camera_error": error}

    def probe(self, payload):
        with self.lock:
            self._idle()
            port = self._port(payload.get("port"))
            baud = payload.get("baudrate", 1_000_000)
            if baud not in BAUDRATES:
                raise RequestError("Unsupported STS3215 baudrate")
            bus = self.bus_factory(port, baud)
            try:
                with bus.lock:
                    return {"port": port, "baudrate": baud, **probe_bus(bus)}
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
                self.assembled_state.update(saved, phase="complete")
                self.generation += 1
                self.mode, self.port = "verification", port
                self.robot_id = self.role = None
                self.torque, self.error = False, None
                self.heartbeat = time.monotonic()
                controller.start()
        return self.status()

    def calibration_start(self, payload):
        require(payload, "confirm", "START CALIBRATION")
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
                self.calibration_state = {
                    "phase": "center",
                    "mins": {},
                    "maxes": {},
                    "offsets": {},
                    "path": str(path),
                }
                self.heartbeat, self.error = time.monotonic(), None
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
                for motor_id in IDS.values():
                    bus.write("Homing_Offset", motor_id, 0)
                    bus.write("Min_Position_Limit", motor_id, 0)
                    bus.write("Max_Position_Limit", motor_id, 4095)
                raw = validate_raw(bus.positions())
                offsets = {joint: position - 2047 for joint, position in raw.items()}
                if any(abs(value) > 2047 for value in offsets.values()):
                    raise RequestError(
                        "Invalid middle pose encoder reading; cancel and inspect the motor setup"
                    )
                for joint, offset in offsets.items():
                    bus.write("Homing_Offset", IDS[joint], offset)
                raw = validate_raw(bus.positions())
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
        self.stop()
        return self.status()

    def control_start(self, payload):
        require(payload, "confirm", "ENABLE MOTION")
        mode = payload.get("mode")
        if mode not in ("manual", "leader", "handguide"):
            raise RequestError("Choose manual, leader or handguide mode")
        with self.lock:
            self._idle()
            port = self._port(payload.get("port"))
            robot_id = safe_slug(payload.get("robot_id"))
            calibration = self.load_calibration(robot_id, "follower", port)
            generation = self.generation
            leader, bus = None, self.bus_factory(port)
            with self.stop_lock, self.lifecycle_lock:
                if generation != self.generation:
                    bus.close()
                    raise RequestError("Session cancelled by Stop")
                self.pending_bus = bus
            try:
                with bus.lock:
                    bus.check_six()
                    if generation != self.generation:
                        raise RequestError("Session cancelled by Stop")
                    # Never rewrite saved EEPROM calibration when starting motion.
                    if bus.read_calibration() != calibration:
                        raise RequestError(
                            "Motor calibration differs from the saved profile; recalibrate "
                            "before motion"
                        )
                    bus.disable()
                    raw = validate_raw(bus.positions(), calibration)
                    if any(
                        not calibration[j]["range_min"] <= raw[j] <= calibration[j]["range_max"]
                        for j in JOINTS
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
                            leader.disable()
                            validate_raw(leader.positions(), leader_calibration)
                        self.leader_calibration = leader_calibration
                    if mode != "handguide":
                        for joint, motor_id in IDS.items():
                            if generation != self.generation:
                                raise RequestError("Session cancelled by Stop")
                            bus.write("Lock", motor_id, 0)
                            bus.write("Operating_Mode", motor_id, 0)
                            bus.write("Acceleration", motor_id, 10)
                            bus.write("Goal_Velocity", motor_id, 100)
                            bus.write("P_Coefficient", motor_id, 16)
                            bus.write("I_Coefficient", motor_id, 0)
                            bus.write("D_Coefficient", motor_id, 32)
                            if joint == "gripper":
                                bus.write("Max_Torque_Limit", motor_id, 500)
                                bus.write("Protection_Current", motor_id, 250)
                                bus.write("Overload_Torque", motor_id, 25)
                            bus.write("Goal_Position", motor_id, raw[joint])
                        for motor_id in IDS.values():
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
                self.mode, self.port, self.robot_id, self.role = mode, port, robot_id, "follower"
                self.positions, self.action = positions, dict(positions)
                self.torque = mode != "handguide"
                self.error = None
                self.heartbeat = time.monotonic()
        return self.status()

    def jog(self, payload):
        with self.lock:
            if self.mode != "manual" or not self.torque:
                raise RequestError("Start manual control before jogging a joint")
            joint, delta = payload.get("joint"), payload.get("delta")
            if (
                joint not in JOINTS
                or isinstance(delta, bool)
                or not isinstance(delta, (int, float))
                or not math.isfinite(delta)
                or not 0 < abs(delta) <= 2
            ):
                raise RequestError("Jog one valid joint by at most 2 degrees (gripper 2 percent)")
            if time.monotonic() - self.last_jog < 0.1:
                raise RequestError("Wait at least 100 ms between jog commands")
            if time.monotonic() - self.heartbeat > 2:
                raise RequestError("Control heartbeat expired; stop and reconnect")
            bus, generation = self.bus, self.generation
            if bus is None:
                raise RequestError("Session has stopped")
            with bus.lock:
                if generation != self.generation:
                    raise RequestError("Session has stopped")
                raw = validate_raw(bus.positions(), self.calibration)
                current = normalized(joint, raw[joint], self.calibration)
                target = denormalized(joint, current + delta, self.calibration)
                # Keep a small margin away from captured mechanical limits.
                entry = self.calibration[joint]
                target = max(entry["range_min"] + 8, min(entry["range_max"] - 8, target))
                if generation != self.generation:
                    raise RequestError("Session has stopped")
                bus.write("Goal_Position", IDS[joint], target)
                self.action[joint] = normalized(joint, target, self.calibration)
            self.last_jog = time.monotonic()
        return self.status()

    def stop(self, reason=None):
        with self.stop_lock:
            self.stopping = True
            try:
                return self._stop(reason)
            finally:
                self.stopping = False

    def _stop(self, reason=None):
        # Invalidate in-flight capture before waiting on any camera/serialization.
        with self.lifecycle_lock:
            self.generation += 1
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
                        device.disable()
                        if device is bus and self.original_calibration is not None:
                            device.write_calibration(self.original_calibration)
            except Exception as caught:
                errors.append(str(caught))
            finally:
                device.close()
        self.original_calibration = None
        if errors:
            self.error = (
                ((reason + ". ") if reason else "")
                + "Stop attempted but motor acknowledgement failed. Cut motor power if needed. "
                + "; ".join(errors)
            )
        if self.calibration_state["phase"] in ("center", "ranges"):
            if reason or errors:
                self.calibration_state.update(
                    interrupted_stage=self.calibration_state["phase"],
                    phase="interrupted",
                    error=self.error,
                )
            else:
                self.calibration_state["phase"] = "idle"
        return self.status()

    def recording_start(self, payload):
        with self.lock:
            if self.mode not in ("manual", "leader", "handguide"):
                raise RequestError("Connect a calibrated arm in Teleoperate before recording")
            if self.writer and not self.writer.finalized:
                raise RequestError("Finalize the previous dataset before starting another")
            fps = integer(payload.get("fps", 10), 1, 20, "FPS")
            task = payload.get("task")
            if not isinstance(task, str) or not task.strip() or len(task) > 1000:
                raise RequestError("Describe the task in 1–1000 characters")
            camera_index = payload.get("camera_index", 0)
            image_shape = None
            if camera_index is not None:
                integer(camera_index, 0, 8, "Camera index")
                image_shape = tuple(self.camera.read(camera_index).shape)
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
        with bus.lock:
            if generation != self.generation:
                return
            raw = validate_raw(bus.positions(), None if mode == "calibration" else self.calibration)
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
                return
            positions = {
                joint: normalized(joint, value, self.calibration) for joint, value in raw.items()
            }
            self.positions = positions
            if mode == "leader" and leader is not None:
                with leader.lock:
                    leader_raw = validate_raw(leader.positions(), self.leader_calibration)
                for joint in JOINTS:
                    desired = normalized(joint, leader_raw[joint], self.leader_calibration)
                    target = positions[joint] + max(-2.0, min(2.0, desired - positions[joint]))
                    target_raw = denormalized(joint, target, self.calibration)
                    entry = self.calibration[joint]
                    target_raw = max(
                        entry["range_min"] + 8, min(entry["range_max"] - 8, target_raw)
                    )
                    if generation != self.generation:
                        return
                    bus.write("Goal_Position", IDS[joint], target_raw)
                    self.action[joint] = normalized(joint, target_raw, self.calibration)
            elif mode == "handguide":
                self.action = dict(positions)
        writer = self.writer
        if writer and writer.episode_active and time.monotonic() >= self.next_record:
            image = None
            if self.camera_index is not None:
                image = self.camera.read(self.camera_index)[:, :, ::-1].copy()  # OpenCV BGR -> RGB
            with self.lock:
                if (
                    generation == self.generation
                    and writer is self.writer
                    and writer.episode_active
                ):
                    captured_at = time.monotonic()
                    writer.add_frame(dict(self.positions), dict(self.action), image, captured_at)
                    self.next_record += 1.0 / writer.fps

    def _loop(self):
        while not self.finished.wait(0.025):
            try:
                self._tick()
                self.camera.close_idle()
            except Exception as caught:
                if self.writer and self.writer.episode_active:
                    self.writer.episode_active = False
                    self.recording_error = str(caught)
                self.stop("Hardware session stopped: " + str(caught))

    def _watchdog(self):
        while not self.finished.wait(0.1):
            if self.mode != "idle" and time.monotonic() - self.heartbeat > (
                10 if self.mode in {"calibration", "setup", "verification"} else 2
            ):
                self.stop(
                    "Setup heartbeat timed out; ID work stopped"
                    if self.mode == "setup"
                    else "Browser heartbeat timed out; motor torque disabled"
                )

    def close(self):
        self.finished.set()
        self.stop("Companion shutting down")
        self.camera.close()
        for thread in self.threads:
            thread.join(timeout=1)


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
            if path == "/v1/status":
                return self._send(200, lab.status())
            if path == "/v1/devices":
                return self._send(200, lab.devices())
            if path == "/v1/camera.jpg":
                index = integer(int(query.get("index", ["0"])[0]), 0, 8, "Camera index")
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
        except (BrokenPipeError, ConnectionResetError):
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
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise RequestError("Expected a JSON object")
            lab, path = self.server.lab, urlsplit(self.path).path
            if path == "/v1/probe":
                result = lab.probe(payload)
            elif path == "/v1/setup/motor":
                result = lab.setup_motor(payload)
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
            elif path == "/v1/calibration/start":
                result = lab.calibration_start(payload)
            elif path == "/v1/calibration/center":
                result = lab.calibration_center(payload)
            elif path == "/v1/calibration/finish":
                result = lab.calibration_finish(payload)
            elif path in ("/v1/calibration/cancel", "/v1/control/stop", "/v1/stop"):
                result = lab.stop()
            elif path == "/v1/control/start":
                result = lab.control_start(payload)
            elif path == "/v1/control/jog":
                result = lab.jog(payload)
            elif path == "/v1/control/heartbeat":
                lab.heartbeat = time.monotonic()
                result = lab.status()
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
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            self._error(error)

    def _error(self, error):
        if isinstance(error, HardwareError):
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
        default=Path.home() / "Library/Application Support/OpenJensen/robot-lab",
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
