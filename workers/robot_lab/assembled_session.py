"""Bounded UI sessions for experimental assembled-chain ID setup.

This module never opens a serial port on import or construction. Its callbacks
publish cached state and must not acquire a lock held while waiting for motor I/O.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path

from dataset import atomic_json
from hardware import JOINTS
from identify import IdentificationUnavailable


def durable_json(path, value):
    atomic_json(path, value)
    path.chmod(0o600)
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class CalibrationRevisions:
    """Invalidate old profiles before ID writes without deleting calibration files."""

    def __init__(self, data_dir):
        self.path = Path(data_dir) / "motor-setup-revisions.json"
        self.lock = threading.RLock()

    def _read(self):
        if self.path.is_symlink():
            raise ValueError("Motor setup revision file must not be a symbolic link")
        if not self.path.exists():
            return {"version": 1, "ports": {}}
        if self.path.is_symlink() or self.path.stat().st_size > 1024 * 1024:
            raise ValueError(
                "Motor setup revision file is unavailable; calibration cannot be trusted"
            )
        value = json.loads(self.path.read_text())
        if (
            not isinstance(value, dict)
            or value.get("version") != 1
            or not isinstance(value.get("ports"), dict)
        ):
            raise ValueError("Motor setup revision file is invalid; calibration cannot be trusted")
        for record in value["ports"].values():
            if (
                not isinstance(record, dict)
                or not isinstance(record.get("revision"), str)
                or not isinstance(record.get("calibrations"), dict)
            ):
                raise ValueError(
                    "Motor setup revision records are invalid; calibration cannot be trusted"
                )
            if any(not isinstance(entry, dict) for entry in record["calibrations"].values()):
                raise ValueError("Motor setup calibration acknowledgements are invalid")
        return value

    @staticmethod
    def _key(port):
        return hashlib.sha256(port.encode()).hexdigest()

    def mark_dirty(self, port):
        with self.lock:
            value = self._read()
            revision = uuid.uuid4().hex
            value["ports"][self._key(port)] = {
                "port": port,
                "revision": revision,
                "marked_at_ns": time.time_ns(),
                "calibrations": {},
            }
            durable_json(self.path, value)
            return revision

    def acknowledge(self, port, robot_id, role, path):
        with self.lock:
            value = self._read()
            record = value["ports"].get(self._key(port))
            if record is None:
                return
            record["calibrations"][role + "/" + robot_id] = {
                "revision": record["revision"],
                "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            }
            durable_json(self.path, value)

    def fresh(self, port, robot_id, role, path):
        with self.lock:
            record = self._read()["ports"].get(self._key(port))
            if record is None:
                return True
            acknowledgement = record.get("calibrations", {}).get(role + "/" + robot_id, {})
            return (
                acknowledgement.get("revision") == record.get("revision")
                and acknowledgement.get("sha256")
                == hashlib.sha256(Path(path).read_bytes()).hexdigest()
            )

    def required(self, port):
        if not port:
            return False
        with self.lock:
            record = self._read()["ports"].get(self._key(port))
            return record is not None and not any(
                entry.get("revision") == record.get("revision")
                for entry in record.get("calibrations", {}).values()
                if isinstance(entry, dict)
            )


def empty_assembled_state():
    return {
        "phase": "idle",
        "port": None,
        "ids": [],
        "positions": {},
        "selected_joint": None,
        "detected_id": None,
        "candidate_delta": None,
        "mapping": {},
        "message": "Start experimental setup to assign distinct IDs with the chain connected.",
        "error": None,
        "log": [],
        "progress": None,
        "calibration_required": False,
        "experimental": True,
        "motion_active": False,
        "wiggle_id": None,
        "wiggle_ticks": None,
    }


class AssembledSession:
    """One reserved port, cancellable background work and explicit map acceptance."""

    def __init__(
        self,
        port,
        data_dir,
        on_change,
        *,
        engine_factory=None,
        detector=None,
        observe_timeout=45,
        verified_so101=False,
    ):
        if engine_factory is None or detector is None:
            from assembled import AssembledSetup, detect_movement

            engine_factory = engine_factory or AssembledSetup
            detector = detector or detect_movement
        self.port = port
        self.data_dir = Path(data_dir)
        self.on_change = on_change
        self.detector = detector
        self.observe_timeout = min(120, max(0.1, observe_timeout))
        self.lock = threading.RLock()
        self.io_lock = threading.RLock()
        self.stop_event = threading.Event()
        self.observation_event = threading.Event()
        self.observation_token = 0
        self.detected_at = None
        self.wiggle_at = None
        self.workers = []
        self.operation = "split"
        self.state = {
            **empty_assembled_state(),
            "port": port,
            "phase": "scanning",
            "calibration_required": True,
        }
        self.engine = engine_factory(
            port,
            self.data_dir,
            stop_event=self.stop_event,
            io_lock=self.io_lock,
            progress=self._progress,
            verified_so101=verified_so101,
        )

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.state)

    def _publish(self, *, terminal=False, **changes):
        with self.lock:
            if self.stop_event.is_set() and changes.get("phase") != "stopped":
                if changes == {"motion_active": False}:
                    self.state["motion_active"] = False
                return
            self.state.update(changes)
            message = changes.get("message")
            if message and (not self.state["log"] or message != self.state["log"][-1]):
                self.state["log"] = [*self.state["log"], str(message)[:500]][-40:]
            snapshot = self.snapshot()
        self.on_change(self, snapshot, terminal)

    def _progress(self, snapshot):
        changes = {}
        if self.operation in {"split", "apply"} and snapshot.get("message"):
            changes["message"] = str(snapshot["message"])[:1000]
        if isinstance(snapshot.get("ids"), list):
            changes["ids"] = [
                sid for sid in snapshot["ids"] if type(sid) is int and 0 <= sid <= 253
            ][:253]
        if isinstance(snapshot.get("positions"), dict):
            changes["positions"] = dict(snapshot["positions"])
        with self.lock:
            if self.operation == "split" and snapshot.get("phase") == "splitting":
                changes["phase"] = "splitting"
            elif self.operation == "apply":
                changes["phase"] = "applying"
        if "round" in snapshot:
            changes["progress"] = snapshot["round"]
        self._publish(**changes)

    def _thread(self, action):
        worker = threading.Thread(target=action, daemon=True, name="robot-lab-assembled")
        with self.lock:
            self.workers.append(worker)
        worker.start()

    def _close(self):
        with self.io_lock:
            self.engine.close()

    def _fail(self, error):
        if self.stop_event.is_set():
            return
        try:
            self._close()
        except Exception as cleanup_error:
            error = RuntimeError(str(error) + ". Stop cleanup failed: " + str(cleanup_error))
        finally:
            with self.lock:
                self.operation = None
            self._publish(
                phase="error", error=str(error)[:1000], message=str(error)[:1000], terminal=True
            )

    def start(self):
        self._thread(self._split)

    def _split(self):
        try:
            ids = self.engine.split()
            if (
                len(ids) != 6
                or len(set(ids)) != 6
                or any(type(sid) is not int or not 0 <= sid <= 253 for sid in ids)
            ):
                raise ValueError(
                    "Exactly six independent motor IDs are required before mapping joints"
                )
            with self.lock:
                self.operation = None
            self._publish(
                phase="ready",
                ids=sorted(ids),
                message=(
                    "Six independent IDs found. Observe one highlighted joint at "
                    "a time, then confirm its detected ID."
                ),
            )
        except Exception as error:
            self._fail(error)

    def _ensure_live(self):
        if self.stop_event.is_set() or self.state["phase"] in {
            "idle",
            "scanning",
            "splitting",
            "applying",
            "wiggling",
            "complete",
            "error",
            "stopped",
        }:
            raise ValueError("Complete motor separation before observing or confirming joints")

    def observe(self, joint):
        if joint not in JOINTS:
            raise ValueError("Choose one of the six SO-101 joints")
        with self.lock:
            self._ensure_live()
            if joint in self.state["mapping"]:
                raise ValueError(
                    "This joint is already confirmed. Restart identification to change its mapping."
                )
            self.observation_event.set()
            self.observation_event = threading.Event()
            event = self.observation_event
            self.observation_token += 1
            token = self.observation_token
            self.detected_at = None
            self.wiggle_at = None
            self.operation = "observe"
        self._publish(
            phase="observing",
            error=None,
            wiggle_id=None,
            wiggle_ticks=None,
            selected_joint=joint,
            detected_id=None,
            candidate_delta=None,
            message=(
                "Move only the highlighted joint gently by hand. This "
                "observation sends no motor commands."
            ),
        )
        self._thread(lambda: self._observe(token, event))

    def _positions(self):
        positions = self.engine.positions()
        with self.lock:
            expected = set(self.state["ids"])
        if set(positions) != expected or any(
            type(value) is not int or not 0 <= value <= 4095 for value in positions.values()
        ):
            raise ValueError(
                "A motor position is missing or invalid; stop and scan the chain again"
            )
        return positions

    def _observe(self, token, event):
        try:
            baseline = self._positions()
            samples = []
            deadline = time.monotonic() + self.observe_timeout
            while (
                not self.stop_event.is_set() and not event.is_set() and time.monotonic() < deadline
            ):
                positions = self._positions()
                now = time.monotonic()
                samples = [*samples[-3:], (now, positions)]
                with self.lock:
                    if token != self.observation_token or event.is_set():
                        return
                    assigned = tuple(self.state["mapping"].values())
                candidate = self.detector(baseline, samples, assigned=assigned, now=now)
                delta = None
                if candidate is not None:
                    if candidate not in positions or candidate in assigned:
                        raise ValueError(
                            "Movement detector returned an invalid or previously assigned motor"
                        )
                    change = abs(positions[candidate] - baseline[candidate]) % 4096
                    delta = min(change, 4096 - change)
                with self.lock:
                    if token != self.observation_token or event.is_set():
                        return
                    self.detected_at = now if candidate is not None else None
                    self._publish(positions=positions, detected_id=candidate, candidate_delta=delta)
                event.wait(0.1)
            with self.lock:
                if token != self.observation_token or event.is_set() or self.stop_event.is_set():
                    return
                self.operation = None
                self.detected_at = None
            self._publish(
                phase="ready",
                detected_id=None,
                candidate_delta=None,
                message="Observation timed out. Select Observe again and move one joint at a time.",
            )
        except Exception as error:
            if (
                not event.is_set()
                and not self.stop_event.is_set()
                and token == self.observation_token
            ):
                self._fail(error)

    def wiggle(self, motor_id):
        with self.lock:
            self._ensure_live()
            if type(motor_id) is not int or motor_id not in self.state["ids"]:
                raise ValueError("Select a discovered motor address")
            if motor_id in self.state["mapping"].values():
                raise ValueError("This motor already has a confirmed joint")
            self.observation_event.set()
            self.observation_token += 1
            self.detected_at = self.wiggle_at = None
            self.operation = "wiggle"
            self._publish(
                phase="wiggling",
                error=None,
                wiggle_id=None,
                wiggle_ticks=None,
                detected_id=None,
                selected_joint=None,
                message=f"Identifying address {motor_id}; support the arm",
            )
        self._thread(lambda: self._wiggle(motor_id))

    def _wiggle(self, motor_id):
        try:
            ticks = self.engine.identify(
                motor_id, lambda active: self._publish(motion_active=active)
            )
            with self.lock:
                if self.stop_event.is_set():
                    return
                self.operation = None
                self.wiggle_at = time.monotonic()
                self._publish(
                    phase="ready",
                    motion_active=False,
                    wiggle_id=motor_id,
                    wiggle_ticks=ticks,
                    message=f"Address {motor_id} motion finished. Select the joint you saw move.",
                )
        except IdentificationUnavailable as error:
            with self.lock:
                self.operation = None
                self.wiggle_at = None
            self._publish(
                phase="ready",
                motion_active=False,
                wiggle_id=None,
                error=str(error),
                message=str(error),
            )
        except Exception as error:
            self._fail(error)

    def accept_wiggle(self, joint, motor_id):
        with self.lock:
            self._ensure_live()
            if joint not in JOINTS or joint in self.state["mapping"]:
                raise ValueError("Choose an unconfirmed joint")
            if (
                type(motor_id) is not int
                or motor_id != self.state["wiggle_id"]
                or self.wiggle_at is None
                or time.monotonic() - self.wiggle_at > 60
            ):
                raise ValueError("Request a fresh wiggle before confirming its joint")
            self._accept_mapping(joint, motor_id)

    def accept(self, joint, motor_id):
        with self.lock:
            self._ensure_live()
            if joint not in JOINTS or joint != self.state["selected_joint"]:
                raise ValueError("Observe the selected joint before confirming its motor")
            if (
                type(motor_id) is not int
                or motor_id != self.state["detected_id"]
                or motor_id not in self.state["ids"]
            ):
                raise ValueError("Confirm the current detected motor ID")
            if motor_id in self.state["mapping"].values():
                raise ValueError("That motor is already assigned to another joint")
            if self.detected_at is None or time.monotonic() - self.detected_at > 1.5:
                raise ValueError(
                    "Movement evidence is stale. Move the highlighted joint and confirm again."
                )
            self._accept_mapping(joint, motor_id)

    def _accept_mapping(self, joint, motor_id):
        if motor_id in self.state["mapping"].values():
            raise ValueError("That motor is already assigned to another joint")
        mapping = {**self.state["mapping"], joint: motor_id}
        directory = self.data_dir / "assembled"
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        receipt = directory / (
            hashlib.sha256(self.port.encode()).hexdigest()[:20] + ".mapping.json"
        )
        durable_json(receipt, {"port": self.port, "ids": self.state["ids"], "mapping": mapping})
        self.observation_event.set()
        self.observation_token += 1
        self.operation = None
        self.detected_at = None
        self._publish(
            mapping=mapping,
            wiggle_id=None,
            wiggle_ticks=None,
            phase="review" if len(mapping) == 6 else "ready",
            detected_id=None,
            candidate_delta=None,
            message=("Joint identity confirmed. Review all six mappings before saving motor IDs."),
        )

    def finish(self):
        with self.lock:
            self._ensure_live()
            mapping = dict(self.state["mapping"])
            if (
                set(mapping) != set(JOINTS)
                or set(mapping.values()) != set(self.state["ids"])
                or len(set(mapping.values())) != 6
            ):
                raise ValueError(
                    "Confirm all six different joint identities before saving motor IDs"
                )
            self.observation_event.set()
            self.observation_token += 1
            self.operation = "apply"
        self._publish(
            phase="applying",
            detected_id=None,
            message=(
                "Saving the reviewed motor IDs. Keep supporting the arm "
                "during the final verification reboot."
            ),
        )
        self._thread(lambda: self._apply(mapping))

    def _apply(self, mapping):
        try:
            ids = self.engine.apply(mapping)
            if sorted(ids) != list(range(1, 7)):
                raise ValueError("Canonical motor IDs were not independently verified")
            self._close()
            with self.lock:
                self.operation = None
            self._publish(
                phase="complete",
                ids=list(range(1, 7)),
                mapping={joint: index + 1 for index, joint in enumerate(JOINTS)},
                message=(
                    "Motor IDs saved and verified after reboot. Keep the chain "
                    "connected and calibrate the arm next."
                ),
                terminal=True,
            )
        except Exception as error:
            self._fail(error)

    def cancel(self):
        self.stop_event.set()
        self.observation_event.set()
        with self.lock:
            self.observation_token += 1
            self.operation = None
        # An already-started ID transaction may finish its essential EEPROM
        # re-lock before this lock is released. No later transaction can start.
        cleanup_error = None
        try:
            self._close()
        except Exception as caught:
            cleanup_error = caught
        deadline = time.monotonic() + 0.25
        for worker in list(self.workers):
            if worker is not threading.current_thread():
                worker.join(timeout=max(0, deadline - time.monotonic()))
        self._publish(
            phase="stopped",
            detected_id=None,
            candidate_delta=None,
            error=str(cleanup_error)[:1000] if cleanup_error else self.state.get("error"),
            message=("Stop cleanup failed: " + str(cleanup_error))[:1000]
            if cleanup_error
            else (
                "Setup stopped. Partial motor IDs are preserved; restart setup to scan and resume."
            ),
            terminal=True,
        )
        if cleanup_error is not None:
            raise RuntimeError(
                "Post-reboot torque-off could not be confirmed: " + str(cleanup_error)
            ) from cleanup_error
