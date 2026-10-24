"""One bounded wiggle of a saved mapping; no ID, calibration or journal writes."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from pathlib import Path

from assembled import AssembledCancelled, AssembledError, AssembledSetup
from assembled_session import durable_json
from hardware import IDS


def saved_mapping(data_dir, port):
    path = Path(data_dir) / "assembled" / (hashlib.sha256(port.encode()).hexdigest()[:20] + ".json")
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 256 * 1024:
        raise ValueError("Save the six-joint mapping before checking motor motion")
    record = json.loads(path.read_text())
    if (
        record.get("port") != port
        or record.get("phase") != "complete"
        or record.get("mapping") != IDS
        or record.get("ids") != list(range(1, 7))
        or record.get("pending") is not None
        or not record.get("reboot_cleanup", {}).get("verified_off")
        or record.get("reboot_cleanup", {}).get("pending")
    ):
        raise ValueError("Complete and save motor mapping before checking motion")
    return {"port": port, "mapping": dict(IDS), "ids": list(range(1, 7))}


def latest_saved_mapping(data_dir):
    for path in sorted(
        (Path(data_dir) / "assembled").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        try:
            if path.is_symlink() or path.stat().st_size > 256 * 1024:
                continue
            port = json.loads(path.read_text()).get("port")
            if isinstance(port, str):
                return saved_mapping(data_dir, port)
        except ValueError, OSError, TypeError, AttributeError:
            continue
    return None


def empty_verification():
    return {
        "phase": "idle",
        "port": None,
        "joint": None,
        "motor_id": None,
        "ticks": None,
        "motion_active": False,
        "error": None,
        "run_id": None,
    }


class VerificationEngine(AssembledSetup):
    # A verification scan must never replace the completed setup recovery journal.
    def _publish(self, journal=False, **updates):
        return super()._publish(journal=False, **updates)

    def _check(self):
        if self.stop_event.is_set():
            raise AssembledCancelled("Motor verification stopped")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise AssembledError("Motor verification timed out; check the USB connection")

    def check_mapping(self, motor_id, on_motion):
        from identify import wiggle

        saved_mapping(self.journal_path.parent.parent, self.port)
        self.state["ids"] = list(range(1, 7))
        self.deadline = time.monotonic() + 12
        self._open()
        with self.io_lock:
            # The completed journal fixes the six addresses. Repeating discovery
            # over 254 addresses can exhaust the USB time budget before motion.
            # Still validate stable encoder replies, model and stored ID for all
            # six motors, then require every torque register to be off.
            ids = self._verify_six(list(IDS.values()))
            if any(self._read(sid, 40) != 0 for sid in ids):
                raise AssembledError("Release every motor's torque before verification")

            def motion_state(active):
                self.identification_cleanup = active
                on_motion(active)

            return wiggle(self.bus, motor_id, ids, self.stop_event, motion_state=motion_state)


class MappingVerification:
    def __init__(self, port, joint, data_dir, on_change, *, engine_factory=None):
        import uuid

        self.lock = threading.RLock()
        self.io_lock = threading.RLock()
        self.stop_event = threading.Event()
        self.on_change = on_change
        self.state = {
            **empty_verification(),
            "phase": "checking",
            "port": port,
            "joint": joint,
            "motor_id": IDS[joint],
            "run_id": uuid.uuid4().hex,
        }
        self.engine = (engine_factory or VerificationEngine)(
            port, data_dir, stop_event=self.stop_event, io_lock=self.io_lock, verified_so101=True
        )
        self.worker = threading.Thread(
            target=self._run, daemon=True, name="robot-lab-mapping-check"
        )

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.state)

    def _publish(self, terminal=False, **changes):
        with self.lock:
            self.state.update(changes)
            snapshot = self.snapshot()
        self.on_change(self, snapshot, terminal)

    def start(self):
        self.worker.start()

    def _motion(self, active):
        self._publish(motion_active=active, phase="wiggling" if active else "checking")

    def _run(self):
        error, ticks = None, None
        try:
            ticks = self.engine.check_mapping(self.state["motor_id"], self._motion)
        except Exception as caught:
            error = str(caught)
        finally:
            try:
                self.engine.close()
            except Exception as caught:
                error = (error + "; " if error else "") + "Cleanup: " + str(caught)
            self._publish(
                terminal=True,
                phase="stopped" if self.stop_event.is_set() else "error" if error else "done",
                ticks=ticks,
                error=error,
            )

    def cancel(self):
        self.stop_event.set()
        # The engine's bounded serial operations and wiggle cleanup finish first.
        self.worker.join(timeout=15)
        if self.worker.is_alive():
            raise ValueError("Motor verification cleanup is still running")


class MappingChecks:
    """Operator observations, bound to the exact completed setup journal."""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.lock = threading.RLock()

    def revision(self, port):
        saved_mapping(self.data_dir, port)
        path = (
            self.data_dir / "assembled" / (hashlib.sha256(port.encode()).hexdigest()[:20] + ".json")
        )
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def path(self, port):
        return (
            self.data_dir / "mapping-checks" / (hashlib.sha256(port.encode()).hexdigest() + ".json")
        )

    def snapshot(self, port):
        with self.lock:
            empty = {"port": port, "checks": {}, "revision": None, "error": None}
            if not port:
                return empty
            try:
                revision = self.revision(port)
                empty["revision"] = revision
                path = self.path(port)
                if not path.exists():
                    return empty
                if path.is_symlink() or path.stat().st_size > 65536:
                    raise ValueError("Saved motion checks need inspection")
                value = json.loads(path.read_text())
                if value.get("revision") != revision or value.get("port") != port:
                    return empty
                checks = value["checks"]
                if not isinstance(checks, dict) or any(
                    joint not in IDS
                    or not isinstance(entry, dict)
                    or type(entry.get("correct")) is not bool
                    or not isinstance(entry.get("run_id"), str)
                    or not isinstance(entry.get("checked_at"), (int, float))
                    for joint, entry in checks.items()
                ):
                    raise ValueError("Saved motion checks are invalid")
                return {**empty, "checks": copy.deepcopy(checks)}
            except (ValueError, KeyError, TypeError, AttributeError, OSError) as error:
                return {**empty, "error": str(error)}

    def record(self, verification, revision, correct):
        with self.lock:
            port, joint = verification["port"], verification["joint"]
            state = self.snapshot(port)
            if state["error"] or state["revision"] != revision:
                raise ValueError("Motor mapping changed; repeat the motion check")
            old = state["checks"].get(joint)
            if old and old["run_id"] == verification["run_id"]:
                if old["correct"] != correct:
                    raise ValueError("Repeat the wiggle before changing its observation")
                return
            state["checks"][joint] = {
                "run_id": verification["run_id"],
                "correct": correct,
                "checked_at": time.time(),
            }
            durable_json(self.path(port), state)
