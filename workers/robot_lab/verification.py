"""One bounded wiggle of a saved mapping; no ID, calibration or journal writes."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from pathlib import Path

from assembled import AssembledSetup
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
        except (ValueError, OSError, TypeError, AttributeError):
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

    def check_mapping(self, motor_id, on_motion):
        saved_mapping(self.journal_path.parent.parent, self.port)
        self.state["ids"] = list(range(1, 7))
        self.deadline = time.monotonic() + 12
        self._open()
        return self.identify(motor_id, on_motion)


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
