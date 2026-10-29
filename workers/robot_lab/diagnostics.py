"""Bounded, private diagnostics; disk I/O never runs on the motor/Stop threads."""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

MAX_BYTES = 4 * 1024 * 1024
BACKUPS = 7


class Diagnostics:
    def __init__(self, root, *, max_bytes=MAX_BYTES, backups=BACKUPS):
        self.root = Path(root) / "diagnostics"
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.max_bytes, self.backups = max_bytes, backups
        self.boot_id = uuid.uuid4().hex
        self.lock = threading.Lock()
        self.events = deque(maxlen=100)
        self.samples = deque(maxlen=120)
        self.last_stop = None
        self.error = None
        self.dropped = 0
        self.pending = queue.Queue(maxsize=2048)
        self.closed = threading.Event()
        self.last_sample = 0.0
        self._recover()
        self.worker = threading.Thread(target=self._run, name="robot-lab-diagnostics", daemon=True)
        self.worker.start()
        self.emit("companion_started")

    def _recover(self):
        try:
            path = self.root / "last-stop.json"
            if path.is_file():
                self.last_stop = json.loads(path.read_text())
            # Read bounded tails, including the previous file across rotation.
            for filename, target in (
                ("events.jsonl", self.events),
                ("telemetry.jsonl", self.samples),
            ):
                for suffix in (".1", ""):
                    path = self.root / (filename + suffix)
                    if not path.is_file():
                        continue
                    with path.open("rb") as stream:
                        stream.seek(max(0, path.stat().st_size - 512 * 1024))
                        for line in stream:
                            try:
                                event = json.loads(line)
                                if isinstance(event, dict) and "event" in event:
                                    target.append(event)
                            except ValueError, UnicodeDecodeError:
                                continue  # Recover after an interrupted final write.
        except (OSError, ValueError) as error:
            self.error = "Cannot recover Robot Lab diagnostics: " + str(error)

    def emit(self, event, **details):
        entry = {
            "id": uuid.uuid4().hex,
            "time": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "boot_id": self.boot_id,
            "event": event,
            **details,
        }
        with self.lock:
            (self.samples if event == "position_sample" else self.events).append(entry)
            if event == "session_stopped" and details.get("retain_as_last_stop"):
                self.last_stop = entry
        try:
            self.pending.put_nowait(entry)
        except queue.Full:
            with self.lock:
                self.dropped += 1
                self.error = "Diagnostic writer queue full; some entries could not be saved"
        return entry

    def sample(self, **details):
        now = time.monotonic()
        with self.lock:
            if now - self.last_sample < 2:
                return
            self.last_sample = now
        self.emit("position_sample", **details)

    def snapshot(self, *, full=False):
        with self.lock:
            result = {
                "enabled": True,
                "directory": str(self.root),
                "boot_id": self.boot_id,
                "error": self.error,
                "dropped_entries": self.dropped,
                "retention_bytes": self.max_bytes * (self.backups + 1) * 2,
                "last_stop": self.last_stop,
                "events": list(self.events) if full else list(self.events)[-8:],
            }
            if full:
                result["samples"] = list(self.samples)
            return result

    def _append(self, filename, entry):
        path = self.root / filename
        body = (json.dumps(entry, allow_nan=False, separators=(",", ":")) + "\n").encode()
        needs_newline = False
        if path.exists() and path.stat().st_size:
            with path.open("rb") as stream:
                stream.seek(-1, os.SEEK_END)
                needs_newline = stream.read(1) != b"\n"
        if needs_newline:
            body = b"\n" + body
        if path.exists() and path.stat().st_size + len(body) > self.max_bytes:
            for index in range(self.backups, 0, -1):
                previous = path if index == 1 else self.root / (filename + f".{index - 1}")
                if previous.exists():
                    os.replace(previous, self.root / (filename + f".{index}"))
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "ab") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(body)
            stream.flush()
            if entry["event"] != "position_sample":
                os.fsync(stream.fileno())
        if entry["event"] == "session_stopped" and entry.get("retain_as_last_stop"):
            temporary = self.root / ".last-stop.tmp"
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.root / "last-stop.json")

    def _run(self):
        while not self.closed.is_set() or not self.pending.empty():
            try:
                entry = self.pending.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                filename = (
                    "telemetry.jsonl" if entry["event"] == "position_sample" else "events.jsonl"
                )
                self._append(filename, entry)
            except (OSError, ValueError, TypeError) as error:
                with self.lock:
                    self.error = "Robot Lab diagnostics could not be saved: " + str(error)
            finally:
                self.pending.task_done()

    def close(self):
        self.closed.set()
        self.worker.join(timeout=2)
