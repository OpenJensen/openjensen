"""Lossless local capture journal; only finalized episodes may enter the writer."""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

from .contracts import Settings, canonical


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_new(path: Path, value: dict) -> None:
    if os.path.lexists(path):
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    # This directory is exclusively owned by one session, never untrusted writers.
    temporary.rename(path)
    sync_directory(path.parent)


class Journal:
    def __init__(self, root: Path, settings: Settings):
        self.root = root
        self.settings = settings
        root.mkdir(exist_ok=False)
        sync_directory(root.parent)
        self.session_id = uuid.uuid4().hex
        atomic_new(
            root / "session.json",
            {"schema_version": 1, "session_id": self.session_id, **settings.metadata()},
        )
        self.current = None
        self.count = 0
        self.bytes = 0
        self.events = []
        self.stream = None
        self.event_stream = None

    def begin(self, episode: str, task: str) -> None:
        if self.current is not None:
            raise RuntimeError("An episode is already being recorded")
        self.current = self.root / episode
        self.current.mkdir()
        sync_directory(self.root)
        (self.current / "frames").mkdir()
        self.stream = (self.current / "trajectory.jsonl").open("xb")
        self.event_stream = (self.current / "events.jsonl").open("xb")
        sync_directory(self.current)
        self.count = self.bytes = 0
        self.events = []
        atomic_new(
            self.current / "started.json",
            {"schema_version": 1, "episode_id": episode, "instruction": task},
        )

    def event(self, value: dict) -> None:
        if self.current is not None:
            if len(self.events) >= 512:
                raise RuntimeError("Episode intervention limit exceeded")
            self.event_stream.write(canonical(value) + b"\n")
            self.event_stream.flush()
            os.fsync(self.event_stream.fileno())
            self.events.append(value)

    def append(self, record: dict, rgb: bytes) -> None:
        if self.current is None or self.stream is None:
            raise RuntimeError("No active recording")
        if self.count >= self.settings.max_steps:
            raise ValueError("Episode step limit exceeded")
        if len(rgb) != self.settings.sim.width * self.settings.sim.height * 3:
            raise ValueError("RGB size differs from capture contract")
        name = f"frames/{self.count:06d}.rgb"
        value = record | {"frame": name, "rgb_sha256": hashlib.sha256(rgb).hexdigest()}
        row = canonical(value) + b"\n"
        if self.bytes + len(row) + len(rgb) > self.settings.max_episode_bytes:
            raise ValueError("Episode byte limit exceeded")
        with (self.current / name).open("xb") as frame:
            frame.write(rgb)
            frame.flush()
            os.fsync(frame.fileno())
        sync_directory(self.current / "frames")
        self.stream.write(row)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.count += 1
        self.bytes += len(rgb) + len(row)

    def finish(self, termination: str, outcome: str) -> None:
        if self.current is None:
            return
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.event_stream.flush()
        os.fsync(self.event_stream.fileno())
        self.stream.close()
        self.event_stream.close()
        # Commit last. A missing/partial receipt makes the capture inadmissible.
        atomic_new(
            self.current / "episode.json",
            {
                "schema_version": 1,
                "episode_id": self.current.name,
                "finalized": True,
                "frames": self.count,
                "bytes": self.bytes,
                "termination": termination,
                "outcome": outcome,
                "events": self.events,
                "events_sha256": hashlib.sha256(
                    (self.current / "events.jsonl").read_bytes()
                ).hexdigest(),
                "trajectory_sha256": hashlib.sha256(
                    (self.current / "trajectory.jsonl").read_bytes()
                ).hexdigest(),
            },
        )
        self.current = self.stream = self.event_stream = None

    def abort(self) -> None:
        # Keep every captured byte; never claim finalization after capture failure.
        if self.stream is not None:
            self.stream.close()
        if self.event_stream is not None:
            self.event_stream.close()
        self.current = self.stream = self.event_stream = None
