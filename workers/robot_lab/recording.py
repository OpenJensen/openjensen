"""Measured recording cadence and server-owned collection timers; no motor commands."""

import math
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

MAX_RECORDING_FPS = 40


class CaptureRate:
    """Measure distinct control and camera samples before fixing dataset timing."""

    def __init__(self, started, camera):
        self.started = started
        self.camera = camera
        self.ticks = deque(maxlen=240)
        self.images = deque(maxlen=240)
        self.io_costs = []
        self.io_error = None
        self.io_required = False
        self.io_thread = None

    def measure_image_io(self, frame, root):
        """Probe actual PNG encoding/writes off the motor thread, retaining no frames."""
        self.io_required = True

        def measure():
            from image_writer import save_png

            try:
                with tempfile.TemporaryDirectory(prefix=".capture-check-", dir=root) as folder:
                    # Exclude one-time Pillow/plugin initialization from sustained throughput.
                    save_png(frame[:, :, ::-1].copy(), Path(folder) / "sample.png")
                    for _ in range(6):
                        started = time.monotonic()
                        save_png(frame[:, :, ::-1].copy(), Path(folder) / "sample.png")
                        (Path(folder) / "sample.json").write_text('{"sample": [0,0,0,0,0,0]}')
                        self.io_costs.append(time.monotonic() - started)
            except Exception as error:
                self.io_error = str(error)

        self.io_thread = threading.Thread(
            target=measure, name="robot-lab-capture-check", daemon=True
        )
        self.io_thread.start()

    def sample(self, now, image_time=None):
        if not self.ticks or now > self.ticks[-1]:
            self.ticks.append(now)
        if image_time is not None and image_time >= self.started:
            if not self.images or image_time > self.images[-1]:
                self.images.append(image_time)

    def select(self, now):
        if self.io_error:
            raise ValueError("Recording storage check failed: " + self.io_error)
        if self.io_required and len(self.io_costs) < 6:
            return None
        streams = [self.ticks] + ([self.images] if self.camera else [])
        if now - self.started < 3 or any(len(values) < 6 for values in streams):
            return None
        # Slow-tail intervals include real serial work and USB camera cadence.
        intervals = []
        for values in streams:
            gaps = sorted(b - a for a, b in zip(values, list(values)[1:]) if b > a)
            intervals.append(gaps[min(len(gaps) - 1, math.ceil(len(gaps) * 0.9) - 1)])
        if self.io_costs:
            intervals.append(max(self.io_costs))
        sustainable = 0.85 / max(intervals)
        if sustainable < 1:
            raise ValueError(
                "Capture is below 1 FPS. Check camera connection or computer load, then retry."
            )
        return min(MAX_RECORDING_FPS, max(1, math.floor(sustainable)))


class Collection:
    def __init__(self, config):
        if not isinstance(config, dict) or set(config) - {
            "episode_time_s",
            "reset_time_s",
            "num_episodes",
            "countdown_s",
        }:
            raise ValueError("Invalid episode collection settings")
        defaults = {"episode_time_s": 30, "reset_time_s": 10, "num_episodes": 10, "countdown_s": 3}
        bounds = {
            "episode_time_s": (1, 300),
            "reset_time_s": (0, 300),
            "num_episodes": (1, 1000),
            "countdown_s": (0, 30),
        }
        self.config = defaults | config
        for key, value in self.config.items():
            low, high = bounds[key]
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{key} must be an integer between {low} and {high}")
        self.phase = "idle"
        self.deadline = None
        self.completed = 0
        self.cue_id = 0
        self.cue = "Ready to collect demonstrations"
        self.saving = False

    def transition(self, phase, cue, now, seconds=None):
        self.phase, self.cue = phase, cue
        self.cue_id += 1
        self.deadline = None if seconds is None else now + seconds

    def pause(self, now, cue="Collection paused"):
        if self.phase not in {"idle", "complete", "paused"}:
            self.transition("paused", cue, now)

    def snapshot(self, now):
        return {
            "enabled": True,
            **self.config,
            "phase": self.phase,
            "remaining_s": max(0, self.deadline - now) if self.deadline is not None else 0,
            "completed": self.completed,
            "cue_id": self.cue_id,
            "cue": self.cue,
            "saving": self.saving,
        }
