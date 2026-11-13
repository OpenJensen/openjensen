"""Independent recorder. Servo loop publishes to a bounded queue, never disk."""

from __future__ import annotations

import base64
import json
import os
import queue
import shutil
import threading
import time

from agent_control import UNITS, durable_json, identity
from dataset import DatasetWriter
from hardware import JOINTS


class AgentCollection:
    def __init__(self, lab):
        self.lab = lab
        self.guard = threading.Lock()
        self.done = threading.Event()
        self.thread = None
        self.writer = None
        self.state = {"status": "idle"}
        self.samples = queue.Queue(maxsize=500)
        self.latest = None

    def status(self):
        return dict(self.state)

    def sample(self, value):
        if self.state.get("status") != "recording":
            return
        try:
            self.samples.put_nowait(value)
            self.latest = value
        except queue.Full:
            self.state.update(error="Controller evidence queue overflow", status="failed")
            self.done.set()

    def start(self, payload):
        with self.guard:
            if self.thread and self.thread.is_alive():
                raise ValueError("An agent recording is already active")
            run_id = identity(payload["run_id"])
            indices = payload["camera_indices"]
            fps = payload.get("fps", 10)
            if (
                not isinstance(indices, list)
                or len(indices) != 2
                or any(type(i) is not int or not 0 <= i <= 63 for i in indices)
                or len(set(indices)) != 2
                or type(fps) is not int
                or not 1 <= fps <= 20
            ):
                raise ValueError("Choose two distinct cameras and 1–20 FPS")
            if self.lab.mode != "agent" or self.lab.agent.owner != run_id:
                raise ValueError("Acquire the bounded agent controller first")
            if self.lab.writer and not self.lab.writer.finalized:
                raise ValueError("Finalize the manual recording first")
            if shutil.disk_usage(self.lab.datasets_dir).free < 10 * 1024**3:
                raise ValueError("Agent collection requires 10 GiB disk headroom")
            self.done.clear()
            self.latest = None
            self.samples = queue.Queue(maxsize=500)
            self.indices = indices
            self.generation = self.lab.generation
            shapes = {}
            for role, index in zip(("wrist", "side"), indices):
                frame = self.lab.camera.read(index, keep_indices=indices)
                shapes["observation.images." + role] = tuple(frame.shape)
                self.lab.camera.start_reader(index)
            self.writer = DatasetWriter(
                self.lab.datasets_dir, "agent-" + run_id, payload["task"], fps, shapes
            )
            self.root = self.writer.root
            self.evidence = self.root / "evidence"
            self.evidence.mkdir(mode=0o700)
            self.state = {
                "status": "recording",
                "run_id": run_id,
                "episode_id": run_id + "-0",
                "dataset_id": self.writer.id,
                "frames": 0,
                "error": None,
                "dropped_samples": 0,
            }
            self.metadata = {
                "schema_version": 1,
                **self.state,
                "camera_roles": dict(zip(("wrist", "side"), indices)),
                "joint_order": list(JOINTS),
                "units": UNITS,
                "calibration": self.lab.calibration,
                "robot_id": self.lab.robot_id,
                "clock": "companion host monotonic_ns",
                "origin_monotonic_ns": time.monotonic_ns(),
                "origin_unix_ns": time.time_ns(),
                "sensor_timestamp_verified": False,
                "sensor_dropped_frames": None,
                "camera_time": "host frame read completion; not sensor exposure time",
                "action_representation": (
                    "last successfully sent controller position setpoint; degrees / gripper percent"
                ),
                "acknowledgment": "transport write returned, not per-motor goal acknowledgment",
                "alignment": (
                    "latest controller sample preceding paired frame assembly; maximum age 250 ms"
                ),
                "lerobot_timestamp": (
                    "frame_index / fps; raw irregular times retained in evidence/frames.jsonl"
                ),
                "environment_kind": payload.get("environment_kind", "physical"),
                "model": payload.get("model"),
                "task_success_verified": False,
            }
            self.writer.start_episode()
            durable_json(self.evidence / "recording.json", self.metadata)
            self.thread = threading.Thread(target=self._record, daemon=True, name="agent-recording")
            self.thread.start()
            return self.status()

    def stop(self, payload):
        if payload.get("run_id") != self.state.get("run_id"):
            raise ValueError("Unknown recording run")
        self.done.set()
        self.lab.stop(
            "Agent collection stopped", expected_generation=self.generation, expected_mode="agent"
        )
        return self.status()

    def observation(self, payload):
        if payload.get("run_id") != self.state.get("run_id"):
            raise ValueError("Unknown recording run")
        oid = identity(payload["observation_id"])
        directory = self.evidence / "observations"
        directory.mkdir(exist_ok=True)
        if not payload["images"] or not set(payload["images"]) <= {"wrist", "side"}:
            raise ValueError("Choose known observation image roles")
        for role, encoded in payload["images"].items():
            data = base64.b64decode(encoded, validate=True)
            if len(data) > 2 * 1024**2 or not data.startswith(b"\xff\xd8"):
                raise ValueError("Invalid model image")
            (directory / (oid + "-" + role + ".jpg")).write_bytes(data)
        durable_json(directory / (oid + ".json"), payload["metadata"])
        return {"saved": True}

    def event(self, payload):
        if not self.writer or payload.get("run_id") != self.state.get("run_id"):
            raise ValueError("Unknown recording run")
        encoded = json.dumps(payload.get("event"), allow_nan=False)
        if len(encoded) > 12000:
            raise ValueError("Trace event too large")
        with self.guard:
            with (self.evidence / "trace.jsonl").open("a") as stream:
                stream.write(encoded + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        return {"saved": True}

    def _record(self):
        previous = {}
        next_frame = time.monotonic()
        try:
            with (
                (self.evidence / "controller.jsonl").open("a") as actions,
                (self.evidence / "frames.jsonl").open("a") as frames,
            ):
                while not self.done.is_set():
                    if self.lab.generation != self.generation or self.lab.mode != "agent":
                        raise ValueError("Controller stopped during collection")
                    while True:
                        try:
                            sample = self.samples.get_nowait()
                        except queue.Empty:
                            break
                        actions.write(json.dumps(sample, allow_nan=False) + "\n")
                    actions.flush()
                    now = time.monotonic()
                    if now < next_frame:
                        self.done.wait(min(0.01, next_frame - now))
                        continue
                    sample = self.latest
                    if sample is None:
                        self.done.wait(0.01)
                        continue
                    captured = time.monotonic_ns()
                    if captured - sample["state_read_monotonic_ns"] > 250_000_000:
                        raise ValueError("Measured controller state is stale")
                    images, times = {}, {}
                    for role, index in zip(("wrist", "side"), self.indices):
                        frame, timestamp = self.lab.camera.read_latest_timed(index)
                        stamp = int(timestamp * 1e9)
                        if stamp <= previous.get(index, 0) or captured - stamp > 500_000_000:
                            raise ValueError("Missing, repeated or stale " + role + " camera frame")
                        times[role] = stamp
                        images["observation.images." + role] = frame[:, :, ::-1]
                        previous[index] = stamp
                    if max(times.values()) - min(times.values()) > 150_000_000:
                        raise ValueError("Camera pair exceeds 150 ms skew")
                    index = self.writer.frames
                    self.writer.add_frame(
                        sample["measured"], sample["sent"], images, captured / 1e9
                    )
                    frames.write(
                        json.dumps(
                            {
                                "run_id": self.state["run_id"],
                                "episode_id": self.state["episode_id"],
                                "frame_index": index,
                                "capture_monotonic_ns": captured,
                                "camera_monotonic_ns": times,
                                "controller_sample_id": sample["sample_id"],
                                "command_id": sample["command_id"],
                                "state_read_monotonic_ns": sample["state_read_monotonic_ns"],
                                "write_returned_monotonic_ns": sample[
                                    "write_returned_monotonic_ns"
                                ],
                            }
                        )
                        + "\n"
                    )
                    frames.flush()
                    self.state["frames"] = self.writer.frames
                    next_frame += 1 / self.writer.fps
                    if now - next_frame >= 1 / self.writer.fps:
                        self.state["dropped_samples"] += 1
                        next_frame = now + 1 / self.writer.fps
                if self.state["status"] == "failed":
                    raise ValueError(self.state["error"])
                while not self.samples.empty():
                    actions.write(json.dumps(self.samples.get_nowait()) + "\n")
                actions.flush()
                os.fsync(actions.fileno())
                os.fsync(frames.fileno())
        except Exception as error:
            self.state.update(status="failed", error=str(error))
            self.lab.stop(
                "Agent recording failed: " + str(error),
                expected_generation=self.generation,
                expected_mode="agent",
            )
        finally:
            self.writer.episode_active = False
            try:
                if self.writer.image_writer:
                    self.writer.image_writer.retry_failed()
                if self.writer.pending.exists():
                    shutil.copytree(self.writer.pending, self.root / "raw", dirs_exist_ok=True)
                if self.writer.frames >= 2:
                    self.writer.save_episode()
                    self.writer.finalize()
                if self.writer.total_frames < 2:
                    self.state.update(
                        status="failed", error="Fewer than two frames; raw capture retained"
                    )
                if self.state["status"] == "recording":
                    self.state["status"] = "saved"
            except Exception as error:
                self.state.update(status="failed", error="Finalization failed: " + str(error))
            finally:
                self.writer.close()
                executions = self.evidence / "executions"
                executions.mkdir(exist_ok=True)
                for command_id, receipt in list(self.lab.agent.cache.items()):
                    if receipt.get("agent_session_id") == self.state["run_id"]:
                        durable_json(executions / (identity(command_id) + ".json"), receipt)
                durable_json(self.evidence / "recording.json", {**self.metadata, **self.state})

    def finish(self, payload):
        if payload.get("run_id") != self.state.get("run_id"):
            raise ValueError("Unknown recording run")
        self.done.set()
        if self.thread:
            self.thread.join(timeout=20)
        if self.thread and self.thread.is_alive():
            return {**self.state, "status": "finalizing"}
        return self.status()
