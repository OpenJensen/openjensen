"""Independent recorder. Servo loop publishes to a bounded queue, never disk."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import queue
import shutil
import threading
import time
from pathlib import Path

from agent_control import LIMITS, UNITS, digest, durable_json, identity
from collection_streams import CameraStreams
from dataset import DatasetWriter
from hardware import JOINTS
from lerobot_control import PROFILE


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
        return {**self.state, "contract_version": 2, "robot_clock_ns": time.monotonic_ns()}

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
            if (self.lab.datasets_dir / ("agent-" + run_id)).exists():
                raise ValueError("Episode identity already exists")
            self.sealed = False
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
                "schema_version": 2,
                "run_parameters": payload.get("run_parameters"),
                "task": payload["task"],
                "experiment": payload.get("experiment", {}),
                "session_id": self.lab.agent.target_session_id,
                "source_family_id": payload.get("source_family_id", run_id + "-0"),
                "collection_source": "supervised_pi"
                if payload.get("environment_kind") == "physical"
                else "synthetic_pi",
                "calibration_sha256": self.lab.agent.calibration_hash(),
                "control_hz_nominal": 40,
                "controller": {
                    "profile": PROFILE,
                    "limits": LIMITS,
                    "version": digest([PROFILE, LIMITS]),
                },
                "kinematics": payload.get("kinematics"),
                "controller_source_sha256": {
                    name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in (
                        "server.py",
                        "agent_control.py",
                        "lerobot_control.py",
                        "agent_collection.py",
                        "collection_streams.py",
                        "camera.py",
                    )
                },
                "application_build": payload.get("application_build"),
                "device_clocks": {
                    "motors": {
                        "clock_id": None,
                        "offset_ns": None,
                        "drift_ppm": None,
                        "uncertainty_ns": None,
                    },
                    "feedback_timestamp": "host read completion",
                },
                "hardware_firmware_version": None,
                "contact_force": None,
                "camera_configuration": {
                    role: {
                        "index": i,
                        "shape_hwc": shapes["observation.images." + role],
                        "mode": getattr(self.lab.camera, "negotiated_modes", {}).get(i),
                        "intrinsics": None,
                        "extrinsics": None,
                        "sensor_clock_id": None,
                        "robot_arrival_clock_id": "robot_monotonic",
                        "sensor_offset_ns": None,
                        "sensor_drift_ppm": None,
                        "sensor_uncertainty_ns": None,
                    }
                    for role, i in zip(("wrist", "side"), indices)
                },
                "outcome_label": "unknown",
                "outcome_evidence": [],
                "recovery": None,
                "human_intervention": None,
                "hardware_model": "SO101",
                "camera_geometry_version": None,
                "clock_domain_ids": {
                    "controller": "robot_monotonic",
                    "core": "application_monotonic",
                    "worker": "node_hrtime",
                    "wall": "unix_utc",
                },
                "clock_unit": "ns",
                "reset_minutes": None,
                "supervision_minutes": None,
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
            self.metadata["camera_configuration_version"] = digest(
                self.metadata["camera_configuration"]
            )
            self.writer.start_episode()
            durable_json(self.evidence / "recording.json", self.metadata)
            self.streams = CameraStreams(self.evidence, dict(zip(indices, ("wrist", "side"))))
            self.lab.camera.set_collection_sink(self.streams.capture)
            self.stream_start_ns = time.monotonic_ns()
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

    def side_frame(self, payload):
        if self.state.get("status") != "recording" or payload.get("run_id") != self.state.get(
            "run_id"
        ):
            raise ValueError("Recording is not active")
        capture = self.lab.camera.latest_timed_jpeg(self.indices[1])
        stamp = (
            int(capture["captured_monotonic"] * 1e9)
            if "captured_monotonic" in capture
            else capture["read_end_monotonic_ns"]
        )
        return {
            "jpeg": base64.b64encode(capture["jpeg"]).decode(),
            "read_complete_robot_ns": stamp,
            "frame_id": f"side-{stamp}",
            "encoded_available_robot_ns": time.monotonic_ns(),
        }

    def observation(self, payload):
        with self.guard:
            return self._save_observation(payload)

    def _save_observation(self, payload):
        if payload.get("run_id") != self.state.get("run_id"):
            raise ValueError("Unknown recording run")
        if self.sealed:
            raise ValueError("Episode evidence is sealed")
        oid = identity(payload["observation_id"])
        directory = self.evidence / "observations"
        directory.mkdir(exist_ok=True)
        if (directory / (oid + ".json")).exists():
            raise ValueError("Observation already recorded")
        if not payload["images"] or not set(payload["images"]) <= {"wrist", "side"}:
            raise ValueError("Choose known observation image roles")
        for role, encoded in payload["images"].items():
            data = base64.b64decode(encoded, validate=True)
            if len(data) > 2 * 1024**2 or not data.startswith(b"\xff\xd8"):
                raise ValueError("Invalid model image")
            (directory / (oid + "-" + role + ".jpg")).write_bytes(data)
        durable_json(directory / (oid + ".json"), payload["metadata"])
        return {"saved": True, "available_robot_ns": time.monotonic_ns()}

    def event(self, payload):
        if not self.writer or payload.get("run_id") != self.state.get("run_id"):
            raise ValueError("Unknown recording run")
        encoded = json.dumps(
            {**payload["event"], "mirror_arrival_robot_ns": time.monotonic_ns()}, allow_nan=False
        )
        if len(encoded) > 12000:
            raise ValueError("Trace event too large")
        with self.guard:
            if self.sealed or payload.get("run_id") != self.state.get("run_id"):
                raise ValueError("Episode evidence is sealed or belongs to another run")
            with (self.evidence / "trace.jsonl").open("a") as stream:
                stream.write(encoded + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        return {"saved": True}

    def _record(self):
        previous = {}
        sequences = {}
        next_frame = time.monotonic()
        try:
            with (
                (self.evidence / "controller.jsonl").open("a") as actions,
                (self.evidence / "frames.jsonl").open("a") as frames,
            ):
                while not self.done.is_set():
                    if self.streams.error:
                        raise ValueError(self.streams.error)
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
                    images, times, cameras = {}, {}, {}
                    for role, index in zip(("wrist", "side"), self.indices):
                        if hasattr(self.lab.camera, "collection_frame"):
                            frame, timestamp, sequence = self.lab.camera.collection_frame(index)
                        else:
                            frame, timestamp = self.lab.camera.read_latest_timed(index)
                            sequence = None
                        stamp = int(timestamp * 1e9)
                        if stamp <= previous.get(index, 0) or captured - stamp > 500_000_000:
                            raise ValueError("Missing, repeated or stale " + role + " camera frame")
                        received = time.monotonic_ns()
                        cameras[role] = {
                            "frame_id": f"{role}-{stamp}",
                            "sensor_capture_ns": None,
                            "read_complete_robot_ns": stamp,
                            "available_robot_ns": received,
                            "source_sequence": sequence,
                            "unselected_reader_frames": (
                                sequence - sequences[index] - 1
                                if sequence is not None and index in sequences
                                else None
                            ),
                            "age_ns": received - stamp,
                        }
                        sequences[index] = sequence
                        times[role] = stamp
                        images["observation.images." + role] = frame[:, :, ::-1]
                        previous[index] = stamp
                    if min(times.values()) < self.stream_start_ns:
                        previous.clear()
                        self.done.wait(0.01)
                        continue
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
                                "cameras": cameras,
                                "pair_available_robot_ns": max(
                                    c["available_robot_ns"] for c in cameras.values()
                                ),
                                "pair_skew_ns": max(times.values()) - min(times.values()),
                                "missed_schedule_slots_before": self.state["dropped_samples"],
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
                        self.state["dropped_samples"] += (
                            int((now - next_frame) * self.writer.fps) + 1
                        )
                        next_frame = now + 1 / self.writer.fps
                if self.state["status"] == "failed":
                    raise ValueError(self.state["error"])
                while not self.samples.empty():
                    actions.write(json.dumps(self.samples.get_nowait()) + "\n")
                actions.flush()
                os.fsync(actions.fileno())
                os.fsync(frames.fileno())
        except Exception as error:
            self.state.update(
                status="failed", error=str(error), failure_robot_ns=time.monotonic_ns()
            )
            self.lab.stop(
                "Agent recording failed: " + str(error),
                expected_generation=self.generation,
                expected_mode="agent",
            )
        finally:
            # Flush successful writes even when a camera or queue failed.
            with (self.evidence / "controller.jsonl").open("a") as remaining:
                while not self.samples.empty():
                    remaining.write(json.dumps(self.samples.get_nowait()) + "\n")
            self.lab.camera.set_collection_sink(None)
            self.streams.finish()
            self.state["camera_stream_frames"] = self.streams.counts
            if self.streams.error:
                self.state.update(status="failed", error=self.streams.error)
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
        with self.guard:
            if not self.sealed:
                # Seal after trailing tool results/run_ended have been mirrored, before import.
                files = {}
                for path in sorted(self.root.rglob("*")):
                    if path.is_file() and not path.is_symlink():
                        with path.open("rb") as stream:
                            files[path.relative_to(self.root).as_posix()] = hashlib.file_digest(
                                stream, "sha256"
                            ).hexdigest()
                durable_json(self.evidence / "seal.json", {"schema_version": 2, "sha256": files})
                self.sealed = True
        return self.status()
