"""Local ACT admission and a heartbeat-bound policy controller.

Inference never owns a serial device. The companion's existing 40 Hz motor loop
applies bounded targets; Stop invalidates the generation before torque-off I/O.
"""

import base64
import hashlib
import json
import math
import os
import queue
import re
import stat
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

from hardware import JOINTS, HardwareError
from motion import control_limits
from policy_contract import (
    read_json,
    validate_policy_config,
    validate_processors,
    validate_smolvla_assets,
)

UNITS = ["degrees"] * 5 + ["percent"]
MAX_SPEED = 10.0
MAX_ACCELERATION = 20.0


def admit(root, ident):
    if not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", ident):
        raise ValueError("Choose an installed policy")
    folder = root / ident
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("Policy directory is unavailable")
    manifest = read_json(folder / "run_policy.json")
    kind = manifest.get("policy_type")
    offline = manifest.get("offline_only", False)
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or kind not in {"act", "smolvla"}
        or type(offline) is not bool
        or (
            not offline
            and (
                manifest.get("joint_names") != list(JOINTS)
                or manifest.get("state_units") != UNITS
                or manifest.get("action_units") != UNITS
            )
        )
        or manifest.get("checkpoint") != "pretrained_model"
        or (
            not offline and (type(manifest.get("fps")) is not int or not 1 <= manifest["fps"] <= 40)
        )
        or (
            offline
            and manifest.get("fps") is not None
            and (type(manifest["fps"]) is not int or not 1 <= manifest["fps"] <= 240)
        )
        or not re.fullmatch(
            r"observation\.(?:images\.[a-zA-Z0-9_.-]{1,64}|image[0-9]*)",
            manifest.get("camera_key", ""),
        )
    ):
        raise ValueError(
            "Policy needs a native adapter, six SO-101 joints, degrees/percent and 1–40 FPS"
        )
    checkpoint = folder / "pretrained_model"
    config = validate_policy_config(checkpoint)
    if config["type"] != kind:
        raise ValueError("Policy manifest and native model type disagree")
    validate_processors(checkpoint, kind)
    files, total = {}, 0
    paths = list(checkpoint.iterdir())
    if kind == "smolvla":
        assets = validate_smolvla_assets(checkpoint)
        paths = [path for path in paths if path.name != "assets"]
        paths.extend(assets.iterdir())
    for path in paths:
        if (
            path.is_symlink()
            or not path.is_file()
            or path.suffix not in {".json", ".safetensors", ".md", ".txt"}
            and path.name != "LICENSE"
        ):
            raise ValueError(
                "Only native JSON, safetensors and closed tokenizer data are supported"
            )
        info = path.stat()
        total += info.st_size
        if info.st_size <= 0 or total > 2 * 1024**3 or len(files) >= 128:
            raise ValueError("Policy checkpoint exceeds local admission limits")
        files[path.relative_to(checkpoint).as_posix()] = (
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
        )
    required = {
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    }
    if not required <= set(files):
        raise ValueError("Checkpoint needs model weights and its saved pre/postprocessors")
    images = {
        k: v
        for k, v in config["input_features"].items()
        if v.get("type") == "VISUAL" or k.startswith(("observation.images.", "observation.image"))
    }
    if manifest["camera_key"] not in images or set(config["input_features"]) != {
        "observation.state",
        *images,
    }:
        raise ValueError("Checkpoint camera features do not fit this arm")
    return {
        "id": ident,
        "name": str(manifest.get("name", ident))[:100],
        "policy_type": kind,
        "task_required": kind == "smolvla",
        "offline_only": offline,
        "state_units": manifest.get("state_units"),
        "action_units": manifest.get("action_units"),
        "fps": manifest["fps"],
        "camera_key": manifest["camera_key"],
        "camera_keys": list(images),
        "image_shape": images[manifest["camera_key"]]["shape"],
        "checkpoint": checkpoint,
        "files": files,
        "manifest": manifest,
    }


class InferenceProcess:
    def __init__(self, python, checkpoint, device, log, camera_key=None):
        self.queue = queue.Queue(maxsize=4)
        self.lock = threading.Lock()
        self.log = open(log, "a")
        os.chmod(log, 0o600)
        self.process = subprocess.Popen(
            [
                str(python),
                "-u",
                str(Path(__file__).with_name("policy_inference.py")),
                str(checkpoint),
                device,
                *([camera_key] if camera_key else []),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        try:
            kind = read_json(Path(checkpoint) / "config.json").get("type")
            reply = self.queue.get(timeout=180 if kind == "smolvla" else 90)
            if not isinstance(reply, dict) or reply.get("ready") is not True:
                raise ValueError("ACT runtime did not load; inspect the local policy runtime log")
            self.info = reply
        except BaseException:
            self.close()
            raise

    def _read(self):
        try:
            while True:
                line = self.process.stdout.readline(4 * 1024**2)
                if not line:
                    self.queue.put(None, timeout=1)
                    return
                self.queue.put(json.loads(line), timeout=1)
        except Exception:
            try:
                self.queue.put(None, timeout=1)
            except queue.Full:
                pass

    def request(self, payload, timeout=2):
        with self.lock:
            if self.process.poll() is not None:
                raise ValueError("Local policy inference process stopped")
            self.process.stdin.write(json.dumps(payload, allow_nan=False) + "\n")
            self.process.stdin.flush()
            try:
                reply = self.queue.get(timeout=timeout)
            except queue.Empty:
                self.close()
                raise ValueError("Local policy inference timed out; motion was stopped") from None
            if not isinstance(reply, dict):
                raise ValueError("Local policy inference failed; inspect its runtime log")
            if isinstance(reply.get("error"), str):
                raise ValueError(reply["error"])
            return reply

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        for stream in (self.process.stdin, self.process.stdout):
            if stream:
                stream.close()
        self.log.close()


class PolicyRunner:
    def __init__(self, lab, *, process_factory=InferenceProcess):
        self.lab = lab
        self.root = lab.data_dir / "policies"
        self.root.mkdir(mode=0o700, exist_ok=True)
        if self.root.is_symlink():
            raise ValueError("Policy store cannot use a symbolic link")
        self.process_factory = process_factory
        self.runtime = None
        self.model = None
        self.phase = "idle"
        self.error = None
        self.thread = None
        self.cancel = threading.Event()
        self.lock = threading.RLock()
        self.operator_deadline = 0.0
        self.run_deadline = 0.0
        self.predicted = None
        self.last_prediction = 0.0
        self.last_health = 0.0
        self.velocity = {j: 0.0 for j in JOINTS}
        self.commanded = {}
        self.steps = 0
        self.inference_ms = None
        self.started = None
        self.generation = None
        self.check = None
        self.preview = None
        self.task = ""
        self.operation = 0
        self.load_in_progress = False

    def runtime_config(self, policy_type=None):
        path = self.lab.data_dir / "policy-runtime.json"
        if not path.exists():
            raise ValueError("Install a compatible local policy runtime before loading")
        info = path.stat()
        if info.st_uid != os.getuid() or info.st_mode & 0o077 or not stat.S_ISREG(info.st_mode):
            raise ValueError("Policy runtime settings must be a private owned file")
        value = read_json(path, 8192)
        python = Path(value.get("python", ""))
        if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
            raise ValueError("Configured policy Python runtime is unavailable")
        routing = value.get("policy_devices", {})
        if (
            not isinstance(routing, dict)
            or set(routing) - {"act", "smolvla"}
            or any(device not in {"cpu", "mps", "cuda"} for device in routing.values())
        ):
            raise ValueError("Policy runtime device routing is invalid")
        device = routing.get(policy_type, value.get("device", "cpu"))
        if device not in {"cpu", "mps", "cuda"}:
            raise ValueError("Choose a supported local inference device")
        return python, device

    def catalog(self):
        policies = []
        for folder in sorted(self.root.iterdir())[:100]:
            if not folder.is_dir() or folder.is_symlink():
                continue
            try:
                item = admit(self.root, folder.name)
                policies.append(
                    {k: v for k, v in item.items() if k not in {"checkpoint", "files", "manifest"}}
                    | {
                        "compatible": True,
                        "message": "Native checkpoint admitted for local inference",
                    }
                )
            except (ValueError, OSError, KeyError, TypeError) as error:
                policies.append(
                    {
                        "id": folder.name,
                        "name": folder.name,
                        "compatible": False,
                        "message": str(error),
                    }
                )
        try:
            _, device = self.runtime_config()
            runtime = {
                "configured": True,
                "device": device,
                "message": "Load a policy to verify this runtime",
            }
        except (ValueError, OSError) as error:
            runtime = {"configured": False, "device": None, "message": str(error)}
        return {"policies": policies, "runtime": runtime, "state": self.snapshot()}

    def snapshot(self):
        with self.lock:
            return {
                "phase": self.phase,
                "policy_id": self.model["id"] if self.model else None,
                "name": self.model["name"] if self.model else None,
                "error": self.error,
                "fps": self.model["fps"] if self.model else None,
                "steps": self.steps,
                "preflight": self.check,
                "preview": self.preview,
                "policy_type": self.model["policy_type"] if self.model else None,
                "task_required": self.model["task_required"] if self.model else False,
                "offline_only": self.model["offline_only"] if self.model else False,
                "warm_inference_ms": self.runtime.info.get("warm_inference_ms")
                if self.runtime
                else None,
                "inference_ms": self.inference_ms,
                "started": self.started,
                "limits": {
                    "max_speed": MAX_SPEED,
                    "max_acceleration": MAX_ACCELERATION,
                    "max_run_seconds": 120,
                    "operator_heartbeat_seconds": 2,
                },
                "device": self.runtime.info.get("device") if self.runtime else None,
            }

    def load(self, payload):
        if set(payload) != {"policy_id"}:
            raise ValueError("Choose one installed policy")
        with self.lock:
            if (
                self.load_in_progress
                or self.phase in {"running", "loading", "checking", "installing"}
                or self.thread
                and self.thread.is_alive()
            ):
                raise ValueError("Stop the current policy operation before loading")
            self.phase, self.error = "loading", None
            self.load_in_progress = True
            self.check = None
            self.preview = None
            self.operation += 1
            operation = self.operation
        try:
            model = admit(self.root, payload["policy_id"])
            python, device = self.runtime_config(model["policy_type"])
            if self.runtime:
                self.runtime.close()
                self.runtime = None
            runtime = self.process_factory(
                python, model["checkpoint"], device, self.root / "runtime.log", model["camera_key"]
            )
            if runtime.info.get("camera_key") != model["camera_key"]:
                runtime.close()
                raise ValueError("Loaded policy camera differs from its admitted manifest")
            with self.lock:
                cancelled = operation != self.operation
                if not cancelled:
                    self.runtime, self.model, self.phase = runtime, model, "ready"
            if cancelled:
                runtime.close()
                raise ValueError("Policy loading was cancelled")
        except Exception as error:
            with self.lock:
                if operation == self.operation:
                    self.phase, self.error = "error", str(error)
            raise
        finally:
            with self.lock:
                self.load_in_progress = False
        return self.lab.status()

    def _task(self, payload):
        task = payload.get("task", "")
        if (
            not isinstance(task, str)
            or len(task) > 1000
            or (self.model and self.model["task_required"] and not task.strip())
        ):
            raise ValueError("This policy needs a task instruction of at most 1000 characters")
        return task

    def preview_prediction(self, payload):
        """One explicitly requested offline observation; never opens a motor bus."""
        if set(payload) - {"synthetic", "camera_index", "state", "jpeg", "task"}:
            raise ValueError("Choose one preview observation")
        task = self._task(payload)
        synthetic = payload.get("synthetic") is True
        state = [0.0] * 6 if synthetic else payload.get("state")
        if (
            not isinstance(state, list)
            or len(state) != 6
            or any(type(v) not in {int, float} or not math.isfinite(v) for v in state)
        ):
            raise ValueError("Preview needs six finite model state values")
        with self.lab.lock, self.lock, self.lab.lifecycle_lock:
            if (
                self.phase != "ready"
                or not self.runtime
                or not self.model
                or self.lab.mode != "idle"
                or self.lab.bus is not None
                or self.lab.pending_bus is not None
            ):
                raise ValueError("Load a policy and stop hardware control before previewing")
            self.phase, self.error = "checking", None
            self.operation += 1
            operation = self.operation
        try:
            if synthetic:
                # Explicit neutral test, not a claim of a measured robot observation.
                import io

                from PIL import Image

                buffer = io.BytesIO()
                Image.new("RGB", (256, 256)).save(buffer, "JPEG")
                jpeg = base64.b64encode(buffer.getvalue()).decode()
                source = "Synthetic zero-state and blank image"
            elif "camera_index" in payload:
                camera = payload["camera_index"]
                if type(camera) is not int or not 0 <= camera <= 63:
                    raise ValueError("Choose an available camera")
                jpeg = base64.b64encode(self.lab.camera.capture_timed(camera)["jpeg"]).decode()
                source = "Selected camera and supplied model state"
            else:
                jpeg = payload.get("jpeg")
                if not isinstance(jpeg, str) or len(jpeg) > 3 * 1024**2:
                    raise ValueError("Upload a bounded preview image")
                base64.b64decode(jpeg, validate=True)
                source = "Uploaded image and supplied model state"
            self.runtime.request({"reset": True})
            before = time.monotonic()
            result = self.runtime.request({"state": state, "jpeg": jpeg, "task": task}, timeout=60)
            action = result.get("action")
            if (
                not isinstance(action, list)
                or len(action) != 6
                or any(type(v) not in {int, float} or not math.isfinite(v) for v in action)
            ):
                raise ValueError("Policy preview did not produce six finite outputs")
            with self.lock:
                if operation != self.operation or self.lab.mode != "idle":
                    raise ValueError("Policy preview was cancelled")
                self.preview = {
                    "policy_id": self.model["id"],
                    "action": action,
                    "state": state,
                    "source": source,
                    "inference_ms": round((time.monotonic() - before) * 1000, 1),
                    "task_conditioned": self.model["task_required"],
                    "executed": False,
                }
        except Exception as error:
            with self.lock:
                if operation == self.operation:
                    self.error = str(error)
            raise
        finally:
            with self.lock:
                if operation == self.operation:
                    self.phase = "ready"
        return self.lab.status()

    def preflight(self, payload):
        """Measure actual camera + normalized native inference, without motor writes."""
        if (
            set(payload) - {"camera_index", "task"}
            or "camera_index" not in payload
            or type(payload["camera_index"]) is not int
            or not 0 <= payload["camera_index"] <= 63
        ):
            raise ValueError("Choose a camera for the policy check")
        task = self._task(payload)
        with self.lock:
            if self.phase != "ready" or not self.runtime or not self.model:
                raise ValueError("Load a policy before checking the camera and inference rate")
            if self.model["offline_only"]:
                raise ValueError("Offline inference only: physical control units are unverified")
            if self.lab.mode != "idle" or self.thread and self.thread.is_alive():
                raise ValueError("Stop motor control before running the policy check")
            self.phase, self.error, self.check = "checking", None, None
            self.operation += 1
            operation = self.operation
        camera, fps = payload["camera_index"], self.model["fps"]
        try:
            self.lab.camera.start_reader(camera)
            warm_deadline = time.monotonic() + 3
            while True:
                try:
                    self.lab.camera.latest_timed_jpeg(camera)
                    break
                except RuntimeError, TimeoutError:
                    if time.monotonic() >= warm_deadline:
                        raise ValueError("Policy camera did not provide a fresh frame") from None
                    time.sleep(0.02)
            self.runtime.request({"reset": True})
            samples, starts, stamps = [], [], []
            count = max(16, 2 * fps)
            due = time.monotonic()
            for _ in range(count):
                with self.lock:
                    if operation != self.operation or self.phase != "checking":
                        raise ValueError("Policy check was cancelled")
                started = time.monotonic()
                capture = self.lab.camera.latest_timed_jpeg(camera, max_age=max(0.15, 2 / fps))
                positions = self.lab.positions
                reply = self.runtime.request(
                    {
                        "state": [positions.get(j, 0.0) for j in JOINTS],
                        "jpeg": base64.b64encode(capture["jpeg"]).decode(),
                        "task": task,
                    },
                    timeout=max(0.5, 3 / fps),
                )
                action = reply.get("action")
                if (
                    not isinstance(action, list)
                    or len(action) != 6
                    or any(type(v) not in {float, int} or not math.isfinite(v) for v in action)
                ):
                    raise ValueError("Policy normalization did not produce six finite actions")
                samples.append(time.monotonic() - started)
                starts.append(started)
                stamps.append(capture["captured_monotonic"])
                due += 1 / fps
                time.sleep(max(0, due - time.monotonic()))
            self.runtime.request({"reset": True})
            actual = (count - 1) / (starts[-1] - starts[0])
            unique = len(set(stamps))
            mean = sum(samples) / count
            longest = max(samples)
            passed = (
                actual >= fps * 0.95
                and mean <= 0.85 / fps
                and longest <= max(0.5, 3 / fps)
                and unique >= count * 0.9
            )
            check = {
                "passed": passed,
                "camera_index": camera,
                "checked_at": time.monotonic(),
                "task_sha256": hashlib.sha256(task.encode()).hexdigest(),
                "fps": fps,
                "measured_fps": round(actual, 1),
                "mean_latency_ms": round(mean * 1000, 1),
                "max_latency_ms": round(longest * 1000, 1),
                "fresh_frames": unique,
                "samples": count,
                "message": "Camera and local ACT inference meet the trained frame rate"
                if passed
                else (
                    "This camera or local runtime cannot sustain the trained frame rate. "
                    "Use a faster inference device or matching camera; "
                    "keep the dataset/model FPS unchanged."
                ),
            }
            with self.lock:
                if operation != self.operation or self.phase != "checking":
                    raise ValueError("Policy check was cancelled")
                self.phase, self.check = "ready", check
        except Exception as error:
            with self.lock:
                if operation == self.operation:
                    self.phase, self.error = "ready", str(error)
            raise
        return self.lab.status()

    def start(self, payload):
        if set(payload) - {"task"} != {
            "confirm",
            "operator_present",
            "workspace_clear",
            "camera_index",
            "duration_seconds",
            "expected_generation",
        }:
            raise ValueError("Review the policy, camera, duration and attended workspace")
        if (
            payload["confirm"] != "RUN POLICY"
            or payload["operator_present"] is not True
            or payload["workspace_clear"] is not True
        ):
            raise ValueError("An attending operator and clear workspace are required")
        task = self._task(payload)
        duration, camera = payload["duration_seconds"], payload["camera_index"]
        if (
            type(duration) is not int
            or not 1 <= duration <= 120
            or type(camera) is not int
            or not 0 <= camera <= 63
        ):
            raise ValueError("Run duration must be 1–120 seconds with an available camera")
        with self.lab.lock, self.lock, self.lab.lifecycle_lock:
            if self.phase != "ready" or not self.runtime or not self.model:
                raise ValueError("Load and verify a policy first")
            if self.model["offline_only"]:
                raise ValueError("This policy is installed for offline inference only")
            if self.thread and self.thread.is_alive():
                raise ValueError("Previous inference is still stopping")
            if (
                type(payload["expected_generation"]) is not int
                or payload["expected_generation"] != self.lab.generation
                or self.lab.mode != "manual"
                or set(self.lab.calibration) != set(JOINTS)
                or not self.lab.torque
                or self.lab.middle_motion
                or self.lab.rest_motion
                or (self.lab.writer and not self.lab.writer.finalized)
            ):
                raise ValueError(
                    "Start full-arm joint control and close recording before running a policy"
                )
            if time.monotonic() - self.lab.heartbeat > 2:
                raise ValueError("The manual operator heartbeat expired")
            if (
                not self.check
                or not self.check["passed"]
                or self.check["camera_index"] != camera
                or time.monotonic() - self.check["checked_at"] > 60
                or (
                    self.model["task_required"]
                    and self.check.get("task_sha256") != hashlib.sha256(task.encode()).hexdigest()
                )
            ):
                raise ValueError("Check this policy and selected camera again before starting")
            current = admit(self.root, self.model["id"])
            if (
                current["files"] != self.model["files"]
                or current["manifest"] != self.model["manifest"]
            ):
                raise ValueError("Policy files changed; load the policy again")
            self.cancel = threading.Event()
            self.predicted = None
            self.commanded = dict(self.lab.positions)
            self.velocity = {j: 0.0 for j in JOINTS}
            self.last_apply = time.monotonic()
            self.steps = 0
            self.generation = self.lab.generation
            self.task = task
            self.started = time.monotonic()
            self.last_prediction = self.last_health = self.started
            self.operator_deadline = self.started + 2
            self.run_deadline = self.started + duration
            self.phase, self.error = "running", None
            self.lab.mode = "policy"
            self.thread = threading.Thread(
                target=self._run,
                args=(camera, duration, self.generation, self.cancel),
                name="robot-lab-policy",
                daemon=True,
            )
            self.thread.start()
        return self.lab.status()

    def heartbeat(self, payload):
        if (
            set(payload) != {"expected_generation"}
            or type(payload["expected_generation"]) is not int
        ):
            raise ValueError("Invalid policy operator heartbeat")
        with self.lock, self.lab.lifecycle_lock:
            if (
                self.phase != "running"
                or self.lab.mode != "policy"
                or payload["expected_generation"] != self.generation
                or self.generation != self.lab.generation
            ):
                raise ValueError("Policy controller ownership changed")
            self.operator_deadline = time.monotonic() + 2
        return self.lab.status()

    def revoke(self, reason=None):
        # Never wait for the inference process or its camera on the Stop path.
        self.cancel.set()
        with self.lock:
            self.operation += 1
            if self.phase == "loading":
                self.phase = "idle"
            if self.phase == "installing":
                self.phase = "ready" if self.runtime and self.model else "idle"
            if self.phase == "checking":
                self.phase = "ready"
                self.check = None
            if self.phase == "running":
                self.phase = "ready" if self.runtime else "idle"
            if reason:
                self.error = reason
            self.predicted = None

    def expired(self):
        return self.phase == "running" and time.monotonic() > min(
            self.operator_deadline, self.run_deadline
        )

    def apply(self, measured, now):
        with self.lock:
            if (
                self.phase != "running"
                or self.generation != self.lab.generation
                or self.cancel.is_set()
            ):
                raise HardwareError("Policy controller authority was revoked")
            self.last_health = now
            if self.expired():
                raise HardwareError("Policy operator heartbeat or run duration expired")
            if now - self.last_prediction > max(0.5, 3 / self.model["fps"]):
                raise HardwareError("Policy predictions fell behind the trained frame rate")
            if self.predicted is None:
                return dict(measured)
            dt = min(0.05, max(0.001, now - getattr(self, "last_apply", now - 0.025)))
            self.last_apply = now
            ranges = control_limits(self.lab.calibration)
            goals = {}
            for j in JOINTS:
                if abs(measured[j] - self.commanded[j]) > 15:
                    raise HardwareError("Policy tracking error exceeded the motion envelope")
                target = self.predicted[j]
                if not ranges[j]["min"] <= target <= ranges[j]["max"]:
                    raise HardwareError("Policy target is outside calibrated joint limits")
                delta = target - self.commanded[j]
                # A proportional velocity goal approaches each target without
                # snapping the final step, which would violate acceleration limits.
                desired = min(MAX_SPEED, max(-MAX_SPEED, delta * 2.0))
                velocity = min(
                    self.velocity[j] + MAX_ACCELERATION * dt,
                    max(self.velocity[j] - MAX_ACCELERATION * dt, desired),
                )
                step = velocity * dt
                goals[j] = self.commanded[j] + step
                if not ranges[j]["min"] <= goals[j] <= ranges[j]["max"]:
                    raise HardwareError("Policy motion envelope reached a calibrated joint limit")
                self.velocity[j] = velocity
            self.commanded = goals
            return dict(goals)

    def _run(self, camera, duration, generation, cancel):
        reason = "Policy run completed; torque disabled"
        try:
            self.runtime.request({"reset": True})
            deadline, due = time.monotonic() + duration, time.monotonic()
            camera_stamps = deque(maxlen=max(16, 2 * self.model["fps"]))
            while not cancel.is_set() and time.monotonic() < deadline:
                if generation != self.lab.generation or self.lab.mode != "policy":
                    return
                if time.monotonic() - due > max(0.15, 2 / self.model["fps"]):
                    raise ValueError("Policy loop cannot maintain the trained frame rate")
                capture = self.lab.camera.latest_timed_jpeg(
                    camera, max_age=max(0.15, 2 / self.model["fps"])
                )
                camera_stamps.append(capture["captured_monotonic"])
                if (
                    len(camera_stamps) == camera_stamps.maxlen
                    and len(set(camera_stamps)) < len(camera_stamps) * 0.9
                ):
                    raise ValueError("Policy camera fell below the trained frame rate")
                positions = dict(self.lab.positions)
                if set(positions) != set(JOINTS) or time.monotonic() - self.last_health > 0.15:
                    raise ValueError("Fresh full-arm encoder observations are unavailable")
                before = time.monotonic()
                result = self.runtime.request(
                    {
                        "state": [positions[j] for j in JOINTS],
                        "jpeg": base64.b64encode(capture["jpeg"]).decode(),
                        "task": self.task,
                    },
                    timeout=max(0.5, 3 / self.model["fps"]),
                )
                action = result.get("action")
                if (
                    not isinstance(action, list)
                    or len(action) != 6
                    or any(type(v) not in {float, int} or not math.isfinite(v) for v in action)
                ):
                    raise ValueError("Policy output is incompatible or non-finite")
                with self.lock:
                    if (
                        cancel.is_set()
                        or generation != self.lab.generation
                        or self.lab.mode != "policy"
                    ):
                        return
                    self.predicted = dict(zip(JOINTS, action, strict=True))
                    self.last_prediction = time.monotonic()
                    self.inference_ms = round((self.last_prediction - before) * 1000, 1)
                    self.steps += 1
                due += 1 / self.model["fps"]
                cancel.wait(max(0, due - time.monotonic()))
        except Exception as error:
            reason = "Policy stopped: " + str(error)
        finally:
            if (
                not cancel.is_set()
                and generation == self.lab.generation
                and self.lab.mode == "policy"
            ):
                try:
                    self.lab.stop(reason, expected_generation=generation, expected_mode="policy")
                except ValueError:
                    pass

    def close(self):
        self.revoke()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=3)
        if self.runtime:
            self.runtime.close()
            self.runtime = None
