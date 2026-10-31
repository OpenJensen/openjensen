"""Local ACT admission and a heartbeat-bound policy controller.

Inference never owns a serial device. The companion's existing 40 Hz motor loop
applies bounded targets; Stop invalidates the generation before torque-off I/O.
"""

import base64
import json
import math
import os
import queue
import re
import stat
import subprocess
import threading
import time
from pathlib import Path

from hardware import JOINTS, HardwareError
from motion import control_limits

UNITS = ["degrees"] * 5 + ["percent"]
MAX_SPEED = 10.0
MAX_ACCELERATION = 20.0


def read_json(path, maximum=1024 * 1024):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("Policy metadata must be a bounded regular file")
    return json.loads(path.read_text())


def admit(root, ident):
    if not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", ident):
        raise ValueError("Choose an installed policy")
    folder = root / ident
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("Policy directory is unavailable")
    manifest = read_json(folder / "run_policy.json")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("policy_type") != "act"
        or manifest.get("joint_names") != list(JOINTS)
        or manifest.get("state_units") != UNITS
        or manifest.get("action_units") != UNITS
        or manifest.get("checkpoint") != "pretrained_model"
        or type(manifest.get("fps")) is not int
        or not 1 <= manifest["fps"] <= 40
        or not re.fullmatch(
            r"observation\.images\.[a-zA-Z0-9_-]{1,64}", manifest.get("camera_key", "")
        )
    ):
        raise ValueError(
            "Policy needs ACT, six ordered SO-101 joints, degrees/percent and 1–40 FPS"
        )
    checkpoint = folder / "pretrained_model"
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("Native ACT checkpoint is unavailable")
    files, total = {}, 0
    for path in checkpoint.iterdir():
        if (
            path.is_symlink()
            or not path.is_file()
            or path.suffix not in {".json", ".safetensors", ".md", ".txt"}
            and path.name != "LICENSE"
        ):
            raise ValueError("Only native JSON and safetensors policy files are supported")
        info = path.stat()
        total += info.st_size
        if info.st_size <= 0 or total > 2 * 1024**3 or len(files) >= 128:
            raise ValueError("Policy checkpoint exceeds local admission limits")
        files[path.name] = (info.st_ino, info.st_size, info.st_mtime_ns)
    required = {
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    }
    if not required <= set(files):
        raise ValueError("Checkpoint needs model weights and its saved pre/postprocessors")
    config = read_json(checkpoint / "config.json")
    images = {
        k: v
        for k, v in config.get("input_features", {}).items()
        if k.startswith("observation.images.")
    }
    if (
        config.get("type") != "act"
        or config.get("input_features", {}).get("observation.state", {}).get("shape") != [6]
        or config.get("output_features", {}).get("action", {}).get("shape") != [6]
        or set(images) != {manifest["camera_key"]}
        or set(config.get("input_features", {})) != {"observation.state", manifest["camera_key"]}
    ):
        raise ValueError("Checkpoint state, action or camera features do not fit this arm")
    shape = images[manifest["camera_key"]].get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or shape[0] != 3
        or any(type(n) is not int or not 1 <= n <= 4096 for n in shape)
    ):
        raise ValueError("Policy camera shape is invalid")
    return {
        "id": ident,
        "name": str(manifest.get("name", ident))[:100],
        "policy_type": "act",
        "fps": manifest["fps"],
        "camera_key": manifest["camera_key"],
        "image_shape": shape,
        "checkpoint": checkpoint,
        "files": files,
        "manifest": manifest,
    }


class InferenceProcess:
    def __init__(self, python, checkpoint, device, log):
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
            reply = self.queue.get(timeout=90)
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
        self.operation = 0

    def runtime_config(self):
        path = self.lab.data_dir / "policy-runtime.json"
        if not path.exists():
            raise ValueError("Install a compatible local ACT runtime before loading a policy")
        info = path.stat()
        if info.st_uid != os.getuid() or info.st_mode & 0o077 or not stat.S_ISREG(info.st_mode):
            raise ValueError("Policy runtime settings must be a private owned file")
        value = read_json(path, 8192)
        python = Path(value.get("python", ""))
        if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
            raise ValueError("Configured policy Python runtime is unavailable")
        device = value.get("device", "cpu")
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
                    | {"compatible": True, "message": "Native ACT checkpoint fits this arm"}
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
                self.phase in {"running", "loading", "checking"}
                or self.thread
                and self.thread.is_alive()
            ):
                raise ValueError("Stop the current policy operation before loading")
            self.phase, self.error = "loading", None
            self.check = None
            self.operation += 1
            operation = self.operation
        try:
            model = admit(self.root, payload["policy_id"])
            python, device = self.runtime_config()
            if self.runtime:
                self.runtime.close()
                self.runtime = None
            runtime = self.process_factory(
                python, model["checkpoint"], device, self.root / "runtime.log"
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
        return self.lab.status()

    def preflight(self, payload):
        """Measure actual camera + normalized native inference, without motor writes."""
        if (
            set(payload) != {"camera_index"}
            or type(payload["camera_index"]) is not int
            or not 0 <= payload["camera_index"] <= 63
        ):
            raise ValueError("Choose a camera for the policy check")
        with self.lock:
            if self.phase != "ready" or not self.runtime or not self.model:
                raise ValueError("Load a policy before checking the camera and inference rate")
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
        if set(payload) != {
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
            self.started = time.monotonic()
            self.last_prediction = self.last_health = self.started
            self.operator_deadline = self.started + 2
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
            if self.phase == "checking":
                self.phase = "ready"
                self.check = None
            if self.phase == "running":
                self.phase = "ready" if self.runtime else "idle"
            if reason:
                self.error = reason
            self.predicted = None

    def expired(self):
        return self.phase == "running" and time.monotonic() > self.operator_deadline

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
                raise HardwareError("Policy operator heartbeat expired")
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
            while not cancel.is_set() and time.monotonic() < deadline:
                if generation != self.lab.generation or self.lab.mode != "policy":
                    return
                if time.monotonic() - due > max(0.15, 2 / self.model["fps"]):
                    raise ValueError("Policy loop cannot maintain the trained frame rate")
                capture = self.lab.camera.latest_timed_jpeg(
                    camera, max_age=max(0.15, 2 / self.model["fps"])
                )
                positions = dict(self.lab.positions)
                if set(positions) != set(JOINTS) or time.monotonic() - self.last_health > 0.15:
                    raise ValueError("Fresh full-arm encoder observations are unavailable")
                before = time.monotonic()
                result = self.runtime.request(
                    {
                        "state": [positions[j] for j in JOINTS],
                        "jpeg": base64.b64encode(capture["jpeg"]).decode(),
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
