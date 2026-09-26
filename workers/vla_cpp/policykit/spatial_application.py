"""Application envelope adapter for the pinned SmolVLA LIBERO Spatial protocol.

GPU execution is opt-in through an operator-prepared Linux runtime. CPU tests
exercise protocol/packaging guards, never claim a successful robot episode.
"""

from __future__ import annotations

import collections
import importlib.metadata
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

from .acceptance import artifact_contract, memory_coverage
from .cuda_bench import percentile
from .provenance import runtime_identity
from .spatial_protocol import ASSETS_REVISION, digest, episode_ids, protocol, verify_assets
from .worker import atomic_json, sha256


def target_identity():
    if platform.system() != "Linux":
        raise ValueError("Spatial evaluation requires the prepared Linux CUDA runtime")
    value = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                "0",
                "--query-gpu=uuid,name,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        )
        .strip()
        .splitlines()
    )
    if len(value) != 1:
        raise ValueError("Exactly one selected GPU identity is required")
    fields = [x.strip() for x in value[0].split(",")]
    if len(fields) != 3 or not fields[0].startswith("GPU-"):
        raise ValueError("GPU identity is unavailable")
    return dict(zip(("gpu_uuid", "name", "driver_version"), fields))


def simulator_identity(lane):
    import yaml

    lane = Path(lane)
    config = yaml.safe_load((lane / "config.yaml").read_text())
    assets = json.loads((lane / "assets.json").read_text())
    if (
        assets.get("revision") != ASSETS_REVISION
        or assets.get("repository") != "lerobot/libero-assets"
        or Path(assets.get("path", "")).resolve() != Path(config["assets"]).resolve()
    ):
        raise ValueError("Spatial requires the pinned prepared LIBERO assets")
    files = {}
    for name in ("bddl_files", "init_states", "assets"):
        root = Path(config[name])
        if not root.is_absolute() or not root.is_dir():
            raise ValueError("Prepared simulator path is unavailable: " + name)
        inventory = {
            p.relative_to(root).as_posix(): sha256(p)
            for p in sorted(root.rglob("*"))
            if p.is_file()
        }
        if not inventory:
            raise ValueError("Prepared simulator inventory is empty: " + name)
        files[name] = digest(inventory)
    return {
        "asset_revision": ASSETS_REVISION,
        "inventories": files,
        "lerobot": importlib.metadata.version("lerobot"),
        "hf_libero": importlib.metadata.version("hf-libero"),
        "mujoco": importlib.metadata.version("mujoco"),
    }


def backend_name(manifest, requested):
    if requested == "native-bf16":
        if manifest["metadata"].get("precision") != "float":
            raise ValueError("Native reference must use the floating source")
        return requested
    if requested != "cpp":
        raise ValueError("Unsupported Spatial evaluation backend")
    precision = manifest["metadata"].get("precision")
    if precision == "float":
        return "cpp-bf16"
    if not isinstance(precision, dict) or precision.get("language") not in {"Q8_0", "Q4_0"}:
        raise ValueError("Unsupported Spatial precision")
    return "cpp-" + precision["language"] + ("-vision" if precision.get("vision") else "")


def stage_name(episode):
    return f"rollout-{episode['task_id']}-{episode['init_state_id']}"


def parse_memory(text, pids, gpu_uuid):
    values = []
    for line in text.splitlines():
        fields = [x.strip() for x in line.split(",")]
        if len(fields) != 3 or fields[0] != gpu_uuid:
            continue
        try:
            pid, memory = int(fields[1]), float(fields[2])
        except ValueError:
            continue
        if pid in pids:
            if not math.isfinite(memory) or memory <= 0:
                return None
            values.append(memory)
    return sum(values) if values else None


def measure_child(command, output, env, stages, target, timeout):
    """Sample only the child process tree, including model load and each rollout."""
    import psutil

    measurements = {
        stage: {
            "gpu_samples": 0,
            "sampled_peak_device_used_mib": None,
            "memory_scope": "GPU process-tree sum sampled every >=100ms; "
            "includes startup; may miss brief peaks",
        }
        for stage in stages
    }
    process = None
    cancelled = False
    cleaning = False
    watched_signals = (signal.SIGTERM, signal.SIGINT)
    previous_handlers = {sig: signal.getsignal(sig) for sig in watched_signals}

    def cancellation(signum, frame):
        nonlocal cancelled
        cancelled = True
        # Popen may receive a signal between creating the child and returning it.
        # Record it until the child handle is assigned, then enter owned cleanup.
        if process is not None and not cleaning:
            raise InterruptedError("Spatial evaluation cancelled")

    started = time.monotonic()
    with (output / "worker.log").open("w") as log:
        try:
            for sig in watched_signals:
                signal.signal(sig, cancellation)
            process = subprocess.Popen(
                command,
                cwd=output,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            if cancelled:
                raise InterruptedError("Spatial evaluation cancelled during process launch")
            while process.poll() is None:
                if time.monotonic() - started > timeout:
                    raise TimeoutError("Spatial evaluation exceeded its wall-clock budget")
                if log.tell() > 64 * 1024 * 1024:
                    raise ValueError("Spatial worker diagnostic output exceeded 64 MiB")
                stage = "reload"
                if (output / "stage.json").is_file():
                    stage = json.loads((output / "stage.json").read_text())["stage"]
                if stage not in measurements:
                    raise ValueError("Unknown evaluation measurement stage")
                try:
                    root = psutil.Process(process.pid)
                    pids = {root.pid, *(child.pid for child in root.children(recursive=True))}
                except psutil.NoSuchProcess:
                    pids = set()
                sample = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                memory = parse_memory(sample.stdout, pids, target["gpu_uuid"])
                if sample.returncode == 0 and memory is not None:
                    measurement = measurements[stage]
                    measurement["gpu_samples"] += 1
                    measurement["sampled_peak_device_used_mib"] = max(
                        measurement["sampled_peak_device_used_mib"] or 0, memory
                    )
                    atomic_json(output / "sampled.json", {"stage": stage})
                time.sleep(0.1)
            if process.returncode:
                raise ValueError("Spatial child failed; inspect its worker.log")
        finally:
            cleaning = True
            try:
                if process is not None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            finally:
                for sig, handler in previous_handlers.items():
                    signal.signal(sig, handler)
    return measurements


def run_package_process(command, *, timeout, **kwargs):
    """Allow the nested Spatial supervisor to clean up before forced termination."""
    import psutil

    process = subprocess.Popen(command, **kwargs)
    try:
        process.wait(timeout=timeout)
    except BaseException:
        try:
            descendants = psutil.Process(process.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            descendants = []
        try:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        finally:
            # Fallback for an unresponsive supervisor; psutil's Process object
            # checks PID reuse before killing a previously observed descendant.
            for child in reversed(descendants):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(descendants, timeout=2)
        raise
    return subprocess.CompletedProcess(command, process.returncode)


def offline_environment(output, lane):
    cache = Path(output) / "empty-hf-cache"
    cache.mkdir(exist_ok=False)
    return {
        **os.environ,
        "HF_HOME": str(cache),
        "HF_HUB_CACHE": str(cache / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(cache / "hub"),
        "TRANSFORMERS_CACHE": str(cache / "transformers"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "LIBERO_CONFIG_PATH": str(Path(lane).resolve()),
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "OMP_NUM_THREADS": "4",
        "VLA_N_THREADS": "4",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }


def spatial_contract(source, manifest):
    """Check the 6-to-8 state correction and exact serialized normalization."""
    import gguf
    import numpy as np
    from safetensors.numpy import load_file

    contract = artifact_contract(source, manifest)
    required = {"chunk_size": 50, "max_action_dim": 32, "real_action_dim": 7, "image_size": 512}
    if any(contract[key] != value for key, value in required.items()):
        raise ValueError("GGUF and Spatial inference protocol disagree")
    reader = gguf.GGUFReader(source / "model.gguf")
    for name, expected in (("real_state_dim", 8), ("max_state_dim", 32), ("num_steps", 10)):
        field = reader.fields.get("smolvla." + name)
        if field is None or field.contents() != expected:
            raise ValueError("GGUF lacks the verified Spatial state/denoising contract: " + name)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    for filename, prefix, feature, dimension in (
        (
            "policy_preprocessor_step_5_normalizer_processor.safetensors",
            "state",
            "observation.state",
            8,
        ),
        ("policy_postprocessor_step_0_unnormalizer_processor.safetensors", "action", "action", 7),
    ):
        saved = load_file(source / "policy" / filename)
        for suffix in ("mean", "std"):
            value = saved.get(feature + "." + suffix)
            tensor = tensors.get(prefix + "_" + suffix)
            if value is None or tensor is None:
                raise ValueError("Missing serialized normalization statistics")
            native = np.asarray(value, dtype=np.float32).reshape(-1)
            converted = gguf.quants.dequantize(tensor.data, tensor.tensor_type).reshape(-1)
            if (
                native.shape != (dimension,)
                or converted.shape != (dimension,)
                or not np.isfinite(native).all()
                or not np.array_equal(native, converted)
            ):
                raise ValueError("GGUF normalization differs from the native processor")
    return contract


def fixture_actions(result, assets):
    """Reject partial/NaN fixtures before they can become parity evidence."""
    rows = result.get("fixture_actions")
    expected = [assets["files"][name] for name in assets["fixtures"]]
    if (
        result.get("fixture_actions_deterministic") is not True
        or not isinstance(rows, list)
        or len(rows) != len(expected)
        or len(set(expected)) != len(expected)
    ):
        raise ValueError("Fixed-fixture evidence is incomplete or nondeterministic")
    for row, expected_hash in zip(rows, expected):
        if not isinstance(row, dict) or row.get("fixture_sha256") != expected_hash:
            raise ValueError("Fixed-fixture identity changed")
        actions = row.get("actions")
        if (
            not isinstance(actions, list)
            or len(actions) != 50
            or any(
                not isinstance(action, list)
                or len(action) != 7
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in action)
                for action in actions
            )
        ):
            raise ValueError("Fixed-fixture actions must be finite complete 50x7 chunks")
    return rows


def inference_identity(runtime):
    """Actual child environment plus ELF libraries loaded by the Python ML process."""
    libraries = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[-1].startswith("/"):
            continue
        path = Path(fields[-1])
        if ".so" in path.name:
            if not path.is_file():
                raise ValueError("Loaded inference library is unavailable for verification")
            resolved = str(path.resolve())
            if resolved not in libraries:
                libraries[resolved] = sha256(path)
    if not libraries:
        raise ValueError("No actual inference native-library inventory is available")
    return {
        **runtime_identity({**runtime, "simulator_lane": None}),
        "loaded_python_native_libraries": libraries,
    }


def evaluate_policy(job):
    from .application import verify

    source, output = Path(job["artifact"]["path"]), Path(job["output_dir"])
    manifest = verify(source)
    runtime, evaluation = job["runtime"], job["parameters"]["evaluation"]
    if evaluation.get("mode") != "libero" or runtime["device"] != "cuda":
        raise ValueError("Spatial evaluation requires explicit LIBERO mode and CUDA")
    if manifest["metadata"].get("task") != "libero_spatial":
        raise ValueError("This policy does not declare Spatial compatibility")
    requested = job.get("evaluation_backend", "cpp")
    backend = backend_name(manifest, requested)
    assets = verify_assets(source, require_reference=requested == "native-bf16")
    spatial_contract(source, manifest)
    simulator = simulator_identity(runtime["simulator_lane"])
    spec, spec_sha = protocol(evaluation, assets, simulator, job.get("final", False))
    target = target_identity()
    identity = {
        **runtime_identity({**runtime, "simulator_lane": None}),
        "backend": requested,
        "spatial_simulator": simulator,
    }
    pending = output / "spatial"
    pending.mkdir(exist_ok=False)
    child_request = {
        "job_id": job["job_id"],
        "source": str(source.resolve()),
        "manifest_sha256": sha256(source / "manifest.json"),
        "backend": backend,
        "runtime": runtime,
        "protocol": spec,
        "protocol_sha256": spec_sha,
        "evaluation": evaluation,
    }
    atomic_json(pending / "request.json", child_request)
    bootstrap = (
        "import sys; sys.path.append("
        + repr(str(Path(__file__).resolve().parents[1]))
        + ("); from policykit.spatial_application import child_main; child_main()")
    )
    command = [sys.executable, "-c", bootstrap, str((pending / "request.json").resolve())]
    stages = ["reload", "timing", *(stage_name(row) for row in spec["episodes"])]
    measured = measure_child(
        command,
        pending,
        offline_environment(pending, runtime["simulator_lane"]),
        stages,
        target,
        job["parameters"].get("timeout_seconds", 7200),
    )
    result_file = pending / "result.json"
    if not result_file.is_file() or result_file.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Spatial worker did not return a bounded result")
    result = json.loads(result_file.read_text())
    if (
        result.get("protocol_sha256") != spec_sha
        or result.get("manifest_sha256") != child_request["manifest_sha256"]
    ):
        raise ValueError("Spatial child did not verify the requested package and protocol")
    if (
        type(result.get("process_id")) is not int
        or result["process_id"] <= 0
        or result["process_id"] == os.getpid()
        or not isinstance(result.get("inference_runtime"), dict)
        or not result["inference_runtime"]
    ):
        raise ValueError("Spatial result lacks independent inference process provenance")
    actions = fixture_actions(result, assets)
    episodes = result.get("episodes", [])
    identities = [
        {key: row.get(key) for key in ("task_id", "init_state_id", "seed", "noise_seed")}
        for row in episodes
    ]
    if identities != spec["episodes"] or any(not complete_episode(row) for row in episodes):
        raise ValueError("Spatial child did not complete the exact requested episodes")
    samples = result.get("samples_ms", [])
    if len(samples) != evaluation["repetitions"] * len(assets["fixtures"]) or any(
        type(value) not in (int, float) or not math.isfinite(value) or value <= 0
        for value in samples
    ):
        raise ValueError("Spatial timing samples are incomplete or nonfinite")
    if (
        verify(source) != manifest
        or target_identity() != target
        or {
            **runtime_identity({**runtime, "simulator_lane": None}),
            "backend": requested,
            "spatial_simulator": simulator_identity(runtime["simulator_lane"]),
        }
        != identity
    ):
        raise ValueError("Spatial artifact, simulator, target or runtime changed during evaluation")
    report = {
        "scope": "libero_final" if job.get("final") else "libero_development",
        "backend": requested,
        "backend_configuration": backend,
        "runtime": {"adapter": identity, "inference": result["inference_runtime"]},
        "target_identity": target,
        "protocol": spec,
        "protocol_sha256": spec_sha,
        "episodes": episodes,
        "requested_episodes": len(episodes),
        "complete_episodes": len(episodes),
        "success_rate": sum(row["task_success"] for row in episodes) / len(episodes),
        "fixture_actions": actions,
        "fixture_actions_deterministic": result["fixture_actions_deterministic"],
        "source_native_model_sha256": assets["checkpoint"]["weights_sha256"],
        "samples_ms": samples,
        "p50_ms": percentile(samples, 50),
        "p95_ms": percentile(samples, 95),
        "measurements": measured,
        **memory_coverage(measured, "cuda"),
        "worker_process_id": os.getpid(),
        "inference_process_id": result["process_id"],
        "artifact_path": str(source.resolve()),
        "artifact_manifest_sha256": sha256(source / "manifest.json"),
        "model_sha256": sha256(source / "model.gguf"),
        "fresh_reload_verified": True,
        "offline_package_assets": True,
        "training_overlap": "not audited",
        "hardware_acceptance_scope": "This run only; sampled memory can miss transients",
    }
    if requested == "native-bf16":
        report["native_model_sha256"] = sha256(source / "policy/model.safetensors")
    return {"report": report}


def complete_episode(row):
    return (
        row.get("status") == "episode_complete"
        and type(row.get("task_success")) is bool
        and type(row.get("benchmark_horizon")) is int
        and row["benchmark_horizon"] == 280
        and type(row.get("steps")) is int
        and 1 <= row["steps"] <= 280
        and (row["task_success"] or row["steps"] == 280)
    )


def paired(report, reference, evaluation, final=True):
    if (
        not reference
        or report.get("protocol_sha256") != reference.get("protocol_sha256")
        or (
            not report.get("target_identity")
            or report.get("target_identity") != reference.get("target_identity")
        )
    ):
        return False
    expected = episode_ids(evaluation, final)
    for value in (report, reference):
        if not isinstance(value.get("protocol"), dict) or digest(value["protocol"]) != value.get(
            "protocol_sha256"
        ):
            return False
        actual = [
            {key: row.get(key) for key in ("task_id", "init_state_id", "seed", "noise_seed")}
            for row in value.get("episodes", [])
        ]
        episodes = value.get("episodes", [])
        if (
            actual != expected
            or value.get("complete_episodes") != len(expected)
            or value.get("requested_episodes") != len(expected)
            or any(not complete_episode(row) for row in episodes)
            or type(value.get("success_rate")) not in (int, float)
            or not math.isfinite(value["success_rate"])
            or value["success_rate"] != sum(row["task_success"] for row in episodes) / len(expected)
            or value.get("memory_coverage", {}).get("complete") is not True
        ):
            return False
    return True


def child_main():
    request_path = Path(sys.argv[1])
    request = json.loads(request_path.read_text())
    output = request_path.parent
    source = Path(request["source"])
    from .application import verify

    manifest = verify(source)
    if sha256(source / "manifest.json") != request["manifest_sha256"]:
        raise ValueError("Package changed before fresh process loaded it")
    assets = verify_assets(source, require_reference=request["backend"] == "native-bf16")
    if digest(request["protocol"]) != request["protocol_sha256"]:
        raise ValueError("Invalid frozen Spatial protocol")
    import torch
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.envs.utils import close_envs
    from lerobot.scripts.lerobot_eval import rollout

    from .spatial_runtime import Runtime, load_fixture, noise

    if importlib.metadata.version("lerobot") != "0.4.4":
        raise ValueError("Spatial adapter requires reviewed LeRobot 0.4.4")
    torch.set_num_threads(4)

    def begin(stage):
        atomic_json(output / "stage.json", {"stage": stage})

    def sampled(stage):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            path = output / "sampled.json"
            if path.exists() and json.loads(path.read_text()).get("stage") == stage:
                return
            time.sleep(0.05)
        raise ValueError("No positive process-tree GPU sample for " + stage)

    begin("reload")
    runtime = Runtime(
        request["backend"],
        Path(request["runtime"]["vendor"]).parent,
        output,
        policy_path=source / "policy",
        backbone_path=source / "backbone",
        model_path=source / "model.gguf",
        build=request["runtime"]["build"],
        vendor=request["runtime"]["vendor"],
    )
    try:
        sampled("reload")
        begin("timing")
        samples = []
        fixture_actions = []
        for name in assets["fixtures"]:
            raw, initial_noise = load_fixture(source / name)
            for _ in range(request["evaluation"]["warmups"]):
                runtime.predict(raw, initial_noise)
            for _ in range(request["evaluation"]["repetitions"]):
                actions, milliseconds = runtime.predict(raw, initial_noise)
                samples.append(milliseconds)
            repeated, _ = runtime.predict(raw, initial_noise)
            if not torch.equal(actions, repeated):
                raise ValueError("Repeated fixed-fixture inference is not deterministic")
            fixture_actions.append(
                {"fixture_sha256": assets["files"][name], "actions": actions[0].tolist()}
            )
        sampled("timing")
        episodes = []

        class ChunkPolicy(torch.nn.Module):
            def reset(self):
                self.queue, self.chunk = collections.deque(), 0

            def select_action(self, observation):
                if not self.queue:
                    actions, _ = runtime.predict(observation, noise(self.seed + self.chunk))
                    self.queue.extend(actions.transpose(0, 1))
                    self.chunk += 1
                return self.queue.popleft()

        for row in request["protocol"]["episodes"]:
            stage = stage_name(row)
            begin(stage)
            config = LiberoEnv(task="libero_spatial", task_ids=[row["task_id"]])
            envs = make_env(config, n_envs=1, use_async_envs=False)
            try:
                env = envs["libero_spatial"][row["task_id"]]
                base = env.envs[0].unwrapped
                states = getattr(base, "_init_states", None)
                if states is None or row["init_state_id"] >= len(states):
                    raise ValueError(
                        "Requested initial state is unavailable; no wraparound allowed"
                    )
                horizon = env.call("_max_episode_steps")[0]
                if type(horizon) is not int or not 1 <= horizon <= request["evaluation"]["steps"]:
                    raise ValueError("Requested step ceiling cannot complete the benchmark episode")
                env.set_attr("init_state_id", row["init_state_id"])
                pre, post = make_env_pre_post_processors(config, runtime.config)
                policy = ChunkPolicy()
                policy.seed = row["noise_seed"]
                data = rollout(
                    env, policy, pre, post, lambda x: x, lambda x: x, seeds=[row["seed"]]
                )
                if not bool(data["done"][0, -1]) or data["action"].shape[1] > horizon:
                    raise ValueError("Spatial episode did not reach a benchmark termination")
                episodes.append(
                    {
                        **row,
                        "status": "episode_complete",
                        "task_success": bool(data["success"].any()),
                        "steps": data["action"].shape[1],
                        "benchmark_horizon": horizon,
                    }
                )
            finally:
                close_envs(envs)
            sampled(stage)
        if verify(source) != manifest:
            raise ValueError("Package assets changed during inference")
        atomic_json(
            output / "result.json",
            {
                "protocol_sha256": request["protocol_sha256"],
                "manifest_sha256": request["manifest_sha256"],
                "samples_ms": samples,
                "fixture_actions": fixture_actions,
                "fixture_actions_deterministic": True,
                "episodes": episodes,
                "process_id": os.getpid(),
                "inference_runtime": inference_identity(request["runtime"]),
            },
        )
    finally:
        runtime.close()
