"""Shared observation/noise contract for native SmolVLA and patched vla.cpp.

Every backend returns unnormalized CPU actions. Saved fixtures pair exact raw
observations and explicit FP32 noise; quality runs replay exactly 50 actions.
"""

import argparse
import collections
import contextlib
import hashlib
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from smolvla_gpu_probe import (
    BACKBONE,
    BACKBONE_REVISION,
    LM_ROOT,
    MODEL,
    REVISION,
    patch_int8_activation_casts,
)


def noise(seed):
    return torch.from_numpy(
        np.random.default_rng(seed).standard_normal((1, 50, 32)).astype("float32")
    )


class Runtime:
    def __init__(self, backend, root, output):
        from benchmark_model import load_checkpoint_weights, load_smolvla_config
        from huggingface_hub import snapshot_download
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        started = time.perf_counter()
        self.backend, self.server, self.sock = backend, None, None
        self.root, self.output = root, output
        self.path = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
        backbone = snapshot_download(BACKBONE, revision=BACKBONE_REVISION, local_files_only=True)
        self.config = load_smolvla_config(self.path)
        self.config.device = "cpu"
        self.config.vlm_model_name = backbone
        self.config.load_vlm_weights = False
        self.config.compile_model = False
        self.pre, self.post = make_pre_post_processors(
            self.config,
            pretrained_path=self.path,
            preprocessor_overrides={
                "device_processor": {"device": "cpu"},
                "tokenizer_processor": {"tokenizer_name": backbone},
            },
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self.prepare_images = lambda batch: SmolVLAPolicy.prepare_images(self, batch)[0]
        if backend.startswith("native-"):
            precision = backend.removeprefix("native-")
            self.dtype = {"f32": torch.float32, "fp16": torch.float16}.get(
                precision, torch.bfloat16
            )
            self.policy = load_checkpoint_weights(
                SmolVLAPolicy(self.config), Path(self.path) / "model.safetensors"
            )
            self.policy.requires_grad_(False).eval().to(dtype=self.dtype)
            if precision == "nf4":
                from benchmark_model import quantize_linears

                quantize_linears(self.policy, roots=(LM_ROOT,))
            elif precision == "int8":
                import bitsandbytes as bnb

                for name, layer in list(self.policy.named_modules()):
                    if name.startswith(LM_ROOT) and isinstance(layer, torch.nn.Linear):
                        packed = bnb.nn.Linear8bitLt(
                            layer.in_features,
                            layer.out_features,
                            bias=layer.bias is not None,
                            has_fp16_weights=False,
                            threshold=6.0,
                        )
                        packed.load_state_dict(layer.state_dict())
                        packed.requires_grad_(False).to("cuda")
                        parent, _, child = name.rpartition(".")
                        self.policy.get_submodule(parent).set_submodule(child, packed)
            patch_int8_activation_casts()
            self.policy.to("cuda")
            self.config.device = "cuda"
        else:
            import importlib
            import tempfile

            from grpc_tools import protoc

            self.proto_dir = tempfile.TemporaryDirectory(prefix="smolvla-proto-")
            proto = root / "vla.cpp/src/serving/vla.proto"
            code = protoc.main(
                ["protoc", f"-I{proto.parent}", f"--python_out={self.proto_dir.name}", str(proto)]
            )
            if code:
                raise RuntimeError("Protobuf generation failed")
            sys.path.insert(0, self.proto_dir.name)
            self.pb = importlib.import_module("vla_pb2")
            import zmq

            with socket.socket() as port:
                port.bind(("127.0.0.1", 0))
                number = port.getsockname()[1]
            address = f"tcp://127.0.0.1:{number}"
            self.log = (output / "server.log").open("w")
            self.server = subprocess.Popen(
                [
                    str(root / "vla.cpp/build-cuda/vla-server"),
                    "--bind",
                    address,
                    str(root / "gguf" / f"smolvla-{backend.removeprefix('cpp-')}.gguf"),
                ],
                stdout=self.log,
                stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + 90
            while True:
                if self.server.poll() is not None:
                    raise RuntimeError("C++ server exited")
                try:
                    with socket.create_connection(("127.0.0.1", number), timeout=0.2):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("C++ server startup")
                    time.sleep(0.1)
            self.sock = zmq.Context.instance().socket(zmq.REQ)
            self.sock.setsockopt(zmq.RCVTIMEO, 120000)
            self.sock.setsockopt(zmq.LINGER, 0)
            self.sock.connect(address)
            if "backend = CUDA" not in (output / "server.log").read_text():
                raise RuntimeError("C++ CUDA backend not established")
        self.startup_s = time.perf_counter() - started
        self.startup_peak_bytes = (
            torch.cuda.max_memory_allocated() if backend.startswith("native-") else None
        )

    def predict(self, raw, initial_noise):
        tick = time.perf_counter()
        batch = self.pre(dict(raw))
        if self.backend.startswith("native-"):
            batch = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in batch.items()}
            self.policy.reset()
            amp = (
                torch.autocast("cuda", dtype=self.dtype)
                if self.dtype != torch.float32
                else contextlib.nullcontext()
            )
            with torch.inference_mode(), amp:
                actions = (
                    self.post(
                        self.policy.predict_action_chunk(batch, noise=initial_noise.to("cuda"))
                    )
                    .float()
                    .cpu()
                )
            torch.cuda.synchronize()
        else:
            req = self.pb.PredictRequest()
            for img in self.prepare_images(batch):
                # Native image resizing and [-1,1] transform are shared. Server
                # applies the inverse affine range transform exactly once.
                arr = ((img[0] + 1) / 2).permute(1, 2, 0).contiguous().numpy()
                field = req.images.add()
                field.encoding = self.pb.Image.F32_RGB_01
                field.height, field.width = arr.shape[:2]
                field.data = arr.tobytes()
            mask = batch["observation.language.attention_mask"][0].bool()
            req.lang_tokens.extend(batch["observation.language.tokens"][0][mask].tolist())
            # C++ owns state normalization and action unnormalization, using GGUF stats.
            state = raw["observation.state"].reshape(-1).tolist()
            req.state.extend(state + [0.0] * (32 - len(state)))
            req.noise.extend(initial_noise.reshape(-1).tolist())
            self.sock.send(req.SerializeToString())
            response = self.pb.PredictResponse()
            response.ParseFromString(self.sock.recv())
            if response.error:
                raise RuntimeError(response.error)
            actions = torch.tensor(response.action_chunk).reshape(
                1, response.chunk_size, response.action_dim
            )[:, :, :7]
        if actions.shape != (1, 50, 7) or not torch.isfinite(actions).all():
            raise ValueError("Invalid action chunk")
        return actions, 1000 * (time.perf_counter() - tick)

    def close(self):
        if self.sock is not None:
            self.sock.close(linger=0)
        if self.server is not None:
            self.server.terminate()
            try:
                self.server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server.kill()
                self.server.wait()
            self.log.close()


def save_fixture(path, raw, initial_noise):
    values = {k: v.cpu().numpy() for k, v in raw.items() if torch.is_tensor(v)}
    values["task"] = np.array(raw["task"])
    values["noise"] = initial_noise.numpy()
    np.savez_compressed(path, **values)


def load_fixture(path):
    with np.load(path, allow_pickle=False) as data:
        raw = {k: torch.from_numpy(data[k]) for k in data.files if k not in ("task", "noise")}
        raw["task"] = data["task"].tolist()
        return raw, torch.from_numpy(data["noise"])


def run(args, result):
    torch.set_num_threads(4)
    if args.backend.startswith("vllm-"):
        from llm_policy_runtime import LLMRuntime

        runtime = LLMRuntime(args.backend, args.root, args.output)
    else:
        runtime = Runtime(args.backend, args.root, args.output)
    result.update(
        startup_s=runtime.startup_s, backend=args.backend, checkpoint=MODEL, revision=REVISION
    )
    try:
        fixtures = sorted(args.fixtures.glob("*.npz"))
        if not fixtures:
            args.fixtures.mkdir(parents=True, exist_ok=True)
            generator = torch.Generator().manual_seed(42)
            for i, task in enumerate(
                [
                    "pick up the black bowl between the plate and the ramekin "
                    "and place it on the plate",
                    "put the red mug on the plate",
                ]
            ):
                raw = {
                    "observation.images.image": torch.rand(1, 3, 360, 360, generator=generator),
                    "observation.images.image2": torch.rand(1, 3, 360, 360, generator=generator),
                    "observation.state": torch.tensor(
                        [[0.05, -0.1, 0.9, 0.0, 0.0, 0.0, 0.02, -0.02]]
                    ),
                    "task": [task],
                }
                save_fixture(args.fixtures / f"synthetic-{i}.npz", raw, noise(42 + i))
            fixtures = sorted(args.fixtures.glob("*.npz"))
        result["startup_peak_allocated_bytes"] = runtime.startup_peak_bytes
        if args.backend.startswith("native-"):
            torch.cuda.reset_peak_memory_stats()
        result["fixtures"] = []
        for path in fixtures:
            raw, n = load_fixture(path)
            for _ in range(args.warmup):
                runtime.predict(raw, n)
            samples = []
            for _ in range(args.samples):
                actions, ms = runtime.predict(raw, n)
                samples.append(ms)
            repeated, _ = runtime.predict(raw, n)
            torch.testing.assert_close(actions, repeated, atol=0, rtol=0)
            np.save(args.output / (path.stem + ".actions.npy"), actions.numpy())
            result["fixtures"].append(
                {
                    "name": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "p50_ms": float(np.median(samples)),
                    "samples_ms": samples,
                    "repeat_exact": True,
                }
            )
        if args.backend.startswith("native-"):
            result["inference_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            result["inference_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        if args.quality:
            from lerobot.envs.configs import LiberoEnv
            from lerobot.envs.factory import make_env, make_env_pre_post_processors
            from lerobot.envs.utils import close_envs
            from lerobot.scripts.lerobot_eval import rollout

            class ChunkPolicy(torch.nn.Module):
                def reset(self):
                    self.queue = collections.deque()
                    self.chunk = 0

                def select_action(self, observation):
                    if not self.queue:
                        n = noise(self.seed + self.chunk)
                        if self.capture and self.chunk < 3:
                            save_fixture(
                                args.fixtures
                                / f"task{self.task}-init{self.episode}-chunk{self.chunk}.npz",
                                observation,
                                n,
                            )
                        actions, ms = runtime.predict(observation, n)
                        self.times.append(ms)
                        self.queue.extend(actions.transpose(0, 1))
                        self.chunk += 1
                    return self.queue.popleft()

            policy = ChunkPolicy()
            policy.capture = args.capture
            policy.times = []
            result["episodes"] = []
            for task in args.task_ids:
                cfg = LiberoEnv(task="libero_spatial", task_ids=[task])
                envs = make_env(cfg, n_envs=1, use_async_envs=False)
                ep, eo = make_env_pre_post_processors(cfg, runtime.config)
                env = envs["libero_spatial"][task]
                try:
                    for episode in range(args.episodes):
                        env.set_attr("init_state_id", episode)
                        policy.task = task
                        policy.episode = episode
                        policy.seed = 42 + episode * 1000
                        data = rollout(
                            env, policy, ep, eo, lambda x: x, lambda x: x, seeds=[42 + episode]
                        )
                        row = {
                            "task": task,
                            "init_state": episode,
                            "seed": 42 + episode,
                            "noise_seed": policy.seed,
                            "success": bool(data["success"].any()),
                            "steps": data["action"].shape[1],
                        }
                        result["episodes"].append(row)
                        print(row, flush=True)
                        (args.output / "result.json").write_text(json.dumps(result, indent=2))
                finally:
                    close_envs(envs)
            result["quality_chunk_ms"] = policy.times
        result["status"] = "passed"
    finally:
        runtime.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--backend", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--quality", action="store_true")
    p.add_argument("--capture", action="store_true")
    p.add_argument("--task-ids", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--episodes", type=int, default=1)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "running",
        "contract": (
            "Raw CPU observations/task plus explicit FP32 noise to complete "
            "unnormalized CPU 50x7 actions; chunk replay 50; preprocessing and RPC included; "
            "simulator excluded."
        ),
    }
    try:
        run(args, result)
    except Exception:
        import traceback

        result.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
