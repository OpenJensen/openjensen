"""Shared observation/noise contract for native SmolVLA and patched vla.cpp.

Every backend returns unnormalized CPU actions. Saved fixtures pair exact raw
observations and explicit FP32 noise; quality runs replay exactly 50 actions.
"""

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from smolvla_gpu_probe import (
    BACKBONE,
    BACKBONE_REVISION,
    MODEL,
    REVISION,
)

# The benchmark and application intentionally execute one shared implementation.
# This trusted sibling path supports the historical standalone script invocation.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vla_cpp"))
from policykit.spatial_runtime import Runtime as _SharedRuntime
from policykit.spatial_runtime import load_fixture, noise, save_fixture


class Runtime(_SharedRuntime):
    def __init__(self, backend, root, output):
        from huggingface_hub import snapshot_download

        super().__init__(
            backend,
            root,
            output,
            policy_path=snapshot_download(MODEL, revision=REVISION, local_files_only=True),
            backbone_path=snapshot_download(
                BACKBONE, revision=BACKBONE_REVISION, local_files_only=True
            ),
        )


def run(args, result):
    torch.set_num_threads(4)
    if args.backend.startswith(("vllm-", "trtllm-")):
        from llm_policy_runtime import LLMRuntime

        runtime = LLMRuntime(args.backend, args.root, args.output)
    else:
        runtime = Runtime(args.backend, args.root, args.output)
    result.update(
        startup_s=runtime.startup_s, backend=args.backend, checkpoint=MODEL, revision=REVISION
    )
    if hasattr(runtime, "integration"):
        result["integration"] = runtime.integration
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
    p.add_argument(
        "--backend",
        required=True,
        choices=[
            "native-bf16",
            "native-fp16",
            "native-f32",
            "native-int8",
            "native-nf4",
            "cpp-bf16",
            "cpp-Q8_0",
            "cpp-Q4_0",
            "cpp-Q8_0-vision",
            "vllm-bf16",
            "trtllm-fp16",
        ],
    )
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
