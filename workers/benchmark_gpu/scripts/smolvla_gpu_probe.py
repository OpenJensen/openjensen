"""Pinned SmolVLA CUDA inference/paired LIBERO smoke probe, one candidate per process.

Synthetic latency is diagnostic, never a LIBERO success score. Downloads must be
completed before invocation so startup measures a warm filesystem cache load.
"""

import argparse
import hashlib
import importlib.metadata
import json
import time
import traceback
from pathlib import Path

MODEL = "lerobot/smolvla_libero"
REVISION = "31d453f7edd78c839a8bbc39744a292686daf0de"
BACKBONE = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
BACKBONE_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
LM_ROOT = "model.vlm_with_expert.vlm.model.text_model.layers."


def patch_int8_activation_casts():
    """Patch only activation dtype lookups, in memory, for LeRobot 0.4.4.

    Packed INT8 storage must never become an activation dtype. Floating layers
    keep exactly the upstream dtype; the floating-reference probe checks parity.
    """
    import ast
    import inspect
    import textwrap

    import bitsandbytes as bnb
    import torch
    from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel

    def activation_dtype(layer):
        if layer.weight.is_floating_point():
            return layer.weight.dtype
        if isinstance(layer, bnb.nn.Linear8bitLt):
            return torch.bfloat16
        raise TypeError(f"Unsupported integer weight storage in {type(layer)}")

    class Rewrite(ast.NodeTransformer):
        count = 0

        def visit_Attribute(self, node):
            if (
                node.attr == "dtype"
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "weight"
            ):
                self.count += 1
                return ast.copy_location(
                    ast.Call(
                        func=ast.Name(id="_activation_dtype", ctx=ast.Load()),
                        args=[node.value.value],
                        keywords=[],
                    ),
                    node,
                )
            return self.generic_visit(node)

    patched = {}
    for name in ("forward_attn_layer", "forward_cross_attn_layer", "forward"):
        original = getattr(SmolVLMWithExpertModel, name)
        tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
        rewrite = Rewrite()
        tree = ast.fix_missing_locations(rewrite.visit(tree))
        namespace = dict(original.__globals__, _activation_dtype=activation_dtype)
        exec(compile(tree, f"<smolvla-int8-{name}>", "exec"), namespace)
        patched[name] = (namespace[name], rewrite.count)
    if sum(count for _, count in patched.values()) != 7:
        raise RuntimeError("LeRobot activation cast layout changed; refusing unchecked patch")
    for name, (method, _) in patched.items():
        setattr(SmolVLMWithExpertModel, name, method)
    return {name: count for name, (_, count) in patched.items()}


def run(args, result):
    started = time.perf_counter()
    import numpy as np
    import torch
    from benchmark_model import load_checkpoint_weights, load_smolvla_config
    from huggingface_hub import snapshot_download
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dtype = torch.float16 if args.candidate == "fp16" else torch.bfloat16
    result["packages"] = {
        p: importlib.metadata.version(p)
        for p in ("torch", "lerobot", "transformers", "bitsandbytes", "peft")
    }
    result["gpu"] = torch.cuda.get_device_name(0)
    result["compute_capability"] = torch.cuda.get_device_capability(0)
    result["cuda_runtime"] = torch.version.cuda
    torch.cuda.reset_peak_memory_stats()
    path = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
    backbone = snapshot_download(BACKBONE, revision=BACKBONE_REVISION, local_files_only=True)
    config = load_smolvla_config(path)
    config.device = "cpu"
    config.load_vlm_weights = False
    config.vlm_model_name = backbone
    config.compile_model = False
    policy = load_checkpoint_weights(SmolVLAPolicy(config), Path(path) / "model.safetensors")
    policy.requires_grad_(False).eval().to(dtype=dtype)
    quantized = []
    torch.cuda.synchronize()
    quantize_started = time.perf_counter()
    if args.candidate == "nf4":
        from benchmark_model import quantize_linears

        quantized = quantize_linears(policy, roots=(LM_ROOT,))
    elif args.candidate == "int8":
        import bitsandbytes as bnb

        def require_float_activation(module, inputs):
            if not inputs[0].is_floating_point():
                raise TypeError(
                    "SmolVLA cast an activation to packed INT8 storage; "
                    "use --safe-int8-casts after floating-reference parity verification"
                )

        for name, layer in list(policy.named_modules()):
            if not name.startswith(LM_ROOT) or not isinstance(layer, torch.nn.Linear):
                continue
            packed = bnb.nn.Linear8bitLt(
                layer.in_features,
                layer.out_features,
                bias=layer.bias is not None,
                has_fp16_weights=False,
                threshold=6.0,
            )
            packed.load_state_dict(layer.state_dict())
            packed.requires_grad_(False).to("cuda")
            packed.register_forward_pre_hook(require_float_activation)
            parent, _, child = name.rpartition(".")
            policy.get_submodule(parent).set_submodule(child, packed)
            quantized.append(name)
    torch.cuda.synchronize()
    result["quantization_seconds"] = time.perf_counter() - quantize_started if quantized else None
    policy.to("cuda")
    config.device = "cuda"
    pre, post = make_pre_post_processors(
        config,
        pretrained_path=path,
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "tokenizer_processor": {"tokenizer_name": backbone},
        },
    )
    torch.cuda.synchronize()
    result.update(
        cached_startup_seconds=time.perf_counter() - started,
        startup_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        startup_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        quantized_modules=quantized,
        precision_map={
            n: {"class": type(m).__name__, "weight_dtype": str(m.weight.dtype)}
            for n, m in policy.named_modules()
            if isinstance(m, torch.nn.Linear)
        },
        chunk_size=config.chunk_size,
        n_action_steps=config.n_action_steps,
        denoising_steps=config.num_steps,
        checkpoint_hashes={
            p.name: hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
            for p in Path(path).iterdir()
            if p.is_file() and p.suffix in (".json", ".safetensors")
        },
    )
    # Saved preprocessing renames these two LIBERO cameras. The checkpoint's
    # empty_cameras=0 leaves the absent third camera absent, as upstream does.
    input_generator = torch.Generator().manual_seed(args.seed)
    from safetensors import safe_open

    stats_path = Path(path) / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    with safe_open(stats_path, framework="pt") as stats:
        state_dim = stats.get_slice("observation.state.mean").get_shape()[0]
    result["state_dim_from_saved_normalization"] = state_dim
    result["state_dim_in_config_metadata"] = config.input_features["observation.state"].shape[0]
    raw = {
        "observation.images.image": torch.rand(1, 3, 360, 360, generator=input_generator),
        "observation.images.image2": torch.rand(1, 3, 360, 360, generator=input_generator),
        "observation.state": torch.zeros(1, state_dim),
        "task": [
            "pick up the black bowl between the plate and the ramekin and place it on the plate"
        ],
    }
    result["synthetic_input_hashes"] = {
        key: hashlib.sha256(value.numpy().tobytes()).hexdigest()
        for key, value in raw.items()
        if torch.is_tensor(value)
    }
    result["task_text"] = raw["task"][0]

    def predict(seed):
        policy.reset()
        torch.manual_seed(seed)
        torch.cuda.synchronize()
        tick = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            batch = pre(dict(raw))
            actions = post(policy.predict_action_chunk(batch)).float().cpu()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - tick
        if actions.shape != (1, config.chunk_size, 7) or not torch.isfinite(actions).all():
            raise ValueError(f"Invalid complete action chunk: {actions.shape}")
        return elapsed, actions

    if args.safe_int8_casts:
        before = predict(args.seed)[1] if args.candidate in ("fp16", "bf16") else None
        result["activation_cast_patch"] = patch_int8_activation_casts()
        if before is not None:
            after = predict(args.seed)[1]
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            result["float_patch_exact_action_parity"] = True
    for i in range(args.warmup):
        predict(args.seed + i)
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for i in range(args.samples):
        elapsed, actions = predict(args.seed + i)
        samples.append(elapsed * 1000)
        if i == 0:
            reference = actions
    _, repeated = predict(args.seed)
    torch.testing.assert_close(reference, repeated, rtol=0, atol=0)
    result["synthetic_inference"] = {
        "status": "passed",
        "observation_kind": "synthetic",
        "samples_ms": samples,
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "action_shape": list(reference.shape),
        "deterministic_repeat": True,
        "boundary": (
            "CPU image/state tensors and task -> saved preprocessing -> full action chunk -> "
            "saved unnormalization -> CPU actions; excludes simulator and transport"
        ),
    }
    np.save(args.output.with_suffix(".actions.npy"), reference.numpy())
    # Persist inference results even if the independent simulator check fails.
    args.output.write_text(json.dumps(result, indent=2))
    if args.libero:
        from lerobot.envs.configs import LiberoEnv
        from lerobot.envs.factory import make_env, make_env_pre_post_processors
        from lerobot.envs.utils import close_envs
        from lerobot.scripts.lerobot_eval import eval_policy_all

        env_config = LiberoEnv(task="libero_spatial", task_ids=args.task_ids)
        envs = make_env(env_config, n_envs=1, use_async_envs=False)
        try:
            env_pre, env_post = make_env_pre_post_processors(env_config, config)
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                metrics = eval_policy_all(
                    envs,
                    policy,
                    env_pre,
                    env_post,
                    pre,
                    post,
                    n_episodes=args.episodes,
                    start_seed=args.seed,
                    max_parallel_tasks=1,
                )
            result["libero_smoke"] = {
                "status": "passed",
                "suite": "libero_spatial",
                "task_ids": args.task_ids,
                "episodes_per_task": args.episodes,
                "start_seed": args.seed,
                "init_state_ids_per_task": list(range(args.episodes)),
                "metrics": metrics,
                "final_quality_benchmark": False,
            }
        finally:
            close_envs(envs)
    result["status"] = "passed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=("fp16", "bf16", "nf4", "int8"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--libero", action="store_true")
    parser.add_argument("--safe-int8-casts", action="store_true")
    parser.add_argument("--task-ids", nargs="+", type=int, default=[0])
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0 or args.episodes < 1:
        parser.error("Require samples >= 1, warmup >= 0 and episodes >= 1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "running",
        "candidate": args.candidate,
        "checkpoint": MODEL,
        "revision": REVISION,
        "seed": args.seed,
        "libero_smoke": None,
        "synthetic_inference": None,
    }
    try:
        run(args, result)
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
