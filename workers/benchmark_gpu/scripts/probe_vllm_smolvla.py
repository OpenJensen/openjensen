"""Prove full SmolVLA execution inside a registered vLLM pooling worker."""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from smolvla_cross_stack import Runtime, load_fixture
from transformers import BertConfig
from vllm_smolvla_plugin import ENVELOPE_TOKENS, ENVELOPE_WIDTH


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fixtures", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "running",
        "integration": (
            "Custom full-policy vLLM V0 pooling worker using native PyTorch "
            "SmolVLA kernels; no claim of vLLM attention acceleration."
        ),
    }
    try:
        torch.set_num_threads(4)
        reference = Runtime("native-bf16", Path.cwd(), args.output)
        pre, post = reference.pre, reference.post
        config = reference.config
        cases = []
        for path in sorted(args.fixtures.glob("*.npz"))[:3]:
            raw, noise = load_fixture(path)
            expected, _ = reference.predict(raw, noise)
            cases.append((path, raw, noise, expected))
        model_dir = args.output / "model"
        model_dir.mkdir()
        adapter = BertConfig(
            hidden_size=ENVELOPE_WIDTH,
            num_hidden_layers=1,
            num_attention_heads=16,
            intermediate_size=ENVELOPE_WIDTH,
            vocab_size=256,
            max_position_embeddings=1024,
        )
        adapter.architectures = ["SmolVLAFullPolicyPooler"]
        adapter.native_checkpoint = reference.path
        adapter.native_backbone = config.vlm_model_name
        adapter.save_pretrained(model_dir)
        (model_dir / "model.safetensors").symlink_to(Path(reference.path) / "model.safetensors")
        del reference
        gc.collect()
        torch.cuda.empty_cache()
        from vllm import LLM, PoolingParams
        from vllm.model_executor.models import ModelRegistry

        ModelRegistry.register_model(
            "SmolVLAFullPolicyPooler", "vllm_smolvla_plugin:SmolVLAFullPolicyPooler"
        )
        started = time.perf_counter()
        llm = LLM(
            model=str(model_dir.resolve()),
            task="embed",
            dtype="float32",
            skip_tokenizer_init=True,
            enable_prompt_embeds=True,
            enforce_eager=True,
            max_model_len=1024,
            max_num_seqs=1,
            max_num_batched_tokens=1024,
            gpu_memory_utilization=0.3,
            enable_chunked_prefill=False,
        )
        result["startup_s"] = time.perf_counter() - started
        from types import SimpleNamespace

        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        shell = SimpleNamespace(config=config)
        rows = []
        for path, raw, noise, expected in cases:
            times = []
            for i in range(4):
                started = time.perf_counter()
                batch = pre(dict(raw))
                images = SmolVLAPolicy.prepare_images(shell, batch)[0]
                state = SmolVLAPolicy.prepare_state(shell, batch)
                flat = torch.cat(
                    [
                        *(im.reshape(-1) for im in images),
                        state.reshape(-1),
                        batch["observation.language.tokens"].float().reshape(-1),
                        batch["observation.language.attention_mask"].float().reshape(-1),
                        noise.reshape(-1),
                    ]
                )
                payload = torch.zeros(ENVELOPE_TOKENS * ENVELOPE_WIDTH, dtype=torch.float32)
                payload[: flat.numel()] = flat
                outputs = llm.encode(
                    {"prompt_embeds": payload.reshape(ENVELOPE_TOKENS, ENVELOPE_WIDTH)},
                    pooling_params=PoolingParams(),
                    use_tqdm=False,
                )
                data = outputs[0].outputs.data.float().cpu()
                assert data[350] > 0, "Full policy forward was not executed"
                actions = post(data[:350].reshape(1, 50, 7)).float().cpu()
                times.append((time.perf_counter() - started) * 1000)
            error = (actions - expected).abs()
            row = {
                "fixture": path.name,
                "p50_ms": float(np.median(times[1:])),
                "mae": float(error.mean()),
                "max_abs": float(error.max()),
                "worker_policy_calls": float(data[350]),
            }
            rows.append(row)
            np.save(args.output / (path.stem + ".actions.npy"), actions.numpy())
            torch.testing.assert_close(actions, expected, atol=0.0001, rtol=0.0001)
        result.update(status="passed", cases=rows, full_action_execution=True)
    except Exception:
        import traceback

        result.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
