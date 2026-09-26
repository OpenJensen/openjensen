"""Complete policy execution in custom LLM workers, using native PyTorch kernels."""

import time
from pathlib import Path

import torch
from benchmark_model import load_smolvla_config
from smolvla_action_envelope import ENVELOPE_TOKENS, ENVELOPE_WIDTH
from smolvla_gpu_probe import BACKBONE, BACKBONE_REVISION, MODEL, REVISION


class LLMRuntime:
    def __init__(self, backend, root, output):
        from huggingface_hub import snapshot_download
        from lerobot.policies.factory import make_pre_post_processors
        from transformers import BertConfig, LlamaConfig

        self.backend = backend
        if backend == "vllm-bf16":
            from vllm import LLM, PoolingParams
            from vllm.model_executor.models import ModelRegistry
        elif backend == "trtllm-fp16":
            import trtllm_smolvla_plugin  # noqa: F401
            from tensorrt_llm import LLM, SamplingParams
            from tensorrt_llm.llmapi import KvCacheConfig
        else:
            raise ValueError("Unsupported custom full-policy engine candidate")
        self.integration = "Custom engine orchestration with native PyTorch SmolVLA kernels"
        tick = time.perf_counter()
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
        model_dir = output / "transport-model"
        model_dir.mkdir()
        config_class = BertConfig if backend == "vllm-bf16" else LlamaConfig
        adapter = config_class(
            hidden_size=ENVELOPE_WIDTH,
            num_hidden_layers=1,
            num_attention_heads=16,
            num_key_value_heads=16,
            intermediate_size=ENVELOPE_WIDTH,
            vocab_size=256 if backend == "vllm-bf16" else 512,
            max_position_embeddings=1024,
        )
        adapter.architectures = [
            "SmolVLAFullPolicyPooler" if backend == "vllm-bf16" else "SmolVLAFullPolicyForAction"
        ]
        if backend == "trtllm-fp16":
            adapter.torch_dtype = torch.float16
        adapter.native_checkpoint = self.path
        adapter.native_backbone = backbone
        adapter.save_pretrained(model_dir)
        (model_dir / "model.safetensors").symlink_to(Path(self.path) / "model.safetensors")
        if backend == "vllm-bf16":
            ModelRegistry.register_model(
                "SmolVLAFullPolicyPooler", "vllm_smolvla_plugin:SmolVLAFullPolicyPooler"
            )
            self.engine = LLM(
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
            self.params = PoolingParams()
        else:
            self.engine = LLM(
                model=str(model_dir.resolve()),
                backend="pytorch",
                dtype="float16",
                skip_tokenizer_init=True,
                max_batch_size=1,
                max_seq_len=1024,
                max_num_tokens=1024,
                disable_overlap_scheduler=True,
                use_cuda_graph=False,
                enable_trtllm_sampler=True,
                kv_cache_config=KvCacheConfig(max_tokens=1024),
            )
            self.engine.input_processor = lambda inputs, params: (
                [0],
                {"mm_embedding": inputs["mm_processor_kwargs"]["envelope"]},
            )
            self.params = SamplingParams(
                max_tokens=1,
                temperature=0,
                end_id=511,
                pad_id=0,
                return_context_logits=True,
                detokenize=False,
            )
        self.startup_s = time.perf_counter() - tick
        self.startup_peak_bytes = torch.cuda.max_memory_allocated()
        self.calls = 0

    def predict(self, raw, noise):
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        tick = time.perf_counter()
        batch = self.pre(dict(raw))
        images = SmolVLAPolicy.prepare_images(self, batch)[0]
        state = SmolVLAPolicy.prepare_state(self, batch)
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
        if self.backend == "vllm-bf16":
            outputs = self.engine.encode(
                {"prompt_embeds": payload.reshape(ENVELOPE_TOKENS, ENVELOPE_WIDTH)},
                pooling_params=self.params,
                use_tqdm=False,
            )
            data = outputs[0].outputs.data.float().cpu()
        else:
            outputs = self.engine.generate(
                {
                    "prompt": "action",
                    "mm_processor_kwargs": {
                        "envelope": payload.reshape(ENVELOPE_TOKENS, ENVELOPE_WIDTH)
                    },
                },
                sampling_params=self.params,
                use_tqdm=False,
            )
            if outputs.context_logits is None:
                raise RuntimeError("TensorRT-LLM did not return the action output buffer")
            data = outputs.context_logits.reshape(-1, 512)[-1].float().cpu()
        if float(data[350]) <= self.calls:
            raise RuntimeError("Worker did not execute a new full-policy request")
        self.calls = float(data[350])
        actions = self.post(data[:350].reshape(1, 50, 7)).float().cpu()
        if actions.shape != (1, 50, 7) or not torch.isfinite(actions).all():
            raise ValueError("Invalid action chunk")
        return actions, (time.perf_counter() - tick) * 1000

    def close(self):
        if self.backend == "trtllm-fp16":
            self.engine.shutdown()
        del self.engine
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
