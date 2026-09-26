"""Full SmolVLA policy in a vLLM V0 pooling worker.

This adapter benchmarks vLLM orchestration around the native policy kernels; it
is not a claim of paged-attention acceleration. A fixed FP32 embedding envelope
preserves the preprocessed images, state, tokens, masks and diffusion noise.
Batch size one only. The returned pooling vector contains the 50x7 actions.
"""

import torch
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.models.interfaces import SupportsV0Only

ENVELOPE_WIDTH = 2048
IMAGE_VALUES = 3 * 512 * 512
PAYLOAD_VALUES = 2 * IMAGE_VALUES + 32 + 48 + 48 + 50 * 32
ENVELOPE_TOKENS = (PAYLOAD_VALUES + ENVELOPE_WIDTH - 1) // ENVELOPE_WIDTH


class SmolVLAFullPolicyPooler(torch.nn.Module, SupportsV0Only):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        from benchmark_model import load_smolvla_config
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        config = vllm_config.model_config.hf_config
        native = load_smolvla_config(config.native_checkpoint)
        native.device = "cpu"
        native.load_vlm_weights = False
        native.compile_model = False
        native.vlm_model_name = config.native_backbone
        self.policy = SmolVLAPolicy(native).requires_grad_(False).eval()
        self._pooler = Pooler.from_config_with_defaults(
            vllm_config.model_config.pooler_config,
            pooling_type=PoolingType.LAST,
            normalize=False,
            softmax=False,
        )
        self.calls = 0

    def load_weights(self, weights):
        values = dict(weights)
        self.policy.load_state_dict(values, strict=True, assign=True)
        self.policy.to(device="cuda", dtype=torch.bfloat16).eval()
        self.policy.config.device = "cuda"
        return {"policy." + name for name in values}

    def get_input_embeddings(self, input_ids):
        # Used only by vLLM's dummy profiling pass; actual requests must provide
        # the full envelope through prompt_embeds.
        return torch.zeros((input_ids.numel(), ENVELOPE_WIDTH), device=input_ids.device)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if inputs_embeds is None:
            return torch.zeros((positions.numel(), ENVELOPE_WIDTH), device=positions.device)
        if inputs_embeds.shape != (ENVELOPE_TOKENS, ENVELOPE_WIDTH):
            raise ValueError("Exactly one complete SmolVLA observation envelope is required")
        if inputs_embeds.dtype != torch.float32:
            raise ValueError("FP32 envelope required to preserve noise and observation values")
        flat = inputs_embeds.reshape(-1)
        offset = 0

        def take(n):
            nonlocal offset
            value = flat[offset : offset + n]
            offset += n
            return value

        images = [take(IMAGE_VALUES).reshape(1, 3, 512, 512) for _ in range(2)]
        state = take(32).reshape(1, 32)
        tokens = take(48).reshape(1, 48).long()
        masks = take(48).reshape(1, 48).bool()
        noise = take(1600).reshape(1, 50, 32)
        image_masks = [torch.ones(1, device=flat.device, dtype=torch.bool) for _ in range(2)]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            actions = self.policy.model.sample_actions(
                images, image_masks, tokens, masks, state, noise=noise
            )
        self.calls += 1
        hidden = torch.zeros_like(inputs_embeds)
        hidden[-1, :350] = actions[:, :, :7].float().reshape(-1)
        hidden[-1, 350] = self.calls
        return hidden

    def pooler(self, hidden_states, pooling_metadata):
        return self._pooler(hidden_states, pooling_metadata)
