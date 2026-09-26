"""Experimental full-policy TensorRT-LLM PyTorch executor adapter.

The request's multimodal tensor carries a lossless observation/noise envelope.
The context-output buffer carries normalized continuous actions; its first 350
values are NOT language logits. One fixed completion token closes the request.
This measures executor orchestration, not TensorRT-compiled policy acceleration.
"""

import torch
from smolvla_action_envelope import unpack
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM, register_auto_model


@register_auto_model("SmolVLAFullPolicyForAction")
class SmolVLAFullPolicyForAction(DecoderModelForCausalLM):
    def __init__(self, model_config):
        torch.nn.Module.__init__(self)
        from benchmark_model import load_smolvla_config
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.model_config = model_config
        self._config = model_config.pretrained_config
        native = load_smolvla_config(self._config.native_checkpoint)
        native.device = "cpu"
        native.load_vlm_weights = False
        native.compile_model = False
        native.vlm_model_name = self._config.native_backbone
        self.policy = SmolVLAPolicy(native).requires_grad_(False).eval()
        self.model = self.policy.model
        self.pp_rank = 0
        self.pp_size = 1
        self.calls = 0

    def post_init(self):
        pass

    @property
    def config(self):
        return self._config

    @property
    def vocab_size_padded(self):
        return 512

    def infer_max_seq_len(self):
        return 1024

    def load_weights(self, weights, **kwargs):
        self.policy.load_state_dict(weights, strict=True, assign=True)
        self.policy.to(device="cuda", dtype=torch.float16).eval()
        self.policy.config.device = "cuda"

    def forward(
        self,
        attn_metadata,
        input_ids=None,
        position_ids=None,
        inputs_embeds=None,
        return_context_logits=False,
        **kwargs,
    ):
        data = kwargs.get("multi_modal_data", [])
        logits = torch.zeros((input_ids.numel(), 512), device=input_ids.device, dtype=torch.float32)
        logits[:, 511] = 100.0
        if data:
            if len(data) != 1:
                raise ValueError("SmolVLA adapter supports one request at a time")
            images, image_masks, tokens, masks, state, noise = unpack(data[0])
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                actions = self.policy.model.sample_actions(
                    images, image_masks, tokens, masks, state, noise=noise
                )
            self.calls += 1
            logits[-1, :350] = actions[:, :, :7].float().reshape(-1)
            logits[-1, 350] = self.calls
        return logits
