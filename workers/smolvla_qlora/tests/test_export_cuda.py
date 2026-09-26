"""Real GPU export parity on a tiny native-shaped policy, not full-model evidence."""

import pytest


@pytest.mark.cuda
@pytest.mark.parametrize("method", ["lora", "qlora"])
def test_materialized_export_preserves_trained_base_and_saved_modules(method):
    torch = pytest.importorskip("torch")
    pytest.importorskip("bitsandbytes")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 CUDA required")
    from peft import LoraConfig, get_peft_model

    from firebird_vla.export import floating_state_dict
    from firebird_vla.model import quantize_linears

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, 32)
            self.head = torch.nn.Linear(32, 8)

        def forward(self, x):
            return self.head(torch.tanh(self.linear(x)))

    torch.manual_seed(42)
    policy = Tiny()
    if method == "qlora":
        quantize_linears(policy, roots=("linear",))
    policy.to("cuda")
    policy = get_peft_model(
        policy, LoraConfig(r=4, lora_alpha=8, target_modules=["linear"], modules_to_save=["head"])
    )
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.02)
    policy.eval()
    inputs = torch.randn(2, 32) * 0.1
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = policy(inputs.cuda()).float().cpu()
    state = floating_state_dict(policy)
    restored = Tiny()
    restored.load_state_dict(state, strict=True)
    with torch.inference_mode():
        actual = restored(inputs)
    torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.05)
    assert set(state) == {"linear.weight", "linear.bias", "head.weight", "head.bias"}
    assert all(
        value.dtype == torch.float32 and value.device.type == "cpu" for value in state.values()
    )
