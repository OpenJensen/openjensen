"""Real bitsandbytes/PEFT test, skipped explicitly on hosts without the CUDA stack."""

import pytest


@pytest.mark.cuda
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_nf4_gradient_freeze_and_adapter_reload(tmp_path, dtype_name):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA GPU")
    if dtype_name == "bfloat16" and torch.cuda.get_device_capability(0)[0] < 8:
        pytest.skip("Native BF16 requires Ampere or newer")
    dtype = getattr(torch, dtype_name)
    bnb = pytest.importorskip("bitsandbytes")
    peft = pytest.importorskip("peft")
    from firebird_vla.model import promote_trainable_parameters, quantize_linears
    from firebird_vla.train import optimizer_step

    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = {}
            self.layers = torch.nn.ModuleDict({"q_proj": torch.nn.Linear(64, 64, bias=False)})

        def forward(self, x):
            layer = self.layers["q_proj"]
            # Reproduce SmolVLA's critical activation-to-weight cast.
            return layer(x.to(layer.weight.dtype))

    def base():
        torch.manual_seed(11)
        model = TinyPolicy().requires_grad_(False)
        quantize_linears(model, roots=("layers.",), compute_dtype=dtype)
        return model

    model = peft.get_peft_model(
        base(),
        peft.LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj"],
            lora_dropout=0,
        ),
    )
    layer = model.base_model.model.layers["q_proj"].base_layer
    assert isinstance(layer, bnb.nn.Linear4bit)
    assert layer.weight.dtype == dtype
    assert layer.weight.quant_state.quant_type == "nf4"
    frozen_before = layer.weight.detach().clone()
    promote_trainable_parameters(model)
    x = torch.randn(2, 64, device="cuda", dtype=dtype)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    with torch.autocast("cuda", dtype=dtype):
        loss = model(x).float().square().mean()
    scaler.scale(loss).backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for name, p in model.named_parameters()
        if "lora_" in name
    )
    assert layer.weight.grad is None
    _, updated = optimizer_step(
        [p for p in model.parameters() if p.requires_grad], optimizer, scaler, 1
    )
    assert updated
    # Packed floating storage is opaque bytes, possibly NaN bit patterns: compare byte views.
    assert torch.equal(layer.weight.view(torch.uint8), frozen_before.view(torch.uint8))
    model.eval()
    expected = model(x).detach()
    model.save_pretrained(tmp_path, save_embedding_layers=False)
    reloaded = peft.PeftModel.from_pretrained(base(), tmp_path).eval()
    torch.testing.assert_close(reloaded(x), expected)
