# Firebird Quant

One packed-weight interface across model families. Firebird Quant discovers
registered dense tensors by shape and dtype, without a SmolVLA/LLM allowlist.
It packs real signed 4-bit or 8-bit codes, records every retained tensor, and
provides a portable eager PyTorch inference path and a reloadable artifact.

This is an isolated worker library and CLI. It does not install Torch into core
or change the application's advertised policy/runtime capabilities. Existing
SmolVLA GGUF deployment remains available through `policykit-quantize`; its native
formats and runtime contract are unchanged.

## Use with a model

From `workers/firebird_quant`, in a separate Python 3.11/3.12 environment:

```sh
uv venv --python 3.11
# CPU Linux: install the CPU wheel before the editable package.
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.11.0
uv pip install --python .venv/bin/python -e '.[test]'
```

On macOS use the ordinary PyPI Torch wheel instead of the CPU index. Intel macOS
uses Torch 2.2.2 with NumPy 1.26.4. The current Torch 2.11.0 verification uses
NumPy 2.2.6. The library allows either NumPy major version; pin the documented pair
for your Torch version instead of downgrading an existing isolated ACT environment.
The CI matrix contains those two explicit pairs, with a dependency consistency check.

```python
import torch
from firebird_quant import Recipe, load_model, quantize

model = load_your_model().eval()  # Your normal, trusted architecture/checkpoint loader.
result = quantize(model, Recipe(bits=4, exclude=("output_head.*",)))
with torch.inference_mode():
    outputs = result.model(*your_inputs)
result.save("model.fbq")
print(result.audit)

# Supply the same architecture, configuration, original dtypes and weight ties.
restored = load_model(build_your_model().eval(), "model.fbq")
with torch.inference_mode():
    reloaded_outputs = restored.model(*your_inputs)
```

`quantize` returns a separate, frozen evaluation model. The original model is
unchanged. Both 4-bit and 8-bit recipes use this same API for MLPs, CNNs, audio
convolutions, recurrent networks, transformer components, diffusion denoisers,
robotics networks, and custom modules that read their registered weights normally.
There is no model-name registry to extend for these cases.

Defaults pack floating tensors with at least two dimensions and 128 elements,
provided packing actually saves tensor bytes. Biases and normalization vectors
remain exact by default. Module buffers remain exact. `include` and `exclude`
are glob patterns on full parameter names; exclusion wins. Set `min_ndim=0`
and `min_elements=0` explicitly to attempt vectors and scalars as well. Shared
parameters share a packed representation, and protecting any alias protects
the entire tied group. Odd lengths and arbitrary convolution dimensions are
padded inside the packed representation and restored to their original shapes.

## Use with a checkpoint or another loader

```sh
firebird-quant quantize model.safetensors --out model.fbq --bits 4
firebird-quant inspect model.fbq
firebird-quant dequantize model.fbq --out restored.safetensors
```

The CLI reads local, single-file safetensors checkpoints. It never imports model
code, downloads a checkpoint, loads pickle, or infers architecture from filenames.
For a sharded checkpoint, use its normal loader and the model API, or pass the
assembled tensor dictionary to `quantize_state_dict`:

```python
from firebird_quant import Recipe, load, quantize_state_dict

state = quantize_state_dict(tensors, Recipe(bits=8, exclude=("*norm*",)))
state.save("weights.fbq")
ordinary_state_dict = load("weights.fbq").dequantize()
```

A tensor dictionary alone cannot distinguish parameters from buffers; its
shape/dtype policy applies to every entry. Supply `parameter_names=set(...)`
to protect buffers, or use `quantize(model)`, which discovers them automatically.
The dequantized dictionary can be strictly loaded by the original framework
adapter. This restores quantized values into floating tensors; it does not
provide a packed execution backend for TensorFlow, JAX, ONNX, or GGUF.

## Format and execution contract

Each contiguous group uses a symmetric scale and signed codes in [-7, 7] or
[-127, 127]. INT4 packs two offset codes per byte. Scales are FP32 (FP64 for
FP64 source weights). This is ordinary round-to-nearest, weight-only quantization,
not a new quantization algorithm and not GGML Q4_0 or an existing calibrated
SmolVLA recipe. Activations keep the model's floating compute dtype. Different
families may need different exclusions, precision, calibration, or task evaluation.

The portable runtime uses PyTorch parametrizations: packed tensors are buffers,
and weights are dequantized when accessed by the original operators. No full
floating master is stored in those parametrizations. This supports convolutions
and functional/custom weight consumers without custom low-bit kernels. It may
be slower than the original model, and operations such as RNNs can cache expanded
weights. Conversion also makes an independent copy of the model. Packed tensor
bytes are not a claim about peak RAM, VRAM, latency, or task accuracy.

`.fbq` is a single safetensors file with a versioned JSON manifest, the precision
map, ties, source tensor hashes, and packed-storage checksums. Publication reloads
and validates the file before making the destination visible and refuses to
overwrite an existing path. The audit separates tensor bytes from actual file
size (which also includes metadata). Checksums detect accidental corruption;
they do not authenticate an untrusted producer. Reload reconstructs tensor-byte
counts and precision coverage from the decoded inventory; imported audit claims
cannot set `quality_verified` or `speedup_verified`. Source hashes and the declared
recipe remain explicitly unverified provenance because the original floating
checkpoint is not available to the loader. Architecture code, processors,
tokenizers, normalization/configuration files and task evaluators remain owned
by the original model package. Keep those with the `.fbq` artifact.

`result.save` always exports the frozen conversion. Device/dtype moves on
`result.model` do not change the saved recipe. Use `load_model`, not a raw
`load_state_dict` of the parametrized model, to restore the packed architecture.

## Compatibility and evidence

Offline tests execute both precisions on deterministic small MLPs, grouped
Conv1d/2d/3d and transposed convolution, BatchNorm/GroupNorm, GRU/LSTM/RNN,
TransformerEncoderLayer, tied embedding/output weights, a residual denoising
block, and a custom functional matrix operation with nonstandard parameter
names. Tests compare the portable runtime with an independently dequantized
ordinary model, save/reload it, check the original is unchanged, and exercise
the CLI in a separate process. These are implementation tests, not pretrained
model quality benchmarks or evidence that every architecture has been tested.

The supported runtime contract is a dense, materialized eager `torch.nn.Module`
using registered tensors. Sparse/previously quantized tensors, meta/lazy weights,
existing parametrizations, TorchScript, custom state-dictionary hooks/extra state,
and shared-storage views require explicit adaptation and fail rather than being
silently presented as supported. Models that keep private cached weight copies,
modify weights during forward, require Parameter identity, use distributed tensor
subclasses, or require compiled/custom fused loaders also need an adapter and
execution validation. Training the converted model is unsupported. GPU execution,
torch.compile, and export to another runtime are not qualified by CPU tests.

For any new checkpoint, compare the floating and packed model with its real
preprocessing and complete task evaluator. The audit deliberately leaves
`quality_verified` and `speedup_verified` false; successful conversion cannot
promote either claim.

```sh
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
```

CI is configured for Torch 2.2.2 / NumPy 1.26.4 and Torch 2.11.0 / NumPy 2.2.6
on Linux CPU and participates in the required native worker verification gate.
The initial PR's hosted jobs did not start because of the account billing block;
workflow configuration is not a completed hosted test result.

## Optional native ACT package worker

The [native ACT package operation](NATIVE_ACT.md) preserves complete saved
processors and source lineage around packed weights, measures full-chunk drift
from FP32, and verifies an independent packed-only CPU reload before publication.
It accepts the existing bounded ACT inference recipe. It does not, by itself,
enable app Quantize or Isaac loading, qualify native SmolVLA, or establish
hardware/task quality. It reuses the existing isolated ACT environment unchanged.
