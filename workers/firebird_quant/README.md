# OPEN JENSEN Quant setup

## Install

From `workers/firebird_quant`, use Python 3.11 or 3.12:

```sh
uv venv --python 3.11
# CPU Linux: install the CPU wheel before the editable package.
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.11.0
uv pip install --python .venv/bin/python -e '.[test]'
. .venv/bin/activate
```

## Quantize a model

Use a materialized eager `torch.nn.Module` with registered dense tensors. Supply
the same architecture, configuration, original dtypes and weight ties when
reloading. Keep the source model and use a new output path.

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

## Quantize a checkpoint

```sh
firebird-quant quantize model.safetensors --out model.fbq --bits 4
firebird-quant inspect model.fbq
firebird-quant dequantize model.fbq --out restored.safetensors
```

To use a state dictionary with an external loader:

```python
from firebird_quant import Recipe, load, quantize_state_dict

state = quantize_state_dict(tensors, Recipe(bits=8, exclude=("*norm*",)))
state.save("weights.fbq")
ordinary_state_dict = load("weights.fbq").dequantize()
```

Choose `bits=4` or `bits=8`; `group_size=64` is the default. Load an `.fbq` file
with `load_model` and its architecture factory, or dequantize it before using an
ordinary state-dictionary loader.

## Run tests

```sh
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
```

For complete ACT policy packages, follow the [native ACT recipe](NATIVE_ACT.md).
