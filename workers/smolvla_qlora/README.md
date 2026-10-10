# SmolVLA worker setup

## Install developer tools

From `workers/smolvla_qlora`, use Python 3.11:

```sh
uv venv --python 3.11
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
.venv/bin/firebird-finetune --help
.venv/bin/firebird-verify --help
.venv/bin/firebird-check-qlora --help
```

For the CUDA environment, smoke run, training, resume and inference loader,
follow [SmolVLA training setup](docs/training/smolvla-qlora.md).
For other native model fixtures, use [QLoRA checks](docs/training/multimodel-qlora-checks.md).

## Set training controls

Set `gradient_accumulation_steps` in the recipe. Count `steps`, learning-rate
schedules, checkpoint intervals and validation intervals in completed optimizer
updates. Resume with the same saved recipe, accumulation value and batch size.
Use [temporal settings](TEMPORAL.md) for prediction and execution horizons.

For native ACT simulator data, supply a complete local snapshot containing
`meta/firebird-demonstrations.json`, with matching joint order, radians, camera
and FPS. Preserve `control-contract.json` at both checkpoint levels when resuming.

## Run numerical tests

Use an existing compatible CPU environment:

```sh
FIREBIRD_TEST_ACCUMULATION_CPU=1 OMP_NUM_THREADS=1 \
  PYTHONPATH=src python -m pytest tests/test_accumulation_numerical.py -q
```

Run `tests/accumulation_native_fixture.py ROOT MODE [CHECKPOINT]` in `baseline`,
`interrupted` and `resumed` modes with generated local image data, an external
timeout and one CPU thread. Keep the scratch directory outside Git.
