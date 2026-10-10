# Export an ACT inference package

## Install

Use uv 0.12.19 and Python 3.12. The worker lock selects LeRobot 0.6.1,
Torch 2.11.0, torchvision 0.26.0 and safetensors 0.8.0.

```sh
cd workers/act_optimizer
uv sync --locked --python 3.12 --extra cpu --extra dev
uv run --no-sync firebird-act-export /absolute/original-checkpoint /absolute/new-export \
  --timeout 120 > /absolute/export-receipt.json
```

Choose an output path that is new and outside the original checkpoint directory.

## Prepare the checkpoint

Supply a complete FP32 ACT checkpoint with ResNet18, one RGB camera, six
state/action coordinates, MEAN_STD normalization, one observation and
`1 <= execution <= prediction <= 1024`. Use saved RGB dimensions from 32 through
2048 with at most 2,073,600 pixels. Include `config.json`, `model.safetensors`,
pre/postprocessor JSON and all referenced statistics files. Use regular files
and preserve optional `temporal-contract.json` and `control-contract.json`.

The output directory contains `manifest.json`, `recipe.json`, `parity.json` and
the inference policy. Reload it in a new process:

```sh
uv run --no-sync python -m firebird_act.probe /absolute/new-export /absolute/new-probe.json
```

Use a new result path outside the package.

## Configure application export

Set `act_export_python` to the pinned interpreter and `act_export_root` to this
worker directory in the local runtime configuration. Materialize and register a
complete ACT training checkpoint, then choose **Export ACT inference package**
in the Fine-tune checkpoint monitor.

The fixed invocation is `python -m firebird_act.application REQUEST.json RESULT.json`.
Supply `schema_version: 1`, `job_id`, `operation: "policy.export"`, an absolute
`output_dir` and `artifact` with `id`, `format: "training_checkpoint"`, absolute
`path` and `manifest_sha256`. Place the new result file directly in `output_dir`.
The application package is written to `output_dir/inference-export/` with its
policy, lineage, verification and manifest files.

## Run tests

```sh
uv run --no-sync python -m pytest -q -rs
uv run --no-sync ruff check --target-version py312 src tests
uv run --no-sync ruff format --check src tests
uv run --no-sync mypy --config-file pyproject.toml src --follow-imports=silent
```

## Verify independent prediction and execution horizons

Use the separately pinned LeRobot 0.6.2 producer environment at revision
`e595b7902714ba51f91e47523f66f89c5181b649` with Accelerate 1.14.0. From the
repository root, select that producer interpreter explicitly and use one new
scratch directory:

```sh
export PYTHONPATH="$PWD/workers/smolvla_qlora/src"
/absolute/producer/bin/python workers/act_optimizer/scripts/native_checkpoint_fixture.py generate /absolute/new-scratch --prediction-horizon 8 --execution-horizon 3
/absolute/producer/bin/python workers/act_optimizer/scripts/native_checkpoint_fixture.py resume /absolute/new-scratch --prediction-horizon 8 --execution-horizon 3
/absolute/producer/bin/python workers/act_optimizer/scripts/native_checkpoint_fixture.py bundle /absolute/new-scratch --prediction-horizon 8 --execution-horizon 3
```

Export through `firebird_act.application`, then pack through
`firebird_quant.native_application`. With the existing 0.6.1 CPU interpreter,
set `PYTHONPATH` to `workers/act_optimizer/src:workers/firebird_quant/src:workers/isaac_sim`
and enable offline execution and one CPU thread:

```sh
workers/act_optimizer/.venv/bin/python workers/act_optimizer/scripts/temporal_http_fixture.py /absolute/export/policy /absolute/packed/verification.json /absolute/new-float.json
workers/act_optimizer/.venv/bin/python workers/act_optimizer/scripts/temporal_http_fixture.py /absolute/packed/policy /absolute/packed/verification.json /absolute/new-packed.json --packed
workers/act_optimizer/.venv/bin/python workers/act_optimizer/scripts/temporal_replay_fixture.py /absolute/packed /absolute/new-replay-proof
```
