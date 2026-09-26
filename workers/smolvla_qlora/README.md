# Isolated SmolVLA LoRA / QLoRA worker

Native SmolVLA quantized fine-tuning, adapter verification, and diagnostics for
multiple native model runtimes. This project uses its own Python 3.11 environment;
ML dependencies are separate from the application's Python 3.14 environment.

Related task: [TRAIN-001 (#41)](https://github.com/sobhanb-eth/firebird-hackathon-codebase/issues/41).
The branch targets codebase `main`. The integration branch exposes LoRA and QLoRA
as methods under the same application fine-tuning operation, with pinned dataset
lineage, resource preflight, checkpoint resume and native export. See the
[workflow guide](../../docs/policy-workflow.md) and [integration evidence](../../docs/workflow-validation.md).
Capabilities appear only when the operator configures a training environment.

## Install and test

Run all worker commands from `workers/smolvla_qlora`:

```sh
uv venv --python 3.11
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
.venv/bin/firebird-finetune --help
.venv/bin/firebird-verify --help
.venv/bin/firebird-check-qlora --help
```

These lightweight checks do not install training dependencies. On a supported
Linux CUDA host, follow the pinned installation and native-runtime instructions in
[SmolVLA training](docs/training/smolvla-qlora.md). See
[multi-model diagnostics](docs/training/multimodel-qlora-checks.md) for the catalog,
backend-specific preparation, and explicit unsupported states. Keep outputs,
weights and datasets outside Git.

## Evidence and limits

The source suite previously passed 63 tests on the RTX 3070 host, including a real
CUDA NF4 gradient, freeze, save and reload test using a tiny synthetic policy.
That is not a full SmolVLA training or robot-quality result. See the
[execution report](https://github.com/sobhanb-eth/firebird-hackathon-prep/blob/a9a56ea105d061012abb79fe937d4237af2bf33e/docs/tasks/evidence/2026-09-26-rtx3070-validation.md)
for the exact source revisions and environment. `TRANSFER_ORIGIN.json` records the
source files relocated into this worker; hardware evidence is historical until
rerun on this branch. Planning and task records remain in the prep repository.
