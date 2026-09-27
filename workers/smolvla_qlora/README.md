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

## Gradient accumulation and exact resume

The bundled ACT full-training adapter and SmolVLA LoRA/QLoRA worker accept
`gradient_accumulation_steps`. Public steps, learning-rate schedules, logging,
validation and checkpoint intervals count **completed optimizer updates**. Native
LeRobot's internal loop and saved training step still count microbatches; the ACT
adapter translates the two units explicitly. Other native families and Psi0.5
retain accumulation of one. This implementation is single-process (`world_size=1`).

New runs weight each microbatch mean by its actual example count in the complete
window, including short epoch tails and windows spanning epochs. Only the current
microbatch is loaded on the device; the worker does not buffer an image window.
Gradient clipping and scheduler advancement occur once per complete successful
update. A non-finite or skipped ACT update fails; it cannot silently consume the
requested update budget. SmolVLA retains its existing bounded overflow handling
and records skipped windows separately.

Checkpoints include hashed `optimization-contract.json` and
`optimization-state.json` records. They bind batch size, accumulation, weighting,
optimizer updates, consumed microbatches/examples and the sampler epoch/offset.
Incomplete windows cannot be published. Accumulated ACT resume also preserves the
saved sampler epoch when Accelerate initializes a fresh loader wrapper. Existing
SmolVLA checkpoints without these records keep their original equal-microbatch
weighting; old accumulation-one native checkpoints remain supported. Changing an
existing checkpoint's optimization contract is rejected.

The generated CPU acceptance fixture runs the production ACT hooks with native
LeRobot 0.6.2, Torch 2.11.0 and Accelerate 1.14.0. It uses a VAE/dropout ACT,
8-step predictions, 3-step execution, batch size 3 and accumulation 3 over five
training frames: one window consumes 3+2+3 examples, the next 2+3+2. An interruption
after a partial window preserves only the preceding complete checkpoint. A fresh
process resumes that checkpoint and exactly matches uninterrupted model tensors,
AdamW state and full predictions. Separate numerical tests compare weighted SGD
and AdamW updates with a large-batch reference and verify scheduler cadence,
clipping, non-finite handling and actual sampler order across epochs. These tests
do not establish large-batch equivalence for models with batch-dependent layers.

Reproduce the opt-in numerical tests with an existing compatible CPU environment:

```sh
FIREBIRD_TEST_ACCUMULATION_CPU=1 OMP_NUM_THREADS=1 \
  PYTHONPATH=src python -m pytest tests/test_accumulation_numerical.py -q
```

`tests/accumulation_native_fixture.py ROOT MODE [CHECKPOINT]` provides the bounded
native proof stages (`baseline`, `interrupted`, `resumed`), using generated local
image data and no model download. Run each stage in a fresh, offline process with
one CPU thread and an external timeout. The optional checkpoint argument permits
read-only reuse of a preserved interrupted checkpoint. Keep generated weights
outside Git. CPU evidence does not establish CUDA memory savings, full SmolVLA
training/resume, multi-GPU correctness, robot quality or cloud execution.
