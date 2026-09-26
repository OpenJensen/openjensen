# Quantized SmolVLA fine-tuning

This branch implements an isolated Python worker for **SmolVLA QLoRA** on a
single NVIDIA GPU. It uses LeRobot's pretrained policy, dataset reader,
processors and flow-matching loss, bitsandbytes NF4 linear layers, and PEFT
LoRA. This is a first native adapter, not a universal VLA trainer or a complete
platform implementation.

**Validation status:** CPU contract tests and recipe dry-run can run locally.
Real NF4 kernels, training, peak VRAM, checkpoint reload and task quality require
the CUDA checks below. No 8 GB fit or robot/simulation success is claimed.

## Install on the training machine

Use Linux x86-64, Python 3.11, and a compatible CUDA GPU. Ampere and newer use
native BF16; T4 uses FP16 with gradient scaling. The lock uses PyTorch 2.7.1's CUDA 12.6 wheels;
the NVIDIA driver must support that runtime. Keep this worker in its own environment.
Allow disk for CUDA packages, the base policy, the approximately 1.3 GB candidate
dataset, and checkpoints, plus enough host RAM to load the base policy in FP32.

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements-smolvla-linux.txt
uv pip install --no-deps -e .
```

The direct dependencies are also pinned in `pyproject.toml` (`.[smolvla]`).
LeRobot 0.4.4 is intentionally pinned because this worker depends on its native
policy and processor APIs. No upstream source files are patched. The Linux lock
is dependency-resolved, not evidence that the CUDA runtime has been exercised.

## Run the integration smoke test first

From `workers/smolvla_qlora`:

```bash
firebird-finetune --config configs/smolvla_qlora.json --smoke --dry-run
firebird-finetune --config configs/smolvla_qlora.json --smoke
firebird-verify outputs/smolvla-qlora-smoke/checkpoint-000002
```

The smoke run performs two optimizer updates, checks for nonzero LoRA gradients,
evaluates one held-out batch, saves an adapter checkpoint and a reference action.
The separate verification process rebuilds the same quantized base, reloads the
adapter, and compares an action on the same held-out observation. It writes a
verification report beside the checkpoint; failure exits nonzero. This numerical
check is deliberately distinct from task-success evaluation.

An output directory must not already exist. Use `--output-dir` for a new run.
The first training run downloads the pinned public dataset and model files.
No weights, metrics or dataset files are uploaded, and no paid compute is started.

## Fine-tune and resume

```bash
firebird-finetune --config configs/smolvla_qlora.json

# Resume an interrupted run from a completed checkpoint, retaining the original recipe:
firebird-finetune --config configs/smolvla_qlora.json \
  --resume outputs/smolvla-qlora/checkpoint-000100 \
  --output-dir outputs/smolvla-qlora-resumed

firebird-verify outputs/smolvla-qlora/checkpoint-020000
```

Edit the JSON recipe before starting a different experiment. Steps count optimizer
updates, not microbatches. Defaults are batch size 64, accumulation 1, LoRA rank 16,
learning rate 1e-4, and 20,000 updates. The visible batch/step defaults match the
Kite SmolVLA setup; the adapter optimizer settings remain this worker's recipe.
Checkpoints are saved every five completed updates. These are starting settings,
not tuned hyperparameters or a promise of memory fit on every GPU. The final
partial batch of an epoch is retained.
Resume restores optimizer, scheduler, gradient scaler (FP16), torch/CUDA/Python RNG
state and the shuffled batch cursor, including windows skipped after gradient
overflow. It requires the original recipe and compute precision except the output directory;
extending a completed schedule is a new experiment. Exact bitwise reproducibility
across different GPU architectures or kernel versions is not promised.

The example uses `codywang/so101_pickup_test`, pinned to the revision documented
in the repository, with its front camera, six-dimensional state/action and task
strings. Episodes are split deterministically 24/6 at seed 42; normalization is
computed only from the training frames. Cameras are selected explicitly, action
chunks use the dataset FPS, and padded end-of-episode actions are masked in the
native loss. Different action units/controllers/cameras still require a compatible
dataset and downstream environment. Loss is not a success rate.

## What is quantized and trained

- Language-backbone and action-expert transformer linear weights are frozen,
  packed NF4 with double quantization. BF16 or FP16 **storage containers** preserve the
  native model's activation casts; the contained weights remain packed 4-bit.
- Expert attention and MLP projections receive trainable LoRA adapters.
- State/action/time projections are trained and saved in full precision. The
  vision encoder, embeddings and normalization weights remain unquantized and frozen.
- Autocast uses native BF16, or FP16 on T4 with FP32 trainable parameters and dynamic
  loss scaling. Gradients are unscaled before clipping; overflow skips the update
  and scheduler advancement. SmolVLA's attention-score calculation stays FP32.
  The recipe does not enable gradient checkpointing or distributed training.

The FP16 path has CPU contract coverage and optional CUDA regression tests. It has
not been validated by a T4 training run in this change.

This is adapter fine-tuning over quantized base weights. It is neither fake
quantization nor a claim that every tensor occupies four bits. The code first
loads pretrained weights on CPU, packs selected layers one at a time onto CUDA,
and never moves a full unquantized policy onto the GPU. In particular, it does not
replace the fine-tuned VLA backbone with a freshly downloaded unrelated VLM.

## Outputs and loading

Each run contains a resolved recipe, episode splits, training-only statistics,
quantized module names, GPU/parameter information, JSONL losses and peak allocated/
reserved GPU memory, run status, and `latest.json`. Failed/interrupted runs retain
earlier complete checkpoints. Data decoding occurs through PyAV, including the
candidate dataset's AV1 videos; decoder failures propagate visibly.

Each atomic `checkpoint-NNNNNN` contains PEFT weights (including trained
projections), native policy config, recipe, normalization statistics, splits,
optimizer/scheduler/RNG/cursor state, a reference action, dependency versions and
SHA-256 file hashes. SHA-256 detects corruption; it is not a signature proving origin.

The bundle is an **adapter plus an immutable base reference**, not a merged or
standalone packed policy. Reload requires the pinned base and backbone tokenizer/
config snapshots in the Hugging Face cache, or network access to obtain them.
Preprocessing is reconstructed from that pinned tokenizer, saved policy config
and training-only statistics. Use this loader, not generic `AutoModel` loading:

```python
import torch
from firebird_vla.checkpoint import load_for_inference

policy, preprocess, postprocess = load_for_inference(
    "outputs/smolvla-qlora/checkpoint-001000"
)
policy.reset()  # At each episode boundary.
# observation is a raw BATCHED LeRobot dict: front image in [0,1], state, task list.
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    action = postprocess(policy.select_action(preprocess(observation)))
```

Do not merge LoRA into the packed weights and present that as the verified
representation. Deployment/export formats and closed-loop simulation evaluation
are separate work. There is no automatic robot execution in this worker.

## Developer checks

```bash
uv pip install -e '.[dev]'
pytest -q
ruff check src tests
python -m compileall -q src
```

CPU tests exercise recipe rejection, disjoint splits, held-out leakage prevention,
finite statistics, resume batch ordering and corrupt checkpoint rejection. The
CUDA test exercises real NF4 packing, LoRA gradients, base-weight freezing and
adapter reload on a tiny network, and explicitly skips without CUDA dependencies.
On the GPU environment, run `pytest -q -m cuda` and the full smoke/reload sequence
above before treating this adapter as runtime-validated.

Upstream references:
[pinned SmolVLA implementation](https://github.com/huggingface/lerobot/tree/8fff0fde7c79f23a93d845d1a50e985de01f8b8a/src/lerobot/policies/smolvla),
[LeRobot PEFT guide](https://huggingface.co/docs/lerobot/peft_training),
[bitsandbytes quantization](https://huggingface.co/docs/transformers/quantization/bitsandbytes),
[packed floating storage](https://huggingface.co/docs/bitsandbytes/fsdp_qlora#quantized-data-storage).
