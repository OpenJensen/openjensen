# SmolVLA training recipe

## Install on the training machine

Run from `workers/smolvla_qlora` on Linux x86_64 with Python 3.11 and an NVIDIA
driver compatible with CUDA 12.6. Use BF16 on Ampere/newer GPUs or FP16 on T4.
The requirements pin LeRobot 0.4.4 and Torch 2.7.1.

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements-smolvla-linux.txt
uv pip install --no-deps -e .
```

## Run the smoke recipe

```bash
firebird-finetune --config configs/smolvla_qlora.json --smoke --dry-run
firebird-finetune --config configs/smolvla_qlora.json --smoke
firebird-verify outputs/smolvla-qlora-smoke/checkpoint-000002
```

Choose a new output directory with `--output-dir`. The first run fetches the
pinned dataset/model files. The supplied recipe uses
`codywang/so101_pickup_test`, one front camera, six state/action coordinates and
deterministic episode splits.

## Fine-tune and resume

```bash
firebird-finetune --config configs/smolvla_qlora.json

# Resume an interrupted run from a completed checkpoint, retaining the original recipe:
firebird-finetune --config configs/smolvla_qlora.json \
  --resume outputs/smolvla-qlora/checkpoint-000100 \
  --output-dir outputs/smolvla-qlora-resumed

firebird-verify outputs/smolvla-qlora/checkpoint-020000
```

Edit the JSON recipe before a new run. `steps` counts optimizer updates. Resume
with the original recipe and precision, changing only the output directory.
Checkpoints are written beneath `outputs/smolvla-qlora/checkpoint-NNNNNN`.

## Load a checkpoint

Keep the pinned base and tokenizer/config snapshots in the Hugging Face cache,
or provide network access to fetch them. Use a raw batched LeRobot observation
with front images in `[0,1]`, state and a task list:

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

Reset the policy at each episode boundary.

## Run developer checks

```bash
uv pip install -e '.[dev]'
pytest -q
ruff check src tests
python -m compileall -q src
```

In the GPU environment, also run `pytest -q -m cuda` and the smoke/reload commands.
