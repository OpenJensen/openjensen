# Psi-Zero worker

This adapter trains the published Psi-Zero action expert while keeping its Qwen
vision-language model frozen, matching the upstream fine-tuning recipe. It does
not offer LoRA, QLoRA, or a GGUF quantization path. The adapter still needs a real
GPU validation run before operational readiness can be claimed.

The source and weights are pinned in `environment.toml`. Provision this separate
environment on a cloud GPU through SkyPilot, not on the application host:

```sh
git clone https://github.com/physical-superintelligence-lab/Psi0.git /opt/firebird/Psi0
git -C /opt/firebird/Psi0 checkout 4f3720d45e102b36d7c3e9465ab8062274170518
cd /opt/firebird/Psi0
uv venv .venv-psi --python 3.11
UV_PROJECT_ENVIRONMENT=.venv-psi GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --group psi --index-strategy unsafe-best-match
uv pip install --python .venv-psi/bin/python flash_attn==2.7.4.post1 --no-build-isolation
```

The cloud image needs NVIDIA drivers and FFmpeg libraries. The Firebird runner
selects the official FlashAttention wheel matching the installed Torch C++ ABI
and verifies its pinned SHA-256. A CUDA toolkit and development headers are only
needed when the environment requires a source-build fallback.
Expose the Firebird worker source on `PYTHONPATH`, set `FIREBIRD_PSI_ROOT` to that
checkout, and invoke `python -m firebird_vla.psi_application REQUEST RESULT`.
Use one visible GPU with BF16 support; the catalog conservatively requests at
least 40 GB until memory is measured on a real run.

Only LeRobot v2.0/v2.1 datasets are supported by Psi's pinned upstream loader.
The dataset must have task instructions, exactly one selected camera, and
`action` plus `observation.state` (or the upstream `states` alias), each with
1–36 dimensions. Smaller vectors are padded to the released 36-dimensional
expert, and padded dimensions and terminal actions are masked from the loss.
The published 30-step action chunk is retained.

The adapter downloads only the pinned VLM and expert subdirectories and the
chosen dataset revision on the cloud worker. It splits by episode and calculates
normalization statistics from training episodes alone. Saved checkpoints include
the complete action expert, optimizer, scheduler, RNG state, deterministic data
position, recipe, split, and a fixed inference probe. The frozen VLM is referenced
by its immutable Hub revision instead of copied into every checkpoint.

Set `FIREBIRD_GCS_PREFIX` and `FIREBIRD_CHECKPOINT_EXPORT_ROOT` through the runner
to publish checkpoints to Google Cloud. Each final result is verified in a fresh
process by checking action parity. Held-out flow-matching loss is recorded as
validation loss; it is not a measurement of robot task success.

Official sources: [Psi0 repository](https://github.com/physical-superintelligence-lab/Psi0),
[released checkpoints](https://huggingface.co/USC-PSI-Lab/psi-model).
