# Psi-Zero worker setup

Use a cloud worker with one visible BF16-capable GPU and at least 40 GB GPU
memory, NVIDIA drivers and FFmpeg libraries. Pin sources and weights to
[environment.toml](environment.toml).

## Install

Provision the worker through SkyPilot and run:

```sh
git clone https://github.com/physical-superintelligence-lab/Psi0.git /opt/firebird/Psi0
git -C /opt/firebird/Psi0 checkout 4f3720d45e102b36d7c3e9465ab8062274170518
cd /opt/firebird/Psi0
uv venv .venv-psi --python 3.11
UV_PROJECT_ENVIRONMENT=.venv-psi GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --group psi --index-strategy unsafe-best-match
uv pip install --python .venv-psi/bin/python flash_attn==2.7.4.post1 --no-build-isolation
```

Use the official FlashAttention wheel matching the installed Torch C++ ABI;
provide CUDA toolkit headers when building from source.

## Configure and run

Select the named Psi0 interpreter and the absolute OPEN JENSEN source path:

```sh
export FIREBIRD_PSI_ROOT=/opt/firebird/Psi0
PYTHONPATH=/absolute/openjensen/workers/smolvla_qlora/src \
  /opt/firebird/Psi0/.venv-psi/bin/python -m firebird_vla.psi_application \
  /absolute/request.json /absolute/new-result.json
```

Supply a pinned LeRobot v2.0/v2.1 dataset revision with task instructions, one
selected camera and `action` plus `observation.state` (or `states`), each with
1–36 dimensions. Use the upstream 30-step action chunk.

Set `FIREBIRD_GCS_PREFIX` and `FIREBIRD_CHECKPOINT_EXPORT_ROOT` through the runner
to select the checkpoint destination. Keep the resolved recipe and episode split
with the saved checkpoint when resuming.

Sources: [Psi0](https://github.com/physical-superintelligence-lab/Psi0),
[checkpoints](https://huggingface.co/USC-PSI-Lab/psi-model).
