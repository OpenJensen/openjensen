#!/usr/bin/env bash
set -euo pipefail

[[ "${POLICY_RUNTIME:-}" == packed-act-cpu && "${POLICY_DEVICE:-}" == cpu ]]
[[ "${POLICY_MODEL_FORMAT:-}" == firebird_quant ]]
[[ "${POLICY_JOB_TIMEOUT:-}" == 4500 ]]
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$HOME/sky_workdir:$HOME/firebird-quant-src:$HOME/firebird-act-src"

# Pin Torch's own pools as well as BLAS; this is a bound, not a speed claim.
exec timeout --signal=TERM --kill-after=60 "$POLICY_JOB_TIMEOUT" \
  "$HOME/vla-env/bin/python" -c '
import runpy
import torch
assert torch.version.cuda is None, "Packed ACT requires CPU-only Torch"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
runpy.run_module("sim_worker.rollout.server", run_name="__main__")
' --host 0.0.0.0 --port 8080 --backend lerobot --device cpu \
  --checkpoint "$HOME/vla-checkpoint" --model-id "$MODEL_ID" \
  --state-dim "$POLICY_STATE_DIM" --action-steps "$POLICY_ACTION_STEPS" \
  --camera-key "$POLICY_CAMERA_KEY"
