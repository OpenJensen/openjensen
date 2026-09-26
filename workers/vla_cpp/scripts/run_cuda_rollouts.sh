#!/usr/bin/env bash
# Run inside firebird-quant-sim with --gpus device=0 and the workspace mounted.
set -euo pipefail
cd /workspace
run="${1:-smolvla-cuda-libero-v1}"
for preset in float_reference lm_q8 lm_q4 lm_q8_vision_q8 lm_q4_vision_q8; do
    python3 -m policykit.cuda_rollout \
        --preset "$preset" --task-id 0 --init-state-id 0 --seed 42 \
        --steps 500 --action-steps 4 --port 5560 --run "$run" \
        --manifest artifacts/cuda/runs/smolvla-cuda-v1/results.json \
        --runtime-build artifacts/cuda/build --output-root artifacts/cuda/runs \
        --tokenizer /workspace/artifacts/cuda/modelopt-metadata --require-cuda
done
