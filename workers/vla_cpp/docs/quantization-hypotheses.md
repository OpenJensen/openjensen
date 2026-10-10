# Quantization hypothesis register

This historical research register records the September 26, 2026 experiments and
their proposed admission criteria. It does not describe completed application
capabilities. See the [current application workflow](../../../docs/policy-workflow.md)
and [native acceptance contract](native-acceptance.md) for supported operations
and remaining verification requirements.

## Patched-runtime follow-up: 2026-09-26

The [SmolVLA patch](../patches/README.md) resolves the original runtime rejection. All four candidates now complete 20 timed CPU predictions with finite outputs. Runtime allocation confirms 224 packed LM matrices and, for vision variants, 72 packed vision matrices. BF16 produces identical fixed-input action values to the unpatched runtime at the harness's printed precision.

Observed peak process RSS falls from 1192.65 MiB (floating reference) to 908.34 MiB (LM Q8), 757.35 MiB (LM Q4), 831.91 MiB (LM+vision Q8), or 681.93 MiB (LM Q4 + vision Q8). In the initial sequential run the corresponding p50 values are 4269.3, 4436.6, 4664.3, 4995.4 and 5218.2 ms; these measurements do not establish a quantization speedup. A separate floating-reference control checks run-order drift. Full evidence is in `artifacts/docker/runs/smolvla-packed-v2/` and `smolvla-packed-control/`.

H1–H4 now pass runtime compatibility and show memory savings. Their task-quality hypotheses remain untested. Synthetic action differences are diagnostic only, not a substitute for paired LIBERO success rates. No deployment winner is selected.

## Original unpatched screen: 2026-09-26

On the Docker CPU lane, the SmolVLA floating reference loads and produces predictions (1074.18 MiB; exploratory p50 4637.7 ms over five timed calls). H1–H4 produce smaller files (792.93, 642.93, 716.99 and 566.99 MiB respectively), but **all fail the runtime-load gate** with `gguf unsupported dtype`. The pinned vla.cpp v0.3.0 SmolVLA GGUF reader accepts F32/BF16, rejecting Q8_0/Q4_0. This is a runtime compatibility finding, not a measured task-quality failure. H0 task success remains unmeasured. No quantized artifact qualifies for deployment. Full local logs, audits and results are in `artifacts/docker/runs/smolvla-screen-v1/`; see [benchmark protocol](quantization-benchmark.md).

## Decision rule

The proposed selection rule chooses the **smallest candidate that passes** the
benchmark gate for the selected VLA, dataset, simulator task and hardware lane.

Proposed experiment acceptance gate:

- completes all smoke rollouts;
- loses no more than **5 absolute percentage points** of closed-loop LIBERO success versus the same model's BF16 reference;
- is smaller or faster than the BF16 reference; and
- loads within the target's memory budget.

The proposed report retains every rejected candidate and the reason it failed.
The 5-point threshold is an experiment constraint, not a universal robotics-safety claim.

## Experiments, in order

| ID | Hypothesis | Candidate | What falsifies it |
|---|---|---|---|
| H0 | The pipeline is reproducible. | BF16 GGUF baseline | Cannot create or complete fixed-seed rollouts. |
| H1 | LM Q8 is effectively lossless. | LM Q8_0; vision/action expert float | Success drops >5 points or no useful size/latency gain. |
| H2 | LM Q4 is the smallest safe CPU artifact. | LM Q4_0; vision/action expert float | Fails success gate. |
| H3 | Vision Q8 is worth its additional compression. | LM Q8_0 + vision Q8_0; action expert float | Incremental memory/latency gain does not justify success loss. |
| H4 | Aggressive full-front-end packing remains usable. | LM Q4_0 + vision Q8_0; action expert float | Fails success gate. |
| H5 | QLoRA makes training feasible on constrained CUDA GPUs without reducing final deployment quality. | 4-bit LM + LoRA training, then merge → PTQ | Final policy misses BF16/LoRA quality gate. |
| H6 | QAT recovery rescues a rejected INT4 candidate. | Short fake-quant recovery training | Does not beat the corresponding PTQ candidate. |

## Historical proposed workflow

The original proposal was:

1. Run H0–H4 automatically for SmolVLA on one selected task and hardware lane.
2. Choose the first passing artifact in this preference order: `lm_q4_vision_q8` → `lm_q4` → `lm_q8_vision_q8` → `lm_q8` → `bf16`.
3. Return the selected GGUF, target runtime configuration, evidence report, videos, and an explicit fallback to BF16 if no compressed candidate passes.
4. Expose H5 and H6 as **experimental compiler profiles**, not product claims, until they pass the same closed-loop gate on an RTX 3070 or GCloud lane.

## Design rationale

- The action expert/output head stays FP16/BF16 in H0–H4: it is the highest-risk control component.
- Vision packing is measured independently instead of being silently enabled.
- Fine-tuning remains editable PyTorch/LeRobot training; GGUF is only generated after training for deployment.
- Every decision is reproducible from the dataset revision, model revision, command manifest, hardware profile, artifact hash, and simulator seeds.
