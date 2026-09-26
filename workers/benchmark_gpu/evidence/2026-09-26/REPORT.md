# GPU measurements — 2026-09-26

Partial evidence snapshot. The full comparison is still running; no winning stack is selected.

## L4: uncontended compute window

The authorized collection pause lasted about eight minutes. All eight GPU workers resumed. Ten pre-run utilization samples were 0%. Paused collectors retained 2,043 MiB; benchmark memory below is process-scoped and excludes them. All eight candidates and the BF16 repeat completed 14 identical fixtures, with one warmup and five samples per fixture.

| Candidate | Median of fixture p50 (ms/chunk) | Cached startup (s) | Sampled process peak (MiB) |
|---|---:|---:|---:|
| native-bf16 | 348.80 | 27.79 | 1186 |
| cpp-bf16 | 100.28 | 2.28 | 1162 |
| native-fp16 | 347.36 | 27.95 | 1186 |
| cpp-Q8_0 | 99.34 | 3.06 | 1018 |
| native-int8 | 418.20 | 29.16 | 1036 |
| cpp-Q4_0 | 101.62 | 2.84 | 946 |
| native-nf4 | 386.49 | 29.29 | 950 |
| cpp-Q8_0-vision | 101.02 | 2.84 | 944 |
| native-bf16-repeat | 350.64 | 27.92 | 1186 |

The startup boundary excludes imports and downloads. Timing includes full preprocessing, all denoising steps, postprocessing and CPU output, plus C++ RPC. Sampling at 100 ms can miss brief memory peaks. These values are specific to this GPU and run window.

## RTX 3070: completed paired quality

| Candidate | Successes / 20 | Median of fixture p50 (ms/chunk) |
|---|---:|---:|
| cpp-Q4_0 | 8/20 | 110.92 |
| cpp-Q8_0 | 15/20 | 109.30 |
| cpp-bf16 | 14/20 | 112.72 |
| native-bf16 | 15/20 | 325.29 |
| native-fp16 | 15/20 | 312.30 |
| native-int8 | 15/20 | 372.32 |
| native-nf4 | 15/20 | 377.31 |

Twenty paired episodes cover ten LIBERO Spatial tasks and two fixed initial states. This is a small behavior-retention sample, not a quality-equivalence or generalization claim. RTX is a shared WSL desktop; its timings are separate from the uncontended L4 window.

## C++ action fidelity

Across the 14 shared fixtures, C++ BF16 versus native BF16 had RMSE 0.005298 and maximum absolute error 0.339498. The largest error was a synthetic gripper value; gripper signs agreed on all fixtures. Real-observation maximum error was 0.013786. Native FP16 versus BF16 real-observation maximum error was 0.012943. These are numerical differences, not exact parity. See the raw parity summary for channel-specific metrics.

## Engine integration

vLLM 0.9.2 custom V0 pooling execution matched same-environment native BF16 exactly on three fixtures. This uses native PyTorch policy kernels inside vLLM and requires the pooling-input forwarding patch. The probe timings were contention-affected and are excluded from the table. Full shared-fixture and paired quality execution is in progress.

TensorRT-LLM 0.21.0 complete-action execution passed through a custom PyTorch-executor adapter with the context-logit-capable sampler enabled. All three probe fixtures matched same-environment native FP16 exactly. Probe timings had contention and are excluded from the table. Full paired quality and isolated timing are in progress; INT8/INT4 engine execution remains unvalidated.

Remaining evidence: the other RTX quantized quality rows, vLLM paired quality and isolated timing, TensorRT-LLM complete-action validation, and L4 paired quality. Historical smoke results with different noise handling are excluded from these quality rows.

RTX TensorRT-LLM setup is deferred by user decision because no disk with enough free space was available. See the worker README for the measured storage constraint.

The completed RTX C++ Q4 run scored 8/20 against native BF16's 15/20, a substantial observed quality loss. The draft does not treat reduced memory alone as acceptance.
