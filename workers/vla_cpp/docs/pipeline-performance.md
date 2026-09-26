# Measuring quantization in the VLA pipeline

Quantization changes numerical precision. Measure both task quality and execution
cost against the same floating master, on the same hardware, inputs and runtime.
Do not infer robot-task quality from weight size, action MAE, or language loss.

## Measurements

- **Quality:** closed-loop success rate on identical tasks and seeds; report the
  baseline minus candidate in absolute percentage points, with per-episode
  outcomes and uncertainty. Fixed-input action MAE per real control channel is a
  useful diagnostic, but is not a task-success estimate. Padded channels are excluded.
- **Inference:** warm observation-to-action model-call p50/p90, excluding warmups;
  model loading and benchmark process wall time are separate. A call may produce
  an action chunk, so its reciprocal is not necessarily the robot control rate.
- **Quantization:** elapsed time from invocation through writing and auditing the
  output artifact. Python API audits include imports, input reading, tensor
  packing, output writing and precision verification, excluding audit JSON writing.
  CLI timings additionally include subprocess startup and audit JSON writing.
- **Module cost:** uncached checkpoint conversion + quantization + artifact
  hashing and orchestration. Download, runtime build, and evaluation are outside
  this boundary. Existing floating conversion is marked reused and costs zero
  *in this invocation*, not historically.

`policykit quantize` writes `<artifact>.timing.json`, including the hardware,
output hash, cached-conversion flag, conversion identity and phase times. Cache reuse
requires a complete `<floating-artifact>.conversion.json` whose source-file hashes,
source revision/settings, converter and helper hashes, dependency versions and output
hash all match. Missing provenance or changed inputs rebuild the floating reference
through a temporary file; failed conversion leaves the old reference intact. A stale
prepared source/revision requires `prepare` again. Input hashing and cache validation
are measured separately as `conversion_cache_check_seconds` and included in module
wall time, even when conversion itself is reused. Failed stages retain elapsed
time with `status: failed`. A timed stage that never ran is absent, not zero.
The benchmark accepts preparation timings only when status is complete and the
artifact hash matches, preserves all three independent process logs and wall
measurements, and saves baseline comparisons. Its p50/p90 values are medians of
three per-process summaries, not pooled sample quantiles. Actual load time is
available only when the runtime emits `load_ms`; it is not inferred from sleeps.

The leaderboard displays quantization, conversion, total module wall time,
loading, warm inference, speedup, task-success drop, and break-even calls.
Missing measurements remain unknown. Failed candidates do not get a speedup or
quality comparison. The existing deployment selector's limited gates remain as
documented in `quantization-benchmark.md`; these measurements alone do not certify
a candidate for deployment.

## Summarize existing measured results

Run from the repository root:

```bash
uv run python -m policykit.performance_report \
  --results artifacts/docker/runs/smolvla-packed-v2/results.json \
  --out artifacts/docker/runs/smolvla-performance \
  --calls 1000
```

This produces `PERFORMANCE.md` and `performance.json` without changing the input
run. The packed-v2 run reuses quantization durations measured during screen-v1;
they are not fresh quantization samples and do not estimate variance. Its measured
inference duration comes from the patched packed runtime. Rerun the summarizer
if the input benchmark was still in progress when it was generated.

For N calls with a reused, already-loaded model:

```
quantized cost = quantization_seconds + N * quantized_inference_seconds
floating cost = N * floating_inference_seconds
break_even_calls = ceil(quantization_seconds / (floating_seconds - quantized_seconds))
```

The last expression applies only when quantized inference is faster. These are
estimates using median call duration, not measured end-to-end batch times. Add
separately measured conversion, loading, calibration (where applicable),
evaluation and robot I/O for the complete pipeline budget. If candidates are
built sequentially, add their preparation costs; if in parallel, measure the
actual pipeline wall time instead of summing worker times. Quantize once per new
checkpoint, then reuse the artifact for inference.

For reliable comparisons, run candidates sequentially without competing jobs,
use warmups and multiple independent repetitions, and repeat on deployment
hardware. GPU timing must wait for device completion. Synthetic CPU results do
not establish GPU speed or robot-task performance.

## Output agreement and accuracy

The module now emits three distinct measurements: quantization duration, model
inference duration, and output fidelity. Future `policykit.packed_bench` runs store
`output_fidelity` directly in every candidate result. For existing saved runs:

```bash
uv run python -m policykit.output_fidelity \
  --run-results artifacts/docker/runs/smolvla-packed-v2/results.json \
  --out artifacts/docker/runs/smolvla-performance \
  --atol 0.001 --rtol 0.01
uv run python -m policykit.performance_report \
  --results artifacts/docker/runs/smolvla-packed-v2/results.json \
  --fidelity artifacts/docker/runs/smolvla-performance/output-fidelity.json \
  --control-results artifacts/docker/runs/smolvla-packed-control/results.json \
  --out artifacts/docker/runs/smolvla-performance
```

Fidelity measures MAE, RMSE, maximum and p95 absolute error, exact-value match
fraction, tolerance match fraction, whole-step agreement and per-channel metrics.
The 25 padded channels are excluded: these SmolVLA arrays contain 50 steps x 7
real channels. Exact means equal at the saved precision (9 significant digits),
not a binary comparison of model tensors. A float-reference self-comparison is a
sanity check, not an independent accuracy result.

A value agrees within tolerance when:

```
abs(candidate - reference) <= atol[channel] + rtol * abs(reference)
```

The defaults (absolute 0.001, relative 0.01) are explicit numerical diagnostics.
They are not calibrated robot acceptance criteria. Supply one absolute tolerance
per real channel with `--atol` when channel units/scales differ. Aggregate errors
mix units; inspect the per-channel JSON. Quantized outputs need not match exactly.
A tolerance pass does not establish correct or safe behavior.

To check a representative dataset, export matched observations through the
floating and quantized models with identical preprocessing, state, prompt and
noise, then pass a JSON file to `--pairs`. Every case has a unique ID and 2D
`[steps, real_channels]` arrays; remove padded channels before exporting:

```json
{
  "cases": [
    {
      "id": "episode-001/frame-010",
      "reference": [[0.1, 0.2]],
      "candidate": [[0.101, 0.198]],
      "targets": [[0.1, 0.19]]
    }
  ]
}
```

The numbers above only demonstrate the format, not measured model output.
`targets` is optional. When provided, the report separately compares each model
with target actions (MAE/RMSE/max, per channel, and the change in error). It does
not equate agreement with the floating model to ground-truth accuracy. Without
targets, target error is null. Neither offline metric replaces closed-loop task
success on matched simulator tasks/seeds.

The current evidence covers only one synthetic observation and fixed noise,
not a representative dataset. All four quantized variants differ from the
floating output. The repeated floating baseline also changed inference p50 by
about 26%; the combined report flags this run-order drift. Current latency ratios
cannot establish a reliable quantization-induced speed change.
