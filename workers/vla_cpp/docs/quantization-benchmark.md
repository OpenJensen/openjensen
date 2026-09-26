# Quantization hypothesis benchmark

## Current evidence

The packed-runtime follow-up is `artifacts/docker/runs/smolvla-packed-v2/REPORT.md`.
It preserves the original v1 failures, reuses the exact same artifact hashes, and
adds packed resident-tensor verification, BF16 action regression, loading time,
sampled peak process RSS, and 20-call latency measurements. The runtime fix is
versioned as [a source patch](../patches/README.md), automatically applied by the
configured preparation step. All numerical action comparisons use the 350 real
control values, excluding padded channels, and fixed synthetic inputs/noise.

The actual local report is `artifacts/docker/runs/smolvla-screen-v1/REPORT.md`, with machine-readable evidence in `results.json` next to it. Missing values mean **unmeasured**, never zero. The report is updated after every candidate.

The first lane is Docker Linux CPU on the Intel Mac. It does not establish CUDA or Jetson performance. `policykit.screen` compares one immutable SmolVLA checkpoint across a floating reference and four component-specific PTQ candidates. It records conversion logs, source/runtime commits, artifact SHA256, quantization duration, exact inference commands, tensor precision audits, size, and synthetic inference latency.

| Hypothesis | LM | Vision | Action expert/projector | Evidence required |
|---|---|---|---|---|
| H0: usable reference | Source float | Source float | Source float | Inference works and baseline solves task |
| H1: Q8 compression retains quality | Q8_0 | Source float | Source float | Size/latency improvement and paired task success |
| H2: Q4 improves the deployment tradeoff | Q4_0 | Source float | Source float | Same gates as H1 |
| H3: vision packing adds value | Q8_0 | Q8_0 | Source float | Incremental benefit versus H1 with acceptable quality |
| H4: combined compression wins | Q4_0 | Q8_0 | Source float | Benefit versus H2/H3 with acceptable quality |

Source precision is audited, not inferred from a filename. This checkpoint contains BF16 and F32 tensors. All compressed candidates derive directly from that reference. The PolicyKit quantizer uses an allowlist of LM/vision block matrices, preserving `aex.*`, action/state projections, embeddings, norms and the multimodal projector. The upstream generic quantizer's skip list does not protect `aex.*`; do not use it directly for this matrix.

## Reproduce the engine screen

The prepared Linux Python environment is mounted at `artifacts/docker/tooling`. It contains CPU PyTorch, gguf, safetensors, Hugging Face Hub, PyYAML and pytest. The source checkpoint and `policykit-source.json` must be fully downloaded under `artifacts/docker/sources/smolvla`.

```bash
docker compose run --rm --entrypoint artifacts/docker/tooling/bin/python policykit -m pytest -q
docker compose run --rm --entrypoint artifacts/docker/tooling/bin/python policykit -m policykit.screen --reps 5 --timeout 900
```

The screen reserves a fresh evidence directory before reading inputs or running
conversion. The default remains `smolvla-screen-v1`, also used by the packed benchmark.
An existing directory is rejected without modifying its reports, logs or models.
For another independent screen, pass `--run smolvla-screen-v2` (or another unused
name); this does not change the packed benchmark's default input path.

The screen uses four CPU inference threads, two 512-pixel synthetic camera images, one warmup and five timed predictions per candidate. It runs sequentially to avoid candidates competing for the same CPU. This small screen is exploratory; rerun finalists with more repetitions and independent processes before comparing tail latency. Download/conversion time is not counted as quantization time. Camera capture, preprocessing and robot I/O are not included in engine latency.

## Product admission gate

An engine-only result never selects a deployable winner. Next run the floating reference and viable candidates on identical LIBERO task IDs, seeds and checkpoint-appropriate action chunking, saving separate per-phase/per-task/per-seed logs and videos. Stop candidates that fail execution or an explicitly configured smoke-quality threshold.

Before selection, require an explicit absolute minimum task-success target, the allowed drop versus the floating reference (initial hypothesis: 5 percentage points), the target memory budget and the maximum end-to-end latency. Report paired episode outcomes and uncertainty; a small smoke suite cannot establish a 5-point quality difference. Re-evaluate finalists with enough episodes to support the decision. Choose the fastest candidate meeting every gate; use size as a secondary objective, and report the size/latency frontier rather than assuming lower bit-width is faster.

The existing `BenchmarkRunner`/selector needs those admission gates and per-seed evidence isolation before being used to declare a product winner. Its current smoke pass only establishes execution, and its selector only checks relative success and artifact size.

AWQ, GPTQ, QAT and CUDA-specific runtimes are untested here. They require additional candidate generators and the same task-quality contract; this screen cannot rank them.
