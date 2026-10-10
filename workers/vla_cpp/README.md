# Standalone SmolVLA quantization module

This worker includes the CPU benchmark layer; see
[CPU setup and measurement procedures](docs/cpu-benchmark.md). The module API and
`policykit-worker` command below remain available. The added `policykit` command
runs the benchmark harness.

This package converts an F32/BF16 SmolVLA GGUF into LM Q8_0/Q4_0, optionally with
vision Q8_0, while preserving the action expert, projectors, embeddings, norms and
other protected tensors. It includes the pinned vla.cpp packed-weight loader patch
and a versioned subprocess worker. It has no dependency on the OPEN JENSEN core.

CPU benchmarking and RTX 3070 experiments reuse the shared conversion worker.
Training lives in [`workers/smolvla_qlora`](../smolvla_qlora/README.md).
The application connects these workers through `vla_platform.lifecycle`
and shared project jobs; see the [application workflow](../../docs/policy-workflow.md).
The standalone quantization protocol remains available for experiments.

## Experimental CUDA and NVIDIA quantization

Optional CUDA tools provide isolated engine/rollout experiments and ModelOpt
AWQ/SmoothQuant pilots. Use the [GPU setup guide](docs/gpu-setup.md).
ModelOpt full-policy export and packed runtime deployment are unsupported; a
numerical calibration diagnostic does not produce a deployable policy.

## Application acceptance

The [native acceptance contract](docs/native-acceptance.md) describes the required
GGUF checks, complete memory coverage, Linux runtime fingerprints and fresh-process
package verification. Its CPU regressions are separate from hardware/model evidence.
The separate [LIBERO Spatial adapter](docs/spatial-application.md) shares the
native/CPP observation implementation, verifies offline policy assets and uses
paired native BF16/floating controls. Hardware acceptance remains a required run.

## Install

Run inside `workers/vla_cpp` with Python 3.11 or 3.12:

```sh
uv sync --locked --extra quantize --extra test
uv run policykit-worker --help
uv run python -m pytest -q
```

The benchmark container builds from this directory and runs `policykit`.
The standalone conversion entry point remains `policykit-worker`. Mount a
prepared native checkout, floating checkpoint and writable job/output directory;
the image does not download models or build a native inference runtime.

## Native preparation and contract

Prepare vla.cpp at commit `52439f7c6c362d7bee218b400b9080cc32d75cc3` and apply
`policykit/patches/vla-cpp-smolvla-packed.patch`. The same patch ships in installed wheels as a package resource. Then inspect the exact runtime identity:

```sh
uv run policykit-worker --describe-runtime /absolute/path/to/vla.cpp
```

The returned identity contains the native commit, actual patch hash, upstream
quantizer hash, this package's Python-source hash, and installed NumPy/GGUF versions.
Unexpected native edits fail preparation. Resolve the identity again after changing
worker code or dependencies; identities from another source revision are not interchangeable.

A caller writes `job.json` inside a unique output directory and invokes:

```sh
uv run policykit-worker --job /absolute/path/to/run/job.json
```

Version-one envelope:

```json
{
  "schema_version": 1,
  "run_id": "run-example",
  "attempt_id": "attempt-1",
  "output_dir": "/absolute/path/to/run",
  "recipe_sha256": "SHA256_OF_CANONICAL_RECIPE_JSON",
  "recipe": {
    "schema_version": 1,
    "operation": "policy.quantize",
    "backend": "vla_cpp_smolvla",
    "policy": {"architecture": "smolvla"},
    "source": {"path": "/absolute/path/to/float.gguf", "sha256": "SOURCE_SHA256"},
    "runtime": "REPLACE_WITH_DESCRIBE_RUNTIME_OBJECT",
    "target": "local-cpu-conversion",
    "language": "Q4_0",
    "vision": "Q8_0"
  }
}
```

The example contains placeholders. Canonical JSON uses sorted keys, separators
`(',', ':')`, and `allow_nan=False`; hash its UTF-8 bytes. Callers should also carry
checkpoint revision and observation/action semantics in `policy`; the worker checks
the architecture and source bytes, while the application owns semantic admission.
Use `null` for `vision` to retain its original precision.
All input tensors must be F32 or BF16. F16 tensors, including F16 in protected
groups or mixed-precision masters, are rejected before conversion because the
pinned converter and native loader do not consistently support them.

The worker checks source/runtime identity before and after packing, checks tensor
shapes/precisions and unchanged protected values, and atomically publishes
`bundle/`. It writes `events.jsonl`, `result.json`, a precision audit, hashes,
lineage, packed-weight bytes and GGUF bytes. Failure returns a nonzero exit code.
Callers own scheduling, timeout, cancellation, persistence and artifact registration.
An abruptly killed worker can leave an unpublished staging directory for inspection.

For direct experimental use, `policykit-quantize --in ... --out ...
--vendor-script ... --type Q4_0 --vision-type Q8_0` exposes the original quantizer.
Use the worker for identity checks and atomic publication; the direct CLI writes
its output and audit directly.

## Evidence boundary

Output is `quantized_weights` with `evidence_scope: conversion_only`, null task
success and `deployment_verified: false`. It is not a deployment selection or a
complete policy package. Native checkpoint conversion, QLoRA adapter merging,
training, inference benchmarks and closed-loop evaluation are separate operations.

The module tests perform real packing of small synthetic GGUF tensors, verify all
four precision recipes and protected values, and reject changed inputs/runtime
identities without publishing a bundle. Their minimal test decoder is not evidence
of a complete SmolVLA inference runtime. Follow the CPU checks in the
[measurement procedure](docs/cpu-benchmark.md); CUDA and robot-task quality require
separate validation.
