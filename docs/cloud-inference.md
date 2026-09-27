# Evaluate and Run on Google Cloud

The application stays on xbox-360. **Evaluate** and **Run → Check inference** dispatch an isolated
SkyPilot job for the selected SmolVLA GGUF or previously saved package. The worker
fetches the registered artifact from private GCS and verifies its manifest and
file hashes. Model weights do not download to the application host.

Evaluate performs a real fresh-process native prediction and an instrumented
benchmark with the requested warmup and timed-call counts. The report includes
p50/p95 latency, individual timing samples, sampled whole-GPU memory, memory
coverage, finite action counts, a preview of the real action channels, and the
exact model, executable, library and worker identities. Camera count follows the
artifact's camera selection. Inputs are explicit synthetic images, fixed token
IDs, state and noise, not a replay of the training dataset.

Run on a GGUF creates a package, starts another offline process, and verifies
inference from that exact copied payload before publishing the package to GCS.
Run on an existing package executes that exact package in a fresh process; it
returns a verification report without overwriting or repackaging existing
receipts. The original artifact remains unchanged in both cases.

Inference evidence is retained through teardown: fixed log, GPU CSV, action and
report paths are copied as bounded metadata to the application and durable GCS
objects. Each report lists the retained path, URI, hash, captured size and any
truncation. Logs are limited to 256 KiB each; oversized JSON is omitted instead
of being presented as complete. Returned telemetry references use relative paths
and stable GCS URIs. Original raw receipts can still record their VM execution
paths; those paths are provenance rather than portable locations.

These operations use `evaluation.mode=engine`. They establish native loading,
inference, timing and memory behavior, not robot task success. Reports keep
`success_rate` and `task_success` null, `complete_episodes` zero, and
`deployment_verified` false. The cloud engine does not claim that an SO-101
checkpoint is compatible with LIBERO. Existing operator-prepared native
LIBERO/Spatial environments remain separate.

A training checkpoint must first be quantized or exported to a supported GGUF.
Unsupported artifact formats, architectures, simulator modes and robot quality
limits are rejected before GPU allocation. Public runtime flags
`engine_evaluation` and `run` describe the supported operations independently of
`simulation`, which remains false for these GCP profiles.

The cloud build reuses the pinned vla.cpp commit and packed-weight patch from
quantization, applies the checked-in per-call latency instrumentation, and builds
CUDA prediction/benchmark binaries. CUDA 12.8 compiler/runtime/development
packages and an accelerator-specific architecture are selected on the isolated
Ubuntu VM. The shared SkyPilot deadline, cancellation, result publication and
owned-cluster cleanup apply to both operations. Runtime checks reject silent CPU
fallback and missing memory/timing evidence.

Software validation covers dispatch, admission, package routing, unsupported
modes, bounded setup and single-camera workload selection. The recorded
[SmolVLA L4 checks](jobs-first-verification.md#real-evaluate-and-run-checks) verify
the CUDA build and synthetic-input inference on one selected artifact. They do
not establish robot task success or other model/hardware combinations; fixture
tests remain separate from that live execution evidence.
