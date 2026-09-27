# Jobs-first workflows and cloud inference verification

Verified on 2026-09-27 with the application running on Xbox and separate
SkyPilot-managed NVIDIA L4 workers in GCP. Model weights remained in GCS and on
the disposable workers.

## User flows

Fine-tune, Quantize, Evaluate and Run open saved jobs first. New jobs have their
own forms; choosing a saved job opens its status, input, results and outputs.
Training drafts retain the chosen GPU, method and cameras through Compute settings.
Dataset and checkpoint shortcuts open the appropriate new-job form. Model cards
show bold GPU budgets; these are not model download sizes.

## Real Evaluate and Run checks

Both jobs were submitted using the actual buttons in Chrome, selecting the
existing SmolVLA Q4 policy derived from the trained step-100 checkpoint.
Their implementation provenance is `b0b7c85` (jobs-first workflows and cloud
inference). Commit `d892956` adds preparation-progress reporting and this live
verification record; it does not expand the GPU execution scope.

| Measurement | Evaluate | Run |
| --- | ---: | ---: |
| Backend | CUDA / NVIDIA L4 | CUDA / NVIDIA L4 |
| Warmups / measured calls | 3 / 10 | 3 / 10 |
| Median native prediction | 51.25 ms | 51.93 ms |
| p95 native prediction | 52.78 ms | 53.38 ms |
| Sampled peak GPU memory | 1,038 MiB | 1,038 MiB |
| Finite action outputs | 1,600 | 1,600 |
| Real action channels / padded channels | 6 / 32 | 6 / 32 |
| Camera count / image size | 1 / 512 px | 1 / 512 px |
| Fresh-process reload | Passed | Passed |
| Whole job, including preparation and cleanup | 35.5 minutes | 36.8 minutes |

These latency figures measure native predictions after warmup. They exclude
provisioning, source compilation, artifact transfer and package publication.
Fresh workers currently rebuild the CUDA runtime; that dominates startup and is
a remaining performance limitation. These tests do not establish robot or
simulator task success, and they do not compare Firebird with KiteML performance.

The selected GGUF SHA-256 remained
`d64ebd5847efbbb8efbc7d166b4b50a202804a255f1204908e7a0407197fa19d`.
The two independent workers produced the same six first-step action values.
Run created a reload-verified package with 1,070,404,383 payload bytes and manifest
SHA-256 `69f9ae042c6d2738dcca15611a93a8642ef4d8e1f13f6425f0fb19d976055124`.

Bounded native logs, action samples, GPU CSVs and reports are retained with
hashes and durable GCS locations. All 14 retained evidence files matched their
recorded sizes and SHA-256 hashes locally and in GCS. Both owned clusters were
absent after cleanup; each Xbox job directory stayed below 1 MB with no weight
files. This is synthetic-input execution evidence;
`task_success` and `success_rate` remain null. Run's package requires the prepared
Linux native runtime recorded in its runtime lock; it is not a standalone robot
controller or a claim of task-quality acceptance.

## Preparation visibility

A subsequent control-plane change reports dependency setup, native configuration
and monotonic five-percent compilation updates. Build percentage is explicitly
separate from overall job progress. Replaying the actual successful evaluation
setup log produced the expected 0–100% build events without exposing raw setup
text. This change does not alter the worker or inference code used above.

## Software checks

Upstream `018ade0` was integrated after the live checks above, including immutable
local dataset training copies, ACT inference export and the Teaching view. The
merge preserves saved-job landing pages, separate creation forms, training drafts,
GPU-budget labels and the `/firebird` mount. No additional GPU execution is
claimed for this integration.

Post-merge checks:

- Core: 1,122 passed, two optional/platform skips. The skipped checks require the
  optional GCS file-reader package or a Windows host.
- Application browsers: 275 passed, including desktop/mobile jobs-first flows,
  local snapshot admission, ACT export controls, Teaching session conditions and
  preparation-progress checks.
- Dedicated diagnostics: 18 passed; prefixed export smoke: one passed.
- Ruff, formatting, TypeScript, generated API contracts and production builds passed.

Previously completed supporting checks, recorded before this upstream merge:

- Native quantization/evaluation worker: 253 passed, four platform/vendor skips.

Actual execution coverage remains SmolVLA for cloud Evaluate/Run. Other model
training adapters, simulation setups and physical robot integration have their
own validation scope; this record does not expand it.
