# RTX 3070 follow-up branch validation

This is the historical prep import validation. The codebase transfer preserves
the executable source and tests; see `RTX3070_TRANSFER_ORIGIN.json` for source
hashes. Its PR targets codebase `main` and depends on the quantization module.

Date: September 26, 2026. Base: `feat/quantization_module` at
`edd9fb00d234bd350b566ad66d1a92551bf67a35`. Follow-up branch:
`feat/quantization-rtx3070`. This branch imports the standalone experiment into
`workers/vla_cpp`; it does not import the separate CPU benchmark branch.

## Checks of this imported source

Run from `workers/vla_cpp` on macOS with CPython 3.11.14:

```bash
uv sync --locked --extra quantize --extra test
uv run --no-sync python -m pytest -q
bash -n scripts/build_cuda.sh scripts/run_cuda_rollouts.sh
uv run --no-sync python -m compileall -q policykit scripts
```

Result: **30 passed**, no skips. This includes the existing 21 worker tests and
nine CUDA parser checks. The installed-wheel test also invokes `--help` on the
three experimental modules from outside the source tree with no optional GPU,
PyTorch or simulator packages installed. Shell/Python syntax checks passed.

A replay check imported the new `cuda_bench.samples_from_log` and
`cuda_common.parse_actions` helpers, then read the preserved `smolvla-cuda-v1`
logs. All 15 timing blocks matched their recorded 20 samples exactly, and all
five 350-value action vectors matched the recorded JSON arrays. The stricter
declared action-length check rejects malformed logs without changing these valid
recorded results. No new timing or task-success measurement is inferred from replay.

In the prep repository, checks `python3 scripts/check_docs.py` and
`python3 scripts/task_records.py` validate documentation links/fences and the
existing task index/dependency rules. This PR does not change task-card status.

## Historical hardware evidence

The [RTX report](quantization-rtx3070.md) and
[manifest](quantization-rtx3070-evidence.json) describe the precursor experiment:
300 timed CUDA calls, five paired LIBERO development episodes, and small
AWQ/SmoothQuant calibration runs. Its source snapshot, binaries, image IDs and
artifact hashes are preserved. Its 63-passed/two-skipped Python count belongs to
that broader experiment snapshot, not this branch's smaller suite.

The imported branch has not been rerun through native CUDA inference. Import
changes are documented in `GPU_ORIGIN.json`: worker-local paths, reuse of the
existing SHA-256 helper, dependency-free CUDA helpers, an isolated rollout entry
point and input/setup documentation. Model weights and native quantization math
are unchanged, but branch tests and historical GPU evidence remain distinct.

Packed ModelOpt export/reload, broader paired quality, untouched final evaluation
and deployment-package verification remain open. The worker continues to report
`conversion_only` and `deployment_verified: false`.
