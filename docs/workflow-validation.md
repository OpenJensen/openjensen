# Workflow integration validation — 2026-09-26

Status: **native GPU acceptance is deferred by user decision**. Further Xbox
testing is deferred; the software foundation is approved for merging into main.
This connects application jobs to workers; it does not complete TRAIN-001,
QUANT-001, EVAL-002 or Windows support.
The final integration includes main through `4388c53` (strict capability evidence,
atomic restart reconciliation, isolated local preview and shared API reference).
Hardware runs below
preceded that merge and some later progress, packaging and identity checks; they
are not an exact-final-commit GPU certification.

## Local checks

- Application: **113 passed**, including the new native subprocess workflow,
  cross-project boundaries, cancellation, concurrent artifact downloads, resume
  containment and preservation of completed stages across restart. Main's process
  SIGKILL/recovery tests also pass. Native protocol fixtures are explicitly synthetic;
  their metrics and quality outcomes are not model evidence.
- Quantization/benchmark worker: **99 passed, 4 skipped** (native-environment gates).
- Lightweight training worker: **62 passed, 3 skipped** (CUDA tests on macOS).
- Ruff checks/formatting, generated OpenAPI/client, TypeScript check and static
  frontend build pass. Frontend checks used Node 24.21.0 and pnpm 12.6.0; core uses
  Python 3.14.7. Worker environments use Python 3.11.
- Browser inspection: Dataset remains the initial page, Fine-tune exposes LoRA and
  QLoRA, and Settings & diagnostics holds compression/evaluation preferences and
  recorded measurements. Distill remains visibly planned.

```sh
uv run --frozen pytest -q
uv run --frozen ruff check packages/core scripts tests
uv run --frozen ruff format --check packages/core scripts tests
uv run --frozen python scripts/export_openapi.py
pnpm generate:client
pnpm check:web
pnpm build:web
# Run each worker suite from its own isolated environment; see its README.
```

## Actual RTX 3070 execution

Host: `aayrapetyan@xbox-360`, Ubuntu 22.04 / WSL2, RTX 3070 8 GiB. Existing desktop
and other GPU users remained running. New workspace:
`/home/aayrapetyan/firebird-workflow-integration-20260926`; app data:
`validation-engine`. Original experiment directories and previous PR evidence were
preserved. Full job receipts are `validation-<job-id>.json`; stage requests, logs,
results, bundles and events are below `validation-engine/jobs/<job-id>/`.

1. Job `fb7ceed5-0406-492f-b1e2-e7a94bb56780` imported the prepared floating policy,
   then correctly failed native launch because `LD_LIBRARY_PATH` omitted the native
   build's `bin` directory. Its completed import remained available.
2. Job `72ba7f83-254d-49f7-bcb9-e061967354a9` reused that import and **succeeded**:
   floating reference measurement, actual Q4 and Q8 conversion, fresh CUDA reload,
   1,600 finite action values per probe and four timed calls per policy. It returned
   **diagnostics_only**, without automatic quality promotion. Tiny timing samples
   are not a sustainable throughput claim.
3. Actual CUDA unit checks: **3 passed** — existing NF4 gradient/freeze/save test
   plus LoRA and QLoRA floating-export tests with nonzero adapter deltas, saved
   projections, strict reload and finite/parity checks. These use a small policy.
4. Intake `a8396835-e53a-4ee5-8d8c-ac61e12af3f7` resolved the pinned SO-101 dataset.
   Full-model QLoRA job `08058d50-90c6-41f4-a4c6-133b6d9b9a0f` **succeeded** after
   two optimizer steps, held-out loss evaluation, checkpoint save and a fresh-process
   reload. The probe action's maximum absolute reload difference was **0.0**.
   Observed training losses were 0.1601239294 and 0.0405492298; held-out losses were
   0.0477913953 and 0.0325235799. These are smoke-run values, not convergence evidence.
5. Native floating-export job `1165a7cc-ca8a-475a-8517-37d5093906a5` **succeeded**
   from that registered QLoRA checkpoint. It materializes the actual NF4 base plus
   trained adapters and saved projections.

The full-model smoke exposed and fixed two loading incompatibilities: LeRobot's
serialized `type` must be dispatched through `PreTrainedConfig`, and mixed F32/BF16
checkpoint dtypes must be preserved while still strictly checking keys/shapes.
Earlier failed jobs remain in the workspace (`d39da93c-...`, `68eba962-...`).

Pinned training inputs: `lerobot/smolvla_base` revision
`d9f33c94a60fb382c90dea2164c96845bd955e28`; backbone
`HuggingFaceTB/SmolVLM2-500M-Video-Instruct` revision
`7b375e1b73b11138ff12fe22c8f2822d8fe03467`; dataset
`codywang/so101_pickup_test` revision `ecef85bc07005f771ad86deeff1427f9d72953ed`.
The training recipe used two steps, no warmup, accumulation 1, evaluation/save every
step and one evaluation batch. No policy rollout or robot-quality claim follows.

The prepared LIBERO floating GGUF SHA-256 is
`ed2c28e16ca6bb0d312483b6a57d306f6bb09360fc9a99e37a3a30f1beaa90c3`.
The newly produced LM Q4 GGUF SHA-256 is
`68c6c658b59ca28651bd45c8396f89e3b374951f97d4ed27d4c2228d475c760a`.
Its bytes differ from the older GGUF-version pilot; the older pilot's task success
must not be attributed to this artifact without running its own episodes.

## Remaining hardware gates

These GPU workflow checks are explicitly deferred by the user. They do not
prevent landing the tested software foundation and are not claimed as passed.

SSH to the RTX host stopped connecting after training/export completed. The
following checks were prepared but could not run in that outage:

- Updated application adapter → actual LIBERO episodes → package rerun, using the
  newly generated Q4 artifact, then paired baseline/candidates and unused final states.
- Trained export → GGUF conversion → quantization → native reload/evaluation. Install
  the `convert` extra in the conversion environment before this stage; a quantization-only
  environment lacks the converter's Torch/safetensors dependencies.
- Full-model LoRA smoke and actual cancellation/resume of a full-model training run.
- Exact final source snapshot rerun and independent review; Windows GPU validation.

These are explicit pending gates. The task cards/issues remain open. No deployment
package, end-to-end trained native policy, closed-loop quality approval, or universally
best quantization recipe is claimed by the checks above.
