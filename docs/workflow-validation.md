# Workflow integration validation — 2026-09-26

This is a historical verification record. Interface names and availability below describe the recorded integration, not a new test of the current UI. Use the [workspace guide](workspace-guide.md) for current navigation and capabilities.

Status: **native GPU acceptance is deferred by user decision**. Further Xbox
testing is deferred; the software foundation is approved for merging into main.
This connects application jobs to workers; it does not complete TRAIN-001,
QUANT-001, EVAL-002 or Windows support.
The final integration includes main through `652fdb9` (strict capability evidence,
atomic restart reconciliation, isolated local preview, shared API reference and
sidebar layout). Hardware runs below preceded that merge and some later progress,
packaging and identity checks; they are not an exact-final-commit GPU certification.

## Local checks

### Diagnostics follow-up

The Diagnostics tab now contains the completed dedicated-L4 reference snapshot
from benchmark commit `e5866f0`, alongside the existing L4 and RTX views. These
recorded model measurements remain separate from application job results.

The production UI was exercised against two disposable real application servers:
one with no execution target and one with an explicit synthetic subprocess worker.
Fourteen browser checks passed across desktop and mobile: recorded numbers and
hardware switching, navigation during delayed project loading, Spatial protocol
selection and locking, missing runtime
guidance, policy submission through the real API, persisted results after reload,
missing-policy navigation, unsupported simulation,
active-worker cancellation, and target-discovery failure. Production build and
TypeScript checks passed. These tests verify UI/API/worker wiring; their synthetic
reports are not GPU performance or LIBERO quality evidence. Actual application
GPU acceptance gates below remain deferred.

```sh
uv sync --frozen
pnpm install --frozen-lockfile
pnpm build:web
pnpm test:diagnostics
```

The browser suite starts only local test servers and protocol fixtures. It does
not connect to a benchmark host, download model weights or rent hardware.

### TEST-001: bounded diagnostics worker waits

Diagnostics and workflow browser tests now share a 45-second job observer. Import,
evaluation and cancellation keep explicit expected terminal states; another terminal
state fails immediately with the job ID, stage and recorded error. A deadline keeps
the last observation, including when its HTTP request times out. Earlier transport
failures retain their original cause. Diagnostics has a 120-second enclosing test
budget for import plus evaluation; no application timeout or retry policy changed.
Its disposable servers use ports 8767 and 8766, separate from the existing browser
suite on 8765 and the user's application on 8000.

Final local verification on macOS arm64, Node **24.21.0**, pnpm **12.6.0** and
application Python **3.14.7**:

- TypeScript and production build passed.
- `pnpm test:web --retries=0`: **52 passed**. The existing real subprocess failure,
  wrong-terminal rejection and seven-second delayed successful retry now exercise
  the shared observer; the delay exceeds Playwright's former five-second default.
- `pnpm test:diagnostics --retries=0`: **18 passed**. New checks exhaust a short
  observation budget on a real slow job, prove that waiting did not cancel it,
  explicitly cancel the owned fixture, and reject its unexpected terminal state.
  A separate temporary HTTP fixture verifies deadline diagnostics both before any
  snapshot and after a running snapshot. It contains no model metrics.
- `git diff --check` passed. The first local command wrapper selected older tool
  versions; final checks were repeated using the exact versions above.

Independent review reproduced a raw transport-timeout message that omitted the last
job snapshot. The corrected observer preserves the original timeout as its cause
and consistently reports the bounded job deadline. All temporary HTTP connections
and deliberately slow jobs are explicitly cleaned up. These are software test
receipts only; Linux/native Windows CI for this repair and coordinator review remain
required before TEST-001 completion. No GPU, cloud or user workspace was used.

### WEB-004: project preference readiness

[Windows run 36263941876](https://github.com/sobhanb-eth/firebird-hackathon-codebase/actions/runs/36263941876)
exposed a separate product race during the TEST-001 checks. Both failed attempts
opened Settings before project loading finished. Spatial settings were written
under the empty project key, then a project-specific remount restored the default
Object/engine/500 recipe. A controlled delayed-project probe reproduced that exact
submitted request; the new disabled-control regression also failed on the old UI.

Workflow controls now require a project present in successfully loaded project
data and restoration of that project's preferences. Empty project keys are never
read or written. Settings, job controls and submission stay unavailable while
loading, after a load error, or if the selected project disappears. Navigation,
diagnostics guidance and recorded benchmark comparisons remain accessible.
Project switching and reload preserve each project's saved recipe. If browser
storage is unavailable, editing works for the current mounted view only; settings
do not persist across stage changes or reloads.

Final local checks include the merged offline-preflight change from main
`9c925e5db3c1f2137353fbf7ace30b82e8aaa5c3`, using Node **24.21.0**, pnpm **12.6.0**,
application Python **3.14.7** and the existing native Python **3.11.14** environment:

- Production build and TypeScript checks passed.
- Application: **391 passed, 1 skipped**, with the existing Starlette deprecation
  warning. Native CPU worker: **252 passed, 4 skipped** for conditional platform or
  vendor checks. Ruff checks and formatting passed.
- Browser suite: **64 passed**; diagnostics suite: **18 passed**, both with
  **zero retries**. The delayed-project test checks the actual Spatial request;
  additional checks cover saved A/B recipes and submitted requests, empty/error
  recovery, removed-project refetch, invalid JSON and unavailable browser storage.
- An independent reviewer passed all **14** focused desktop/mobile readiness
  checks. The refetch test advances Playwright's clock past the application's
  five-second cache freshness period, without a wall-clock sleep.

These are application and synthetic-worker checks, not GPU performance or
closed-loop policy acceptance. Exact-head Linux/native Windows CI and coordinator
review remain required before WEB-004 completion. The existing application on port
8000, cloud resources and real policy assets were not touched.

### Existing application checks

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
