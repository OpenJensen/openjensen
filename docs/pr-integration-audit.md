# Integration audit — September 26, 2026

The six author-owned PRs (#43–#47 and #52) form one application integration.
This is a software integration check, **not a completed product acceptance**.
The locked product brief requires measured task quality, target constraints,
independent final evaluation and an exact tested export. Those hardware gates
remain incomplete, so #52 remains a draft and main requires explicit approval.

## Consolidation

- #45 (CPU), #46 (RTX experiments) and #47 (GPU comparison) merge into
  `feat/quantization_module` (#43), preserving their commit ancestry.
- #43 is integrated into `feat/workflow-integration` (#52).
- #44's complete QLoRA branch (`8015a6a`) was already merged into #52 by
  `5942f4f`; there is no additional training-branch diff to merge.
- The workflow uses the existing execution owner, project records and isolated
  Python 3.11 workers. The application remains GPU-independent on Python 3.14.
- No merge into main is part of this audit.

## Reproduced defects fixed

1. The quantization worker's documented `uv sync --locked --extra quantize
   --extra test` failed: the conversion extra had been added without updating
   `workers/vla_cpp/uv.lock`. Regenerated the lock and verified the locked install.
2. POSIX cancellation returned after the worker leader exited on SIGTERM even
   if a native descendant ignored SIGTERM. A real subprocess regression reproduced
   a surviving descendant. Cleanup now kills the remaining process group.
3. If every compressed candidate failed, the optimizer could mark the sole
   floating reference validated. A subprocess workflow regression reproduced
   this false comparison approval. Fewer than two runnable configurations now
   returns diagnostics only and retains failure evidence.
4. Existing application CI did not install or test any of these native worker
   projects. Added separate quantization, lightweight training and benchmark
   jobs, including locked installation and preserved-evidence checksum checks.
   These jobs do not certify CUDA, full model execution or simulator quality.

## Local verification

Pinned application tools: Python 3.14.7, uv 0.12.19, Node 24.21.0, pnpm 12.6.0.
Workers use Python 3.11.14 in separate environments.

| Check | Result |
|---|---|
| Application pytest | 115 passed |
| Quantization/CPU/RTX worker pytest | 99 passed; 4 prepared-native/Linux skips |
| Lightweight training pytest | 62 passed; 3 CUDA/dependency skips |
| GPU benchmark lightweight pytest | 4 passed; 2 native dependency skips |
| Application Ruff and formatting | Passed |
| Training and GPU benchmark Ruff | Passed |
| Generated OpenAPI and TypeScript client | Regenerated; no drift |
| TypeScript and production frontend build | Passed |
| GPU benchmark evidence SHA-256 inventory | All 37 files passed |
| Browser: actual pinned SO-101 intake | Succeeded; 30 episodes, 4,500 frames |
| Browser: fine-tuning and diagnostics | LoRA/QLoRA shown; no-runtime launch disabled; settings render separately |

The application tests exercise real subprocess protocol fixtures, cancellation,
restart, project boundaries, artifact integrity, constrained selection and final
acceptance. Fixture metrics are synthetic and are not model quality evidence.
The single application warning concerns upstream TestClient/httpx deprecation.

## Whole-project fit and remaining gates

| Product requirement | Current evidence / limitation |
|---|---|
| Dataset intake | Actual pinned metadata intake works through API, CLI and web |
| Fine-tuning | Application adapter and lightweight tests pass; prior full QLoRA two-step smoke and fresh reload exist; full LoRA and real resume remain unverified |
| Quantization | Worker tests and prior CUDA reload evidence exist; final integrated trained export → GGUF → packed reload has not run |
| Quality-constrained optimization | Contract tests enforce explicit limits, two runnable configurations and unused final states; actual new artifacts still need paired episodes and final validation |
| Export and Run | Integrity and protocol tests pass; actual final-package simulation acceptance is pending |
| Native backend breadth | SmolVLA paths exist; custom engine probes are limited evidence, not a second complete application optimization backend |
| Distillation | Explicitly planned |
| Windows | Application CI is distinct from Windows GPU acceptance, which remains pending |
| Cloud simulation | Teammate #49 is isolated and explicitly partial; no application capability is registered |

SSH to the documented RTX execution host timed out during this audit. No new GPU
measurement, full-model training, cloud provisioning or simulation run was
performed. Historical Q4 quality includes regressions; no universal winner or
deployment readiness is claimed. See [workflow validation](workflow-validation.md)
and [GPU evidence](../workers/benchmark_gpu/evidence/2026-09-26/REPORT.md).

At the reviewed refs, teammate #48/#50 conflicts with the workflow in shared
workspace/CSS files; #51 also conflicts in README/CSS/generated schemas. These
separate PRs must be reconciled and the combined build tested before claiming
all open team branches integrate together. Main itself is included through
`30b0cab`. The isolated simulation worker does not complete the application's
Evaluate/Run path.
