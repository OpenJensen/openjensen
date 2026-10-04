# Local and hosted verification

Main pushes run the CI workflow (application and native-worker jobs) on GCP Linux runners. Simulation is manual-only. Fork PRs and optional Windows checks retain GitHub-hosted runners. An unstarted job is not a pass. Local macOS and isolated Linux results are recorded with each delivery; Windows is deliberately excluded from default hackathon runs; its optional manual execution, hosted cache behavior and cold/warm timing remain separate acceptance items.

## Affected checks

CI jobs inspect the local Git diff with `.github/scripts/ci_scope.py`. Main pushes always select every check. PRs use their base-to-head merge-base diff; merge groups use their explicit before/after commits. Missing history, invalid event data, unknown paths and manual dispatch select all checks. Rename detection is disabled so both the removed and added paths are considered. Scope-job failure or missing/invalid outputs also selects all downstream checks. Stable `application-verification` and `native-worker-verification` aggregate jobs reject failed, cancelled or unexpectedly skipped suites; only an explicit successful scope decision permits a skip. These are available for future branch rules, without changing account or branch settings.

For PRs and merge groups:

- Application/core/web/test changes run application checks. Core lifecycle/worker contracts also run all native boundaries.
- Worker changes run their matching isolated suite and application adapters. Native training changes additionally check ACT export and teaching data consumers.
- New/unknown workers, shared CI changes and selector tests run every scope.
- Documentation-only changes skip those application/native jobs, except `docs/workspace-guide.md`: it ships in the web app and runs application checks. Simulation runs only through manual dispatch.
- Linux runs core, optional terminal tests, generated API checks, type checking, production build, all browser tests and diagnostics. Push, PR and merge-group application matrices contain Linux only. A manual application run defaults to Linux too; explicitly enabling **windows_browser** adds Windows core/terminal and complete web/build/browser checks. Native desktop packaging remains a separately recorded platform check.

The teaching job checks the real CPU LeRobot recorder/readback and provider proposal contracts in one environment, and voice SDK contracts in another. A separate small decision-worker job runs contracts without installing Torch or downloading Muose; its opt-in real-model test is explicitly skipped. Actual model-scoring evidence remains a separate local receipt. It does not access voice providers, cloud credentials or GPUs. Dependencies are installed on every run; caches contain dependency downloads, not application workspaces, model weights, credentials or virtual environments.

Superseded revisions of the same PR/workflow are cancelled. Each main/manual run keeps a unique concurrency group. `merge_group` support allows these checks to work if a GitHub merge queue is configured later; this does not enable a merge queue or bypass branch rules.

## Run locally

Use the versions in `.node-version`, `package.json` and the workflows. From the repository root:

```sh
uv sync --frozen --all-packages --extra tui
uv run --no-sync pytest -q
uv run --no-sync ruff check packages/core scripts tests .github/scripts
uv run --no-sync ruff format --check packages/core scripts tests .github/scripts
pnpm install --frozen-lockfile
uv run --no-sync python scripts/export_openapi.py
pnpm generate:client
git diff --exit-code -- packages/core/openapi.json apps/web/src/lib/api.generated.ts
pnpm check:web
pnpm build:web
pnpm exec playwright install --with-deps --only-shell chromium
pnpm test:web --retries=0
pnpm test:diagnostics --retries=0
```

The prefix smoke check also verifies the in-app guide and API reference under `/firebird`. Its separate export leaves the normal preview available:

```sh
NEXT_PUBLIC_BASE_PATH=/firebird FIREBIRD_PREFIX_SMOKE_EXPORT=1 pnpm build:web
NEXT_PUBLIC_BASE_PATH=/firebird FIREBIRD_TEST_WEB_DIR="$PWD/apps/web/out/prefix-smoke" pnpm exec playwright test --config tests/web/base-path.config.ts
pnpm build:web
```

The production-browser fixture owns a disposable local server and refuses an occupied port. Keep the developer's application running separately; do not kill a process without identifying it. Some core snapshot/video tests additionally require the isolated CPU reader and FFmpeg installed exactly as the application workflow describes. Native workers keep separate environments and commands in `.github/workflows/application.yml`; the core environment must not absorb their Torch/CUDA dependencies.

The selector itself is exercised with real Git histories, including deletes/renames and filenames containing newlines:

```sh
uv run --no-sync pytest -q tests/test_ci_scope.py
```

If change selection is suspect, manually dispatch CI to request every scope (and explicitly enable **windows_browser** only when Windows verification is wanted). Main pushes already request every scope; do not relax tests, application admission or branch protection. Measure successful hosted runs before reporting a runtime or free-minute saving.
