# Local and hosted verification

GitHub Actions minutes are currently exhausted. An unstarted hosted job is not a pass. Local macOS and isolated Linux results are recorded with each delivery; Windows execution, hosted cache behavior and cold/warm timing remain separate acceptance items.

## Affected checks

Application and native-worker workflows inspect the local Git diff with `.github/scripts/ci_scope.py`. PRs use their base-to-head merge-base diff; main pushes and merge groups use their explicit before/after commits. Missing history, invalid event data, unknown paths and manual dispatch select all checks. Rename detection is disabled so both the removed and added paths are considered. Scope-job failure or missing/invalid outputs also selects all downstream checks. Stable `application-verification` and `native-worker-verification` aggregate jobs reject failed, cancelled or unexpectedly skipped suites; only an explicit successful scope decision permits a skip. These are available for future branch rules, without changing account or branch settings.

- Application/core/web/test changes run application checks. Core lifecycle/worker contracts also run all native boundaries.
- Worker changes run their matching isolated suite and application adapters. Native training changes additionally check ACT export and teaching data consumers.
- New/unknown workers, shared CI changes and selector tests run every scope.
- Documentation-only changes skip those application/native jobs. The independent simulation workflow keeps its existing behavior.
- Linux runs core, optional terminal tests, generated API checks, type checking, production build, all browser tests and diagnostics. Windows retains core/terminal tests; a manual application run with **windows_browser** also runs the complete Windows web/build/browser checks. Native desktop packaging remains a separately recorded platform check.

The teaching job checks the real CPU LeRobot recorder/readback in one environment and voice SDK contracts in another. It does not access voice providers, cloud credentials or GPUs. Dependencies are installed on every run; caches contain dependency downloads, not application workspaces, model weights, credentials or virtual environments.

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

The production-browser fixture owns a disposable local server and refuses an occupied port. Keep the developer's application running separately; do not kill a process without identifying it. Some core snapshot/video tests additionally require the isolated CPU reader and FFmpeg installed exactly as the application workflow describes. Native workers keep separate environments and commands in `.github/workflows/native-workers.yml`; the core environment must not absorb their Torch/CUDA dependencies.

The selector itself is exercised with real Git histories, including deletes/renames and filenames containing newlines:

```sh
uv run --no-sync pytest -q tests/test_ci_scope.py
```

If change selection is suspect, manually dispatch both workflows to request every scope (and enable Windows browser checks when needed). Revert the specific CI change to restore the prior unconditional matrix; do not relax tests, application admission or branch protection. Measure successful hosted runs before reporting a runtime or free-minute saving.
