# Local and hosted verification

Run the applicable checks locally before publishing changes and record the source revision and results. Hosted GitHub Actions is currently unavailable for this project, so hosted execution is not an acceptance gate. Keep the workflow definitions: they specify the intended checks and runner configuration, but a configured or unstarted job is not a passing result.

The application workflow uses self-hosted GCP Linux runners when the repository variable `CI_RUNNER` is `gcp`; otherwise it uses GitHub-hosted Ubuntu 24.04. Fork PRs always use GitHub-hosted runners. Windows checks use `windows-latest` and require the manual **windows_browser** option. Simulation has its own manual-only workflow and uses self-hosted runners unless `SIM_RUNNER` is `hosted`. Native desktop packaging requires separate verification.

## Affected checks

Python checks in the `application` matrix have a 15-minute job budget and default to two shards. The `web` matrix has a 20-minute budget per job: one core leg runs schema, type, build, pure/API/workflow browser checks and diagnostics; separate legs shard the selected desktop and mobile browser projects. Optional repository variables `CI_PY_SHARDS`, `CI_DESKTOP_SHARDS` and `CI_MOBILE_SHARDS` choose 1–12 shards, with defaults of 2, 4 and 6. Every web leg builds its own static export; diagnostics run once per selected OS on the core leg.

The selector `.github/scripts/ci_scope.py` reads a local Git diff. Main pushes use their before/after diff, enable application adapter checks for changed workers, and add mobile coverage whenever desktop coverage is selected. PRs use the base/head merge-base diff. Version-tag pushes and manual dispatch request every scope. Rename detection is disabled so both removed and added paths are considered.

Unknown paths select Python and desktop browser checks; main pushes also add mobile coverage. If PR change detection fails, that same Python/desktop fallback applies. Failed change detection for a push or an unknown event selects every scope. A failed scope job causes downstream jobs to attempt their checks, but the aggregate gates still reject that failed scope result. Stable `application-verification` and `native-worker-verification` gates reject failed, cancelled or unexpectedly skipped suites; only an explicit successful scope decision permits a skip.

For affected paths:

- Core, scripts, deployment and desktop changes select Python and desktop browser checks. Core lifecycle/worker contracts also select all registered native-worker suites; core dataset changes select training, ACT export and teaching consumers.
- Web application and browser-test changes select desktop and mobile browser checks. Other application tests select Python checks.
- Registered worker changes select their matching isolated suite. Main pushes also select application adapter checks; PRs do not add those adapter checks automatically. SmolVLA worker changes additionally select ACT export and teaching; VLA C++ changes also select the benchmark suite.
- Shared CI changes and selector tests select every scope. Unknown worker directories use the unknown-path fallback.
- The root README, `workers/README.md` and `docs/` pages skip application/native checks, except `docs/workspace-guide.md`, which is rendered into the web app and selects desktop browser checks. Documentation inside application or worker directories follows the enclosing directory's scope rules.

Linux is the default application matrix. A manual application run also defaults to Linux; enabling **windows_browser** adds Windows Python/terminal and full web/build/browser checks. Simulation runs only through its separate manual dispatch.

The teaching job checks the CPU LeRobot recorder/readback and provider proposal contracts in one environment, and voice SDK contracts in another. A separate decision-worker job runs contracts without installing Torch or downloading Muose; its real-model test is opt-in. These default worker checks do not access voice providers, cloud credentials or GPUs. Dependencies are installed on every run; caches contain dependency downloads, not application workspaces, model weights, credentials or virtual environments.

Superseded revisions of the same PR/workflow are cancelled. Each main/manual run keeps a unique concurrency group. The application workflow currently handles main-branch pushes, version-tag pushes, PRs and manual dispatch; no `merge_group` event is configured.

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

For local verification, run every affected suite when change selection is uncertain. If hosted execution becomes available, manual dispatch requests every scope; enable **windows_browser** only when Windows verification is wanted. Main pushes use the affected-path selection described above. Do not relax tests, application admission or branch protection, or infer runtime and cost savings from workflow configuration alone.
