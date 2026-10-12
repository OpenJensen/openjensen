# Local and CI verification

GitHub Actions runs the selected Linux application, browser and worker checks for pushes and pull requests. Windows application/browser lanes are a manual opt-in; simulation has a separate manual workflow. Inspect the checks selected for the actual revision and retain local results when a required environment or hosted execution is unavailable.

Browser desktop/mobile lanes exercise web layouts. Worker contract tests and generated CPU fixtures do not establish live GPU training, Isaac execution or installed desktop acceptance. Keep those receipts separate and bound to their exact source/runtime/inputs.

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

Run the prefixed-export check, then rebuild the normal export:

```sh
NEXT_PUBLIC_BASE_PATH=/jensen JENSEN_PREFIX_SMOKE_EXPORT=1 pnpm build:web
NEXT_PUBLIC_BASE_PATH=/jensen JENSEN_TEST_WEB_DIR="$PWD/apps/web/out/prefix-smoke" pnpm exec playwright test --config tests/web/base-path.config.ts
pnpm build:web
```

Run browser checks with ports 8765, 8766 and 8767 available. Install the [isolated reader](../workers/_cpu_readers/README.md) and FFmpeg for dataset/video checks. Run worker checks in each worker's own environment.

Check the change selector:

```sh
uv run --no-sync pytest -q tests/test_ci_scope.py
```
