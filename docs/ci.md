# Development checks

GitHub Actions runs the Linux application, browser and worker checks on every push and pull request, and selects them from the files a change touches. Windows application and browser checks and the simulation workflow run on demand. Check your pull request's jobs; the pull request itself describes the change, not local results.

Browser tests cover desktop and mobile layouts. GPU training, Isaac simulation and desktop installation need their own environments and are not covered by these checks.

## Run the checks locally

Run the checks for the code you change, using the versions in `.node-version`, `package.json` and the workflows. From the repository root:

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
pnpm format:check
pnpm lint:web
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

- Browser checks need ports 8765, 8766 and 8767 free.
- Dataset and video checks need the [isolated reader](../workers/_cpu_readers/README.md) and FFmpeg.
- Worker checks run in each worker's own environment.

To test which checks a change selects:

```sh
uv run --no-sync pytest -q tests/test_ci_scope.py
```
