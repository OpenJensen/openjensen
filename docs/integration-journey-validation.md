# Application journey validation — 2026-09-26

INT-003 verifies the installed application and browser/API/CLI integration using
local, explicitly synthetic worker fixtures. It does not establish model quality,
GPU performance, training convergence, simulation success or hardware acceptance.

## Executed checks

Application source: `74fae22903dc142f8ed9a179f04c7f63aff77f4d`, comprising
`c857f31` plus the `917ca0d` workflow defaults and `9b4acb9` test-reader fixes.
Only test infrastructure and this evidence were changed afterward.
Environment: macOS arm64; Python 3.14.7; uv 0.12.19; Node 24.21.0;
pnpm 12.6.0; Next.js 16.3.6; Playwright 1.63.0 with Chromium.

- Production frontend build and its TypeScript check: passed.
- Full browser suite: **50 passed**, four parallel workers, including four new
  actual API journeys. The final change to download the transformed artifact
  instead of the floating reference was then rechecked: **1 passed**.
- Ruff lint and format checks on the four new/changed Python harness files: passed.
- Clean wheel install, real CLI/API operations and API restart: passed. The
  [machine-readable receipt](evidence/2026-09-26-int003-clean-install.json)
  records the wheel digest, installed module location, synthetic jobs and archive
  digest. The disposable environment and workspace were removed after the run.

The new `workflow` browser project uses the real production export, actual
application routes and SQLite/filesystem persistence. It never intercepts routes:

1. Create a project, inspect the local LeRobot metadata fixture, read the same
   result through the real CLI, reload and retain the metadata-only warnings.
2. Submit an optimizer workflow to a registered CPU fixture subprocess, read
   identical jobs/artifacts through the CLI, reload the browser and download a
   transformed artifact. Check the downloaded bytes against the API response,
   registered manifest digest and every packaged file hash. Require
   `diagnostics_only` with no selected artifact; fixture metrics confer no quality
   approval.
3. Observe a real child-process stdout event, cancel the running job, then verify
   cancelled status and absence of results/downloads after reload and through CLI.
4. Surface an intentional child-process failure, preserve it across reload, and
   successfully start another run without changing the failed historical record.

Each test creates its own project; the server uses a disposable workspace, fixture
source and runtime configuration on port 8765. It refuses to reuse a live server.
Existing API reference/view-contract tests retain their limited route mocks.

## Failures found while integrating the tests

The first full run had **45 passed, 4 failed**. Two view-contract checks assumed
there would be no projects in the shared server; real journey tests invalidated
that assumption. They now explicitly supply an empty project list. Two Q4 request
checks tried to uncheck the initial default checkbox before the remounted Settings
panel restored its saved checked value. They now wait for that saved state before
unchecking, then assert it is unchecked. The expected Q4 request was not relaxed.
Both fixes passed in desktop and mobile projects. No product code was changed.

## Reproduce

After the normal locked dependency installation:

```sh
pnpm build:web
pnpm test:web
uv run --frozen ruff check scripts/verify_clean_install.py tests/web/serve.py tests/fixtures/browser_worker
uv run --frozen ruff format --check scripts/verify_clean_install.py tests/web/serve.py tests/fixtures/browser_worker
uv run --frozen python scripts/verify_clean_install.py --evidence-dir /tmp/firebird-clean-install-evidence
```

The clean-install script is opt-in and POSIX-only because it passes an owned bound
socket to Uvicorn. It builds a wheel, exports frozen non-development dependencies,
creates an empty Python environment and installs that wheel without editable
source paths. It runs outside the checkout and confirms both Torch and PyArrow
are absent and unimported. The installed CLI submits local metadata intake and a
synthetic worker job. After stopping and restarting its server, it verifies the
same job, artifacts, events and transformed archive bytes.

Dependency/build setup may need network access and used the local package cache
on this run. Application operations use local fixtures and loopback only; this
is not an air-gapped dependency bootstrap claim. The registered fixture worker
remains in the checkout by design. The clean-wheel check tests the headless core;
it does not claim that the Python wheel bundles the frontend. Browser checks
separately exercise the production frontend export.

No Hugging Face downloads, GPU runs, cloud provisioning or robot operations were
performed. Windows/Linux browser execution remains a CI check, not a result from
this macOS run. Actual policy training, closed-loop simulation, target-hardware
limits, deployment-package policy execution and UI model-quality acceptance
remain outside this fixture evidence.
