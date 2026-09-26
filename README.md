# Firebird application

A local VLA lifecycle application with persistent projects, dataset intake, selectable LoRA/QLoRA fine-tuning, native quantization, evaluation and reload-verified packages through shared web/CLI jobs. Native operations require configured worker environments. Distillation remains planned.

Start with the [policy workflow and setup guide](docs/policy-workflow.md). Recommended quantization defaults depend on the execution target; detailed metrics and advanced controls live under **Settings & diagnostics**. See [validation evidence](docs/workflow-validation.md) for the scope of actual hardware checks.

Application Git root: this directory. The nested `firebird-hackathon-prep/` directory remains a separate, ignored Git repository. Its [accepted plan](firebird-hackathon-prep/docs/idea/18_stack-and-phased-build-plan.md) and [task register](firebird-hackathon-prep/docs/tasks/README.md) hold planning and coordination records; they are not included in an application-only clone.

## Run locally

Use Node **24.21.0**, pnpm **12.6.0**, Python **3.14.7** and uv **0.12.19**. Worker Python/GPU dependencies are separate and are not needed for intake. First installation requires network access; no remote application service or API key is required.

```sh
uv sync --frozen
pnpm install --frozen-lockfile
pnpm build:web
uv run --frozen firebird serve
```

Open **http://127.0.0.1:8000**. Python serves the static frontend and API; Node is only needed to build/develop it. The app binds to loopback. Hosted authentication and desktop installers are later tasks. `Ctrl+C` stops the server and reconciles active jobs.

If an older installed uv cannot locate Python 3.14.7, use `uvx --from uv==0.12.19 uv sync --frozen`. If your shell resolves an older pnpm, run `npx --yes pnpm@12.6.0 install --frozen-lockfile` and the same prefix for build/check scripts. These do not require changing your global tool versions.

For frontend development, keep the API running and use `pnpm dev:web` in another terminal; open http://127.0.0.1:3000. The development UI calls port 8000. Exported builds use the same origin. `NEXT_PUBLIC_API_URL` can override the API base at build time.

## Real intake

Create a project in the UI, then inspect `codywang/so101_pickup_test` at revision `ecef85bc07005f771ad86deeff1427f9d72953ed`. The reader resolves an immutable Hub revision and reads at most 2 MiB per metadata response. It does not download videos or weights. Public LeRobot v2/v3 metadata is supported; other robotics formats are explicit future adapters.

The same operations are available in the terminal:

```sh
uv run --frozen firebird projects create "My robotics project"
uv run --frozen firebird projects list
uv run --frozen firebird inspect PROJECT_ID --repo-id codywang/so101_pickup_test --revision ecef85bc07005f771ad86deeff1427f9d72953ed
uv run --frozen firebird jobs list PROJECT_ID
uv run --frozen firebird jobs show JOB_ID
uv run --frozen firebird jobs cancel JOB_ID
uv run --frozen firebird capabilities
```

Replace IDs with the returned values. CLI commands output JSON and call the same API. They do not open a second scheduler or write the database.

Metadata counts and schemas are **source-declared**. Intake preserves their provenance, hashes the metadata and warns that action units, calibration, controller semantics, media integrity and simulator compatibility have not been verified. It does not infer task success or a training recipe from a dataset name. Parquet/video previews and dataset-wide validation remain pending.

Local metadata intake is disabled by default. Set `FIREBIRD_LOCAL_DATA_ROOT` to an explicitly permitted dataset directory before starting the server; paths must resolve within it. For example, if it contains `my-dataset/meta/info.json`, inspect with `--path my-dataset`. On PowerShell use `$env:FIREBIRD_LOCAL_DATA_ROOT = 'C:\robotics-data'`; on Linux/macOS use `export FIREBIRD_LOCAL_DATA_ROOT=/path/to/robotics-data`.

## Workspace and code boundaries

- `apps/web`: static Next/React client; generated API types under `src/lib`.
- `packages/core`: GPU-independent Python modular monolith, API, CLI, async SQLite records and versioned migrations.
- `workers`: native environment boundaries and contracts. Metadata and native policy workers run as supervised subprocesses; GPU environments are installed separately.
- `apps/desktop`: reserved later Tauri shell.
- `tests`: persistence, subprocess intake, bounded reads, cancellation, failure and access-boundary checks.

The default workspace is `.firebird/`, ignored by Git. Set `FIREBIRD_DATA_DIR` to choose a different directory. One application process owns it via an OS file lock. SQLite stores projects/job records; each job has its request/result files. Startup marks unfinished jobs interrupted and requires an explicit retry. Do not run multiple server workers on the same workspace. The API reference is at `/docs` and the schema at `/openapi.json`.

`FIREBIRD_WEB_DIR` selects the static build directory (default `apps/web/out`). `FIREBIRD_API_URL` changes the CLI's server URL. Do not expose this initial local server to untrusted networks; hosted access control is not implemented.

## Verify and contribute

```sh
uv run --frozen pytest -q
uv run --frozen ruff check packages/core scripts tests
uv run --frozen ruff format --check packages/core scripts tests
uv run --frozen python scripts/export_openapi.py
pnpm generate:client
pnpm check:web
pnpm build:web
```

Do not hand-edit generated API types. Claim a task card before parallel work, coordinate shared schema/manifest/migration changes, and provide evidence plus independent review. The Linux/Windows CI workflow is scaffolded; actual remote CI and Windows GPU evidence are separate gates. Mac development checks do not prove those gates passed.
