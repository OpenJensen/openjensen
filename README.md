# Firebird application

The first application slice of a local VLA lifecycle platform: **persistent projects, real robotics metadata intake, and visual dataset exploration**, through one Python application and shared web/CLI operations. Dataset → Fine-tune → Distill → Quantize → Evaluate → Run remains the product scope. Training, compression, simulator evaluation and policy execution are **planned**, not working features in this scaffold.

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

Metadata counts and schemas are **source-declared**. Intake preserves their provenance, hashes the metadata and warns that action units, calibration, controller semantics, media integrity and simulator compatibility have not been verified. It does not infer task success or a training recipe from a dataset name. On-demand episode previews can now show camera videos and a small set of recorded action/state rows. Dataset-wide validation remains pending.

Local metadata intake is disabled by default. Set `FIREBIRD_LOCAL_DATA_ROOT` to an explicitly permitted dataset directory before starting the server; paths must resolve within it. For example, if it contains `my-dataset/meta/info.json`, inspect with `--path my-dataset`. On PowerShell use `$env:FIREBIRD_LOCAL_DATA_ROOT = 'C:\robotics-data'`; on Linux/macOS use `export FIREBIRD_LOCAL_DATA_ROOT=/path/to/robotics-data`.

## Explore a dataset visually

The **Sources** view includes two real, revision-pinned starters: SO-101 pickup and SO-100 pick-and-place. Create a project, select a starter or enter a public LeRobot repository, and click **Inspect dataset**. The **Inspection** view preserves metadata counts, warnings, and source provenance.

Click **Load visual preview** to explicitly fetch the episode index and a small sample of recorded data. Select an episode, play or seek its camera views with the shared episode controls, and switch between joint/state and action sample tables. Camera playback uses the source's episode offsets and stops on the selected episode's last frame. Switching views pauses playback.

This operation is separate from metadata-only inspection. Public Hugging Face LeRobot v2/v3 datasets with supported file layouts are supported; local datasets and embedded image columns still expose metadata only. Video URLs point at the inspected commit and stream directly from Hugging Face to the browser. A browser unable to decode the source codec shows a video error and retry control.

The API exposes `GET /api/v1/jobs/{job_id}/episodes?offset=0&limit=6` and `GET /api/v1/jobs/{job_id}/episodes/{episode_index}` after a successful inspection. Preview downloads and Parquet parsing are bounded; large or unsupported sources return a readable limitation. At most five actual sample rows and eight camera references are returned per episode. Numeric values are source samples, not validated controller semantics or model-performance evidence.

## Workspace and code boundaries

- `apps/web`: static Next/React client; generated API types under `src/lib`.
- `packages/core`: GPU-independent Python modular monolith, API, CLI, async SQLite records and versioned migrations.
- `workers`: native environment boundaries and contracts. The lightweight metadata worker runs as a supervised subprocess; GPU environments are not installed yet.
- `apps/desktop`: reserved later Tauri shell.
- `tests`: persistence, subprocess intake, bounded reads, cancellation, failure and access-boundary checks.

The default workspace is `.firebird/`, ignored by Git. Set `FIREBIRD_DATA_DIR` to choose a different directory. One application process owns it via an OS file lock. SQLite stores projects/job records; each job has its request/result files. Startup marks unfinished jobs interrupted and requires an explicit retry. Do not run multiple server workers on the same workspace. The API reference is at `/docs/` and the schema at `/openapi.json`. The reference shares the web workspace shell and theme, and loads its endpoint/model content from the live schema. Build the web client to serve the reference; the JSON schema remains available without a web build.

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
pnpm exec playwright install chromium
pnpm test:web
```

Do not hand-edit generated API types. Claim a task card before parallel work, coordinate shared schema/manifest/migration changes, and provide evidence plus independent review. The Linux/Windows CI workflow is scaffolded; actual remote CI and Windows GPU evidence are separate gates. Mac development checks do not prove those gates passed.
