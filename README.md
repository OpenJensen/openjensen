# Firebird application

A self-hosted VLA workspace with persistent projects, visual dataset exploration, cloud training adapters for all 15 policy choices observed in KiteML, and checkpoint-to-GGUF quantization for SmolVLA. SmolVLA supports LoRA/QLoRA; the other policies use isolated native training environments. Configured native workers provide evaluation, Spatial workflows and reload-verified packages through shared web/CLI jobs. Distillation remains planned.

The application can run on Xbox while SkyPilot provisions GCP workers. Models and datasets download on the worker; checkpoints stay in private GCS. The UI shows live training metrics, labels checkpoints by model and step, and lets you quantize the latest or an earlier checkpoint. Quantization merges the trained weights, packs Q4/Q8 language tensors, and requires an actual native inference sanity check before reporting success. It does not establish robot task success.

[Firebird Quant](workers/firebird_quant/README.md) adds a standalone 4/8-bit quantization API and CLI for dense PyTorch models beyond SmolVLA, including convolutional, recurrent, transformer and custom architectures. It provides portable inference, explicit tensor coverage and verified save/reload; model quality and hardware speed require separate evaluation.

The optional [terminal workbench](docs/terminal.md) and [Tauri desktop shell](apps/desktop/README.md) use the same projects, intake and job records. The desktop shell connects to a separately running local backend; it is not yet a bundled Python installer. Optional [Muose decision scoring](workers/decision/README.md) and [Jev/Mk1.5 proposals](workers/teaching/PROVIDERS.md) have isolated worker interfaces and explicit integration/quality gates.

Start with [cloud training and quantization](docs/skypilot-training.md), [model adapters](docs/native-training.md), or the [Xbox deployment guide](deploy/xbox/README.md). The [live verification record](docs/cloud-training-verification.md) separates completed GPU runs from configuration/dependency checks and pending validation. Local evaluation and robot execution require separately configured native workers; the UI exposes only compatible targets. See the [native policy workflow](docs/policy-workflow.md), [Spatial workflow](docs/spatial-workflow.md) and [upstream validation evidence](docs/workflow-validation.md) for their setup and acceptance scope.

Fine-tune, Quantize, Evaluate and Run open saved jobs first, with separate creation forms. SmolVLA cloud Evaluate and Run execute real CUDA engine checks and package verification; see the [workflow and live inference verification](docs/jobs-first-verification.md) for measured results and limits.

**Run → Native Isaac** supports complete ACT and SmolVLA packages, including uploaded TAR archives, through the shared cup-scene launcher. It saves owned simulation jobs, verified video and downloadable trajectory records. This is experimental execution evidence; scored cup-task Evaluation remains gated on calibration and an agreed success criterion. See [native simulation](docs/native-simulation.md) for the distinction, setup and current boundaries.

**Dataset → Augmentation** supports Gemini Omni lighting, texture and custom appearance edits for selected camera clips, with before/after review and provenance exports. See the [augmentation setup and scope](docs/augmentation.md); this optional feature uses the saved Google Cloud login or a server-side Gemini API key, plus FFmpeg.

Application Git root: this directory. The nested `firebird-hackathon-prep/` directory remains a separate, ignored Git repository. Its [accepted plan](firebird-hackathon-prep/docs/idea/18_stack-and-phased-build-plan.md) and [task register](firebird-hackathon-prep/docs/tasks/README.md) hold planning and coordination records; they are not included in an application-only clone.

**Dataset → Teaching** adds session-bound simulation controls and an optional LiveKit/OpenRouter voice connection. Finalized demonstrations can enter native LeRobot training through verified local copies. See [local training and teaching](docs/local-training.md) for setup and current acceptance limits.

## Run locally

Use Node **24.21.0**, pnpm **12.6.0**, Python **3.14.7** and uv **0.12.19**. GPU dependencies are separate and are not needed for metadata intake. Parquet episode previews use an isolated CPU reader. First installation requires network access; no remote application service or API key is required.

```sh
uv sync --frozen
uv venv --python 3.14.7 workers/_cpu_readers/.venv
uv pip install --python workers/_cpu_readers/.venv/bin/python --no-deps pyarrow==25.0.1
pnpm install --frozen-lockfile
pnpm build:web
uv run --frozen firebird serve
```

On Windows, use `workers/_cpu_readers/.venv/Scripts/python.exe` for the reader installation command. The [CPU reader guide](workers/_cpu_readers/README.md) covers a custom interpreter location and supported preview boundaries. A missing reader produces an explicit preview error; the application never installs one automatically.

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

Metadata counts and schemas are **source-declared**. Intake preserves their provenance, hashes the metadata and warns that action units, calibration, controller semantics, media integrity and simulator compatibility have not been verified. It does not infer task success or a training recipe from a dataset name. On-demand episode previews can now show camera videos and a small set of recorded action/state rows. Full local LeRobot v3 validation is available through the explicit immutable training-copy workflow; ordinary inspection remains metadata-only.

Local metadata intake is disabled by default. Set `FIREBIRD_LOCAL_DATA_ROOT` to an explicitly permitted dataset directory before starting the server; paths must resolve within it. For example, if it contains `my-dataset/meta/info.json`, inspect with `--path my-dataset`. On PowerShell use `$env:FIREBIRD_LOCAL_DATA_ROOT = 'C:\robotics-data'`; on Linux/macOS use `export FIREBIRD_LOCAL_DATA_ROOT=/path/to/robotics-data`.

## Explore a dataset visually

The **Sources** view includes two revision-pinned starters: SO-101 pickup and SO-100 pick & place. Each card uses a real source-camera still; dataset counts and preview checks are recorded in [dataset provenance](apps/web/public/datasets/README.md). Create a project, select a starter or enter a public LeRobot repository, and click **Inspect dataset**. The **Inspection** view preserves metadata counts, warnings, and source provenance.

Click **Load visual preview** to explicitly fetch the episode index and a small sample of recorded data. Select an episode, play or seek its camera views with the shared episode controls, and switch between joint/state and action sample tables. Camera playback uses the source's episode offsets and stops on the selected episode's last frame. Switching views pauses playback.

This operation is separate from metadata-only inspection. Public Hugging Face LeRobot v2/v3 datasets with supported file layouts are supported; local datasets and embedded image columns still expose metadata only. Video URLs point at the inspected commit and stream directly from Hugging Face to the browser. A browser unable to decode the source codec shows a video error and retry control.

The API exposes `GET /api/v1/jobs/{job_id}/episodes?offset=0&limit=6` and `GET /api/v1/jobs/{job_id}/episodes/{episode_index}` after a successful inspection. Preview downloads and Parquet parsing are bounded; large or unsupported sources return a readable limitation. At most five actual sample rows and eight camera references are returned per episode. Numeric values are source samples, not validated controller semantics or model-performance evidence.

## Workspace and code boundaries

- `apps/web`: static Next/React client; generated API types under `src/lib`.
- `packages/core`: GPU-independent Python modular monolith, API, CLI, async SQLite records and versioned migrations.
- `workers`: native environment boundaries and contracts. Metadata and native policy workers run as supervised subprocesses; GPU environments are installed separately.
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

Do not hand-edit generated API types. Claim a task card before parallel work, coordinate shared schema/manifest/migration changes, and provide evidence plus independent review. CI selects affected suites conservatively; Linux is the hackathon default, with Windows checks available by manual dispatch. See [verification and local fallback](docs/ci.md). Full GPU workflows and Windows GPU acceptance remain separate validation gates.

Connect Google Cloud once in **Settings → Compute**, then choose L4, T4, or A100 and click **Start fine-tuning**. Provider preparation and SkyPilot dispatch happen automatically. See [compute settings](docs/compute-settings.md) and [SkyPilot job lifecycle](docs/skypilot-training.md).

The **Fine-tune** run monitor shows reported optimizer steps and percentage, training and validation loss, learning rate, duration, estimated remaining training time, checkpoints, activity and worker output. **Reproducibility** exposes pinned model/dataset inputs, the resolved recipe, seed, episode split and available runtime evidence as downloadable JSON. New runs record evidence from submission onward; older runs show only what their saved files establish. See [training comparison and monitoring](docs/training-comparison.md) for the Kite ML comparison, verification scope and remaining model support.
