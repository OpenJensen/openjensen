# Getting started with OPEN JENSEN

In OPEN JENSEN, JENSEN stands for Joint Embodied Neural Simulation & Execution Network. The command-line client is `firebird`. Application configuration uses `FIREBIRD_*` settings, and workspace data defaults to `.firebird/`.

The [README](../README.md#run-locally) has the standard installation commands. This guide covers tool versions, the first dataset inspection, local data, development mode and the shared CLI.

## Toolchain

The application uses Python 3.14.7, uv 0.12.19, Node 24.21.0 and pnpm 12.6.0. Worker environments have their own Python and native dependencies. Keep them separate from the application environment.

If your installed uv cannot locate Python 3.14.7, use the pinned release:

```sh
uvx --from uv==0.12.19 uv sync --frozen
```

If your shell resolves a different pnpm release, use:

```sh
npx --yes pnpm@12.6.0 install --frozen-lockfile
npx --yes pnpm@12.6.0 build:web
```

The first dependency installation requires network access. The frontend is built once and then served by Python; Node is not a runtime dependency of the built web app. A Python-only install does not include the frontend build.

Parquet previews use a separate CPU reader. Follow the [reader setup guide](../workers/_cpu_readers/README.md) if the UI reports that it is missing. On Windows, its interpreter is `workers/_cpu_readers/.venv/Scripts/python.exe` rather than the POSIX `bin/python` path.

## Inspect a public dataset

Start the application from the repository root:

```sh
uv run --frozen firebird serve
```

Open [localhost:8000](http://127.0.0.1:8000), create a project and select a starter in **Dataset → Sources**. The SO-101 pickup starter uses `codywang/so101_pickup_test` at revision `ecef85bc07005f771ad86deeff1427f9d72953ed`.

Inspection resolves an immutable Hub revision and reads bounded metadata. It does not download model weights or camera videos. Counts and schemas are source-declared; missing action units, calibration and controller semantics remain visible warnings.

Select **Load visual preview** to fetch the episode index and a small set of recorded rows. The episode controls seek camera views together and stop at the selected episode boundary. Camera recordings stream from the pinned Hub revision. Codec errors appear in the player with a retry control.

The preview is not a dataset-wide quality audit. It returns at most five sampled rows and eight camera references per episode. Local sources can expose bounded Parquet samples, but camera playback currently requires a supported public Hub layout.

## Find your next workflow

Open **Guide** in the app for the short [workspace guide](workspace-guide.md), served at `/guide/` on desktop and mobile. Explanations live there; working pages keep their title, controls and results. The separate `/docs/` route remains the API reference.

The sidebar keeps the main capabilities directly accessible:

- **Data:** Dataset, Augmentation and Teaching. Inspect recordings, create visual
  variations or record new demonstrations in the configured teaching simulator.
- **Train:** Fine-tune, Distill and Quantize. Choose the model and preparation
  workflow that matches your policy.
- **Test:** Evaluate, Run and Decision lab. Check policies, try supported execution
  workflows or compare candidate actions with a configured decision model.
- **Workspace:** Cloud runs and Settings & diagnostics. Review cloud activity and
  connect the required compute and services.

To change a recording's appearance, open **Augmentation** directly or select
**Augment this dataset** from its inspection. Dataset, camera and appearance
choices are visible cards. Review generated clips before reusing their action
labels; augmentation does not automatically create a trainable dataset. See the
[augmentation guide](augmentation.md) for setup and limits.

Under **Run**, choose a card by what you want to inspect:

| Choice | What it does | Evidence limit |
| --- | --- | --- |
| **3D simulation** | Runs a compatible ACT or SmolVLA policy in the configured Isaac cup scene and records it. | Experimental execution; pickup success is not scored and calibration remains unverified. |
| **Replay observations** | Compares an ACT policy's actions on recorded observations. | Offline replay does not establish closed-loop task success. |
| **Check inference** | Checks a supported GGUF policy's loading and finite actions in the inference engine. | Execution checks do not establish robot task success. |

Selecting a card opens its workflow; starting a job remains a separate action.
Required workers, model compatibility and any paid-run consent still apply.
Saved results remain available when a worker is unavailable. Use **Evaluate**
for supported scored benchmarks, with an appropriately configured and validated
benchmark runtime.

## Use local data

Local intake is disabled until the operator sets `FIREBIRD_LOCAL_DATA_ROOT` before starting the server. Paths submitted by the UI or CLI must resolve inside that directory.

For example, with a dataset at `/path/to/robotics-data/my-dataset/meta/info.json`:

```sh
export FIREBIRD_LOCAL_DATA_ROOT=/path/to/robotics-data
uv run --frozen firebird serve
```

In PowerShell, use `$env:FIREBIRD_LOCAL_DATA_ROOT = 'C:\robotics-data'` before starting the server.

Choose **Prepare immutable training copy** when you need to train a supported native LeRobot policy from a finalized local v3 dataset. This performs full bounded validation and creates a content-identified snapshot. Metadata inspection alone does not make a dataset eligible for training.

Snapshot preparation requires the isolated CPU reader and FFmpeg/ffprobe. It currently supports Linux and macOS. The dedicated SmolVLA and Psi-Zero workers still require pinned Hub datasets. See [local training and Teaching](local-training.md) for limits, lineage grouping and worker setup.

## Use the CLI

Run the following in a second terminal while the application is running:

```sh
uv run --frozen firebird projects create "My robotics project"
uv run --frozen firebird projects list
uv run --frozen firebird capabilities
```

Use the returned project ID in place of `PROJECT_ID`:

```sh
uv run --frozen firebird inspect PROJECT_ID \
  --repo-id codywang/so101_pickup_test \
  --revision ecef85bc07005f771ad86deeff1427f9d72953ed
uv run --frozen firebird jobs list PROJECT_ID
```

For a permitted local dataset, the corresponding requests are:

```sh
uv run --frozen firebird inspect PROJECT_ID --path my-dataset
uv run --frozen firebird inspect PROJECT_ID --path my-dataset --snapshot-for-training
```

Inspect or cancel a specific job using its returned ID:

```sh
uv run --frozen firebird jobs show JOB_ID
uv run --frozen firebird jobs events JOB_ID
uv run --frozen firebird jobs cancel JOB_ID
```

Commands return JSON and call the same API as the web app. They do not create a second scheduler or write directly to SQLite. The optional [terminal workbench](terminal.md) provides an interactive view over those same records.

## Develop the frontend

Keep the Python API running on port 8000. In another terminal:

```sh
pnpm dev:web
```

Open [localhost:3000](http://127.0.0.1:3000). Development requests use the API on port 8000. Production exports use the same origin as the Python server. `NEXT_PUBLIC_API_URL` can override the API base at frontend build time.

## Application configuration

| Setting | Purpose |
| --- | --- |
| `FIREBIRD_DATA_DIR` | Persistent project/job workspace. Defaults to `.firebird/`. |
| `FIREBIRD_LOCAL_DATA_ROOT` | Explicitly permitted root for local dataset intake. Disabled when unset. |
| `FIREBIRD_WEB_DIR` | Static frontend directory. Defaults to `apps/web/out`. |
| `FIREBIRD_API_URL` | Server URL used by CLI and terminal clients. Defaults to `http://127.0.0.1:8000`. |
| `FIREBIRD_RUNTIME_CONFIG` | Operator-owned configuration for installed native workers. |
| `FIREBIRD_SIMULATION_CONFIG` | Operator-owned profiles for the experimental Native Isaac runner. |

One process owns each workspace through an OS file lock. Do not run multiple server workers against the same data directory. Startup marks unfinished jobs interrupted; recovery requires an explicit supported action. `Ctrl+C` stops the server and reconciles active jobs.

The server binds to loopback and does not provide hosted authentication. Configure cloud compute through [compute settings](compute-settings.md), native workers through the [policy workflow](policy-workflow.md), and simulation through the [Native Isaac guide](native-simulation.md).

The running application exposes its [API reference](http://127.0.0.1:8000/docs/) and [schema](http://127.0.0.1:8000/openapi.json). The reference needs a frontend build; the JSON schema is available without one.
