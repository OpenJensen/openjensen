# OPEN JENSEN

**Joint Embodied Neural Simulation & Execution Network**

**Inspect robot data, train policies, and prepare models for native inference.**

OPEN JENSEN brings dataset exploration, training, quantization and simulation into one self-hosted workspace. The web app, CLI and optional terminal client share the same projects, jobs and artifacts. Training and simulation run in separate worker environments, locally or on configured cloud compute.

[Get started](#run-locally) · [Workspace guide](docs/workspace-guide.md) · [Documentation](#documentation) · [Development](#development)

![OPEN JENSEN dataset explorer showing recorded camera frames and episode data](docs/images/dataset-explorer.png)

*Browse recorded episodes and inspect their camera views, state and action samples.*

## What you can do

| Sector | In the workspace |
| --- | --- |
| Dataset | Inspect public LeRobot v2/v3 datasets, play camera recordings and review sampled state/action rows. Validate local v3 datasets into immutable training copies. |
| Fine-tune | Choose a policy and compatible compute, follow loss and progress, resume supported runs, and download checkpoints with their recipes and provenance. |
| Distill | Train an ACT256 student from an ACT (Action Chunking with Transformers) teacher using selected observations, actions and episode splits. |
| Quantize | Convert SmolVLA checkpoints to GGUF with Q8 or Q4 language-weight quantization. Export completed ACT checkpoints as FP32 inference packages, then create packed INT8/INT4 ACT candidates with fresh CPU reload checks. |
| Evaluate / Run | Measure native inference, inspect full packed ACT predictions on recorded observations, run configured LIBERO evaluations, or launch an ACT/SmolVLA Isaac simulation. |
| Teaching / Augmentation | Record demonstrations through the Teaching interface, connect optional voice services, or review appearance-augmentation candidates before reusing them. |
| Decision lab | Compare text criteria with the configured local Muose scorer. |
| Cloud runs / Settings | Review saved cloud activity, configure connections and compute, and inspect project diagnostics. |

Each operation has its own model, dataset and runtime requirements. Available workflows depend on the configured workers and compatible model adapters.

[OPEN JENSEN Quant](workers/firebird_quant/README.md) is a 4/8-bit quantization library and CLI for dense PyTorch models. Use its native ACT packing path from the workspace or its standalone library and command-line client.

## Run locally

Use the pinned toolchain: **Python 3.14.7**, **uv 0.12.19**, **Node 24.21.0** and **pnpm 12.6.0**. Node is needed to build or develop the frontend. The running application is served by Python.

The command-line client is `firebird`. Application configuration uses `FIREBIRD_*` settings, and workspace data defaults to `.firebird/`.

From the repository root:

```sh
uv sync --frozen
uv venv --python 3.14.7 workers/_cpu_readers/.venv
uv pip install --python workers/_cpu_readers/.venv/bin/python --no-deps pyarrow==25.0.1
pnpm install --frozen-lockfile
pnpm build:web
uv run --frozen firebird serve
```

Open [localhost:8000](http://127.0.0.1:8000). On Windows, use `workers/_cpu_readers/.venv/Scripts/python.exe` in the reader installation command.

The core application and metadata intake need no GPU or provider key. Public Hub inspection needs network access. Training, native inference, simulation, voice and augmentation need their own configured workers or services. The server listens on loopback at the address above.

For tool-version fallbacks, development mode, local dataset configuration and CLI examples, see the [getting started guide](docs/getting-started.md).

For native ACT distillation and packed observation replay, use the separate
[local CPU worker setup](workers/local_cpu/README.md) to install dependencies
and configure the runtimes. For native ACT quantization, configure the
[registered packing worker](workers/firebird_quant/NATIVE_ACT.md).

### Your first project

1. Create a project and open **Dataset → Sources**.
2. Choose the pinned **SO-101 pickup** or **SO-100 pick & place** starter and select **Inspect dataset**.
3. Open **Load visual preview** to browse episodes, camera recordings and sampled actions.
4. In **Settings & diagnostics → Compute**, connect Google Cloud or use **Check this machine → Add worker** for an [installed local SmolVLA worker](docs/compute-settings.md#local-worker-setup). For local compute, enable local runs and save. Choose compatible compute and a recipe in **Fine-tune**. Cloud jobs use your configured account.
5. Open the saved run to review progress, checkpoints and provenance. Use the supported export or quantization path for that model.

## Documentation

Start with the [workspace guide](docs/workspace-guide.md) for a short explanation of every sector. The app also serves it at [`/guide/`](http://127.0.0.1:8000/guide/) through the **Guide** link on desktop and mobile.

| Topic | Guides |
| --- | --- |
| Workspace | [Sector guide](docs/workspace-guide.md), [frontend development](docs/workbench-design.md) |
| Setup and compute | [Getting started](docs/getting-started.md), [compute settings](docs/compute-settings.md), [cloud training](docs/skypilot-training.md) |
| Datasets | [Local training and Teaching](docs/local-training.md), [appearance augmentation](docs/augmentation.md), [source provenance](apps/web/public/datasets/README.md) |
| Policies and artifacts | [Training adapters](docs/native-training.md), [native policy workflow](docs/policy-workflow.md), [ACT export](docs/cloud-act-export.md), [ACT distillation](workers/policy_distillation/README.md), [cloud artifact storage](docs/cloud-artifact-storage.md) |
| Local optimization | [CPU worker setup](workers/local_cpu/README.md), [native ACT packing](workers/firebird_quant/NATIVE_ACT.md), [recorded-observation replay](workers/isaac_sim/NATIVE_REPLAY.md) |
| Evaluation and simulation | [Native Isaac](docs/native-simulation.md), [LIBERO Spatial](docs/spatial-workflow.md), [cloud inference](docs/cloud-inference.md), [cloud run monitoring](docs/cloud-runs.md) |
| Other clients | [Terminal workbench](docs/terminal.md), [desktop connector](apps/desktop/README.md), [Unity viewer](apps/unity/com.firebird.teaching-viewer/README.md) |
| Optional AI services | [Voice connections](docs/teaching-connections.md), [Muose decision worker](workers/decision/README.md), [Jev and Mk1.5 proposals](workers/teaching/PROVIDERS.md) |
| Development | [CI and local checks](docs/ci.md) |

The separate [`/docs/`](http://127.0.0.1:8000/docs/) route serves the [API reference](http://127.0.0.1:8000/docs/) and [OpenAPI schema](http://127.0.0.1:8000/openapi.json).

## Architecture

The Python application owns persistent projects, job state and artifact records. Isolated workers own model-specific dependencies and execution. OPEN JENSEN adopts LeRobot, SkyPilot and native runtimes instead of maintaining a second implementation of each trainer or simulator.

```text
apps/web/          Next.js and React interface, exported as static files
packages/core/     Python API, CLI, optional TUI, SQLite and job orchestration
workers/           Isolated training, inference, simulation and data workers
apps/desktop/      Tauri connector to a separately running local application
apps/unity/        Receive-only simulation viewer
tests/             Core, worker-contract and browser integration checks
docs/              Setup guides and workflow documentation
```

Project data lives in `.firebird/` by default. Set `FIREBIRD_DATA_DIR` to change it. One application process owns each workspace. Cloud training stores checkpoints in private GCS; explicit downloads and export operations can materialize them locally.

## Development

After installing the pinned dependencies:

```sh
uv run --frozen pytest -q
uv run --frozen ruff check packages/core scripts tests
uv run --frozen ruff format --check packages/core scripts tests
pnpm check:web
pnpm build:web
pnpm exec playwright install chromium
pnpm test:web
```

Workers keep separate environments and checks. Do not hand-edit generated API types. Follow the [CI guide](docs/ci.md) for optional terminal checks, schema generation, platform requirements and the full verification sequence.
