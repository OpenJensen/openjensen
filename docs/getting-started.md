# Getting started with OPEN JENSEN

Run the installation commands in the [README](../README.md#run-locally) from the repository root.

## Toolchain

Use Python 3.14.7, uv 0.12.19, Node 24.21.0 and pnpm 12.6.0. Install each worker in its own environment.

To use the pinned uv release:

```sh
uvx --from uv==0.12.19 uv sync --frozen
```

To use the pinned pnpm release:

```sh
npx --yes pnpm@12.6.0 install --frozen-lockfile
npx --yes pnpm@12.6.0 build:web
```

Install the [CPU reader](../workers/_cpu_readers/README.md) for Parquet previews. On Windows, select `workers/_cpu_readers/.venv/Scripts/python.exe` as its interpreter.

## Inspect a public dataset

Start the server:

```sh
uv run --frozen firebird serve
```

1. Open [localhost:8000](http://127.0.0.1:8000) and create a project.
2. Open **Dataset → Sources** and choose the **SO-101 pickup** starter.
3. Select **Inspect dataset**, then **Load visual preview**.
4. Choose an episode and camera; use the playback controls to seek through the recording.

The starter uses `codywang/so101_pickup_test` at revision `ecef85bc07005f771ad86deeff1427f9d72953ed`. If playback fails, use the player's retry control.

## Find your next workflow

Open **Guide** or the [workspace guide](workspace-guide.md), then follow the steps for the page you want to use.

## Use local data

Put a finalized dataset under an allowed source directory and set its root before starting the server:

```sh
export FIREBIRD_LOCAL_DATA_ROOT=/path/to/robotics-data
uv run --frozen firebird serve
```

For PowerShell, set `$env:FIREBIRD_LOCAL_DATA_ROOT = 'C:\robotics-data'`.

For training, install the CPU reader and FFmpeg/ffprobe, then select **Prepare immutable training copy** when inspecting a LeRobot v3 dataset. Follow [local training setup](local-training.md).

## Use the CLI

In a second terminal, create a project and read its returned ID:

```sh
uv run --frozen firebird projects create "My robotics project"
uv run --frozen firebird projects list
uv run --frozen firebird capabilities
```

Replace `PROJECT_ID` with that ID and inspect a public dataset:

```sh
uv run --frozen firebird inspect PROJECT_ID \
  --repo-id codywang/so101_pickup_test \
  --revision ecef85bc07005f771ad86deeff1427f9d72953ed
uv run --frozen firebird jobs list PROJECT_ID
```

For a dataset inside the allowed local root:

```sh
uv run --frozen firebird inspect PROJECT_ID --path my-dataset
uv run --frozen firebird inspect PROJECT_ID --path my-dataset --snapshot-for-training
```

Replace `JOB_ID` with a returned job ID to follow or cancel it:

```sh
uv run --frozen firebird jobs show JOB_ID
uv run --frozen firebird jobs events JOB_ID
uv run --frozen firebird jobs cancel JOB_ID
```

## Develop the frontend

Keep the API on port 8000 and start the frontend in another terminal:

```sh
pnpm dev:web
```

Open [localhost:3000](http://127.0.0.1:3000). Set `NEXT_PUBLIC_API_URL` before building if the frontend needs a different API base.

## Application configuration

Set variables for the configuration you use before starting the server:

| Variable | Value to supply |
| --- | --- |
| `FIREBIRD_DATA_DIR` | Workspace directory; default `.firebird/`. |
| `FIREBIRD_LOCAL_DATA_ROOT` | Allowed directory for local dataset intake. |
| `FIREBIRD_WEB_DIR` | Static frontend directory; default `apps/web/out`. |
| `FIREBIRD_API_URL` | CLI/TUI API address; default `http://127.0.0.1:8000`. |
| `FIREBIRD_RUNTIME_CONFIG` | Installed worker configuration file. |
| `FIREBIRD_SIMULATION_CONFIG` | Isaac runner profile file. |

Start one server process for each workspace directory. Stop it with `Ctrl+C`. Read its [API reference](http://127.0.0.1:8000/docs/) or [OpenAPI schema](http://127.0.0.1:8000/openapi.json) for direct requests.
