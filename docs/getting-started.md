# Getting started with OpenJensen

Start with dataset inspection: it needs the application and CPU reader, with no GPU or provider account. Run the complete [source installation commands](../README.md#quick-start) from the repository root, then configure optional workers when you need them.

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

Install the [CPU reader](../workers/_cpu_readers/README.md) using its complete requirements file. It includes PyArrow, NumPy, h5py and Pillow; PyArrow alone does not enable every import format. On Windows, select `workers/_cpu_readers/.venv/Scripts/python.exe` as its interpreter. Install FFmpeg and ffprobe on the API host for local video conversion and immutable training copies. See the [support matrix](support.md) for platform limits.

## Inspect a public dataset

Start the server:

```sh
uv run --frozen jensen serve
```

1. Open [localhost:8000](http://127.0.0.1:8000) and create a project.
2. Open **Datasets**, select **Hugging Face** under **Import a dataset**, and enter `codywang/so101_pickup_test` in **Dataset repository**.
3. Expand **Revision (optional)** and enter `ecef85bc07005f771ad86deeff1427f9d72953ed` to reproduce this example. Leaving it blank uses the latest `main` revision.
4. Select **Inspect dataset**. After inspection succeeds, the episode list and first episode's camera previews load automatically.
5. Select an episode from **Episodes**, then use **Play**, **Pause** and the playback slider to browse its camera recordings. Open **Frame samples** and choose **Joint / state** or **Action** to inspect sampled values.

Metadata and recordings need network access. If a request fails, use **Retry episode index** or **Retry episode preview**; for a camera playback error, use **Retry video**. The project remains available after restarting with the same workspace directory.

## Find your next workflow

Open **Guide** or the [workspace guide](workspace-guide.md), then follow the steps for the page you want to use.

For browser uploads, open **Datasets**, select **Local files**, choose your folder, ZIP or HDF5 file, and review the detected format, frame rate and mapping. Install the CPU reader and FFmpeg before converting local recordings. See [supported input layouts](dataset-import.md).

## Use local data

Put a finalized dataset under an allowed source directory and set its root before starting the server:

```sh
export JENSEN_LOCAL_DATA_ROOT=/path/to/robotics-data
uv run --frozen jensen serve
```

For PowerShell, set `$env:JENSEN_LOCAL_DATA_ROOT = 'C:\robotics-data'`.

For training, install the CPU reader and FFmpeg/ffprobe, then select **Prepare immutable training copy** when inspecting a LeRobot v3 dataset. Follow [local training setup](local-training.md).

## Use the CLI

In a second terminal, create a project and read its returned ID:

```sh
uv run --frozen jensen projects create "My robotics project"
uv run --frozen jensen projects list
uv run --frozen jensen capabilities
```

Replace `PROJECT_ID` with that ID and inspect a public dataset:

```sh
uv run --frozen jensen inspect PROJECT_ID \
  --repo-id codywang/so101_pickup_test \
  --revision ecef85bc07005f771ad86deeff1427f9d72953ed
uv run --frozen jensen jobs list PROJECT_ID
```

For a dataset inside the allowed local root:

```sh
uv run --frozen jensen inspect PROJECT_ID --path my-dataset
uv run --frozen jensen inspect PROJECT_ID --path my-dataset --snapshot-for-training
```

Replace `JOB_ID` with a returned job ID to follow or cancel it:

```sh
uv run --frozen jensen jobs show JOB_ID
uv run --frozen jensen jobs events JOB_ID
uv run --frozen jensen jobs cancel JOB_ID
```

## Troubleshooting

| What you see | What to check |
| --- | --- |
| The browser cannot reach port 8000 | Keep the server terminal open and read its startup error. For an occupied port, use `jensen serve --port 8001` and open the printed address. |
| The API works but the workspace is missing | Build with `pnpm build:web`. Launch from the repository root, or set `JENSEN_WEB_DIR` to the absolute `apps/web/out` directory. |
| Preview/import reports a missing dependency | Install the complete CPU reader requirements. Set `JENSEN_CPU_READER_PYTHON` for an interpreter outside the default checkout path. |
| Local video conversion cannot start | Confirm `ffmpeg -version` and `ffprobe -version` work on the API host. |
| Hub inspection or playback fails | Inspect the saved intake error; check network access, repository/revision and gated access. Use the player's retry control for playback errors. |
| A model or operation is unavailable | Check Compute for its installed worker/assets, and match the model's GPU memory, dataset format and camera requirements. |
| A submission response is lost | Inspect project job history and reconcile the saved attempt before starting another job. See [submission recovery](job-submissions.md). |
| Projects are missing after launching elsewhere | Check the selected workspace directory. Use an absolute `JENSEN_DATA_DIR` for a persistent installation. |

For a reproducible problem, open a [GitHub issue](https://github.com/OpenJensen/openjensen/issues) with the source revision, OS, relevant tool versions, steps and redacted error output. Keep credentials, private datasets and raw workspace files out of public reports.

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
| `JENSEN_DATA_DIR` | Workspace directory; default `.jensen/`. |
| `JENSEN_LOCAL_DATA_ROOT` | Allowed directory for local dataset intake. |
| `JENSEN_WEB_DIR` | Static frontend directory; default `apps/web/out`. |
| `JENSEN_API_URL` | CLI/TUI API address; default `http://127.0.0.1:8000`. |
| `JENSEN_RUNTIME_CONFIG` | Installed worker configuration file. |
| `JENSEN_SIMULATION_CONFIG` | Isaac runner profile file. |

Start one server process for each workspace directory. Stop it with `Ctrl+C` after active jobs have finished or been cancelled. Use [backup and upgrade](upgrading.md) before changing an existing installation; its compatibility section explains alpha command/settings/workspace aliases. Read the [API reference](http://127.0.0.1:8000/docs/) or [OpenAPI schema](http://127.0.0.1:8000/openapi.json) for direct requests.
