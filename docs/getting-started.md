# Getting started

This guide takes you from a fresh checkout to browsing a robot dataset, then shows how to bring your own data and use the CLI. None of it needs a GPU or a cloud account.

## 1. Install

Run the [Quick start commands](../README.md#quick-start) from the repository root. They install the app, the CPU reader that opens datasets, and the web interface.

You need Python 3.14.7, uv 0.12.19, Node 24.21.0 and pnpm 12.6.0. If your installed uv or pnpm is a different version, run the pinned releases instead:

```sh
uvx --from uv==0.12.19 uv sync --frozen
npx --yes pnpm@12.6.0 install --frozen-lockfile
npx --yes pnpm@12.6.0 build:web
```

- **CPU reader:** install it from its full requirements file (PyArrow, NumPy, h5py and Pillow); PyArrow alone does not open every format. On Windows its interpreter is `workers/_cpu_readers/.venv/Scripts/python.exe`. See the [CPU reader setup](../workers/_cpu_readers/README.md).
- **FFmpeg:** install `ffmpeg` and `ffprobe` on the machine running the app to convert local videos and prepare training copies.

Training, simulation and robot workers each get their own environment later; see [what runs where](support.md).

## 2. Open a public dataset

Start the server:

```sh
uv run --frozen jensen serve
```

1. Open [localhost:8000](http://127.0.0.1:8000) and create a project.
2. Open **Datasets**, select **Hugging Face** under **Import a dataset**, and enter `codywang/so101_pickup_test` in **Dataset repository**.
3. Optional: expand **Revision (optional)** and enter `ecef85bc07005f771ad86deeff1427f9d72953ed` to get exactly the version used here. Left blank, it uses the latest `main`.
4. Select **Inspect dataset**. The episode list and camera previews load when inspection finishes.
5. Pick an episode and press **Play**. Open **Frame samples** and choose **Joint / state** or **Action** to read the values.

If a request fails, use the retry control next to it (**Retry episode index**, **Retry episode preview** or **Retry video**). Projects stay in the workspace when you restart the server.

Next, open **Guide** in the app or the [workspace guide](workspace-guide.md) for every other page.

## 3. Use your own data

**Upload from the browser.** Open **Datasets**, select **Local files**, choose a folder, ZIP or HDF5 file, and review the detected format, frame rate and field mapping. [Supported layouts](dataset-import.md).

**Read from a folder on the server.** Put a finalized dataset under one directory and start the server with it as the allowed root:

```sh
export JENSEN_LOCAL_DATA_ROOT=/path/to/robotics-data
uv run --frozen jensen serve
```

In PowerShell: `$env:JENSEN_LOCAL_DATA_ROOT = 'C:\robotics-data'`.

**Prepare it for training.** When you inspect a local LeRobot v3 dataset, select **Prepare immutable training copy**. This needs the CPU reader and FFmpeg, on Linux or macOS. Then follow [local training](local-training.md).

## 4. Use the CLI

With the server running, open a second terminal:

```sh
uv run --frozen jensen projects create "My robotics project"
uv run --frozen jensen projects list
uv run --frozen jensen capabilities
```

Use the project ID from `projects list` in place of `PROJECT_ID`:

```sh
uv run --frozen jensen inspect PROJECT_ID \
  --repo-id codywang/so101_pickup_test \
  --revision ecef85bc07005f771ad86deeff1427f9d72953ed
uv run --frozen jensen jobs list PROJECT_ID
```

For a dataset under the allowed local root:

```sh
uv run --frozen jensen inspect PROJECT_ID --path my-dataset
uv run --frozen jensen inspect PROJECT_ID --path my-dataset --snapshot-for-training
```

Follow or cancel a job with its `JOB_ID`:

```sh
uv run --frozen jensen jobs show JOB_ID
uv run --frozen jensen jobs events JOB_ID
uv run --frozen jensen jobs cancel JOB_ID
```

The [terminal guide](terminal.md) covers the interactive terminal UI and recipe commands.

## Configuration

Set these before starting the server:

| Variable | Purpose |
| --- | --- |
| `JENSEN_DATA_DIR` | Workspace directory. Default `.jensen/`; use an absolute path for a lasting install. |
| `JENSEN_LOCAL_DATA_ROOT` | The only directory local dataset intake may read. |
| `JENSEN_WEB_DIR` | Built web interface. Default `apps/web/out`. |
| `JENSEN_API_URL` | Server address for the CLI and terminal UI. Default `http://127.0.0.1:8000`. |
| `JENSEN_RUNTIME_CONFIG` | Installed worker configuration file. |
| `JENSEN_SIMULATION_CONFIG` | Isaac runner profile file. |

Run one server per workspace directory. Stop it with `Ctrl+C` once active jobs have finished or been cancelled. Before changing an existing install, read [backup and upgrade](upgrading.md). The running server documents its API at [/docs/](http://127.0.0.1:8000/docs/) and [/openapi.json](http://127.0.0.1:8000/openapi.json).

## Troubleshooting

| What you see | What to check |
| --- | --- |
| The browser cannot reach port 8000 | Keep the server terminal open and read its startup error. If the port is taken, run `jensen serve --port 8001` and open the printed address. |
| The API works but the workspace page is missing | Run `pnpm build:web`, and start from the repository root or set `JENSEN_WEB_DIR` to the absolute `apps/web/out` path. |
| A preview or import reports a missing dependency | Install the full CPU reader requirements. Set `JENSEN_CPU_READER_PYTHON` if its interpreter is outside the checkout. |
| Local video conversion does not start | Check that `ffmpeg -version` and `ffprobe -version` work on the server. |
| Hub inspection or playback fails | Read the saved error; check network access, the repository and revision, and gated access. Use the player's retry control for playback errors. |
| A model or operation is unavailable | Check **Settings → Compute** for its worker and assets, and its GPU memory, dataset format and camera requirements. |
| A submission response was lost | Check the project's job history and reconcile the saved attempt before starting another job. See [submission recovery](job-submissions.md). |
| Projects are missing after starting elsewhere | Check the workspace directory, and set an absolute `JENSEN_DATA_DIR`. |

To report a problem, open a [GitHub issue](https://github.com/OpenJensen/openjensen/issues) with the source revision, OS, tool versions, steps and the error output. Remove credentials and private data first.

## Work on the web interface

Keep the server on port 8000 and start the development frontend in another terminal:

```sh
pnpm dev:web
```

Open [localhost:3000](http://127.0.0.1:3000). Set `NEXT_PUBLIC_API_URL` before building if the frontend should use a different API address.
