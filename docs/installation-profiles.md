# Install only what you use

OpenJensen keeps the app, the robot companion, dataset readers, training workers and the simulator in separate environments, so each computer installs only what it runs. The Quick start `uv sync --frozen` setup is the `core` profile.

## From the command line

```sh
uv run --frozen jensen install list
uv run --frozen jensen install robot --dry-run
uv run --frozen jensen install robot
uv run --frozen jensen install robot --check
```

- `list` shows every profile and whether it is ready.
- `--dry-run` prints the commands without running them.
- `--check` reads package metadata and looks for external tools. It does not import Torch, start a robot, launch a container or contact a cloud provider. Exit code 1 means the profile still needs setup.

An installed `jensen` runs `list`, `--dry-run` and `--check` without downloading anything; `uv run --frozen` may first sync a missing core environment. A passing check does not prove that the robot, GPU training or simulator works.

| Profile | Installs | Runs on |
| --- | --- | --- |
| `core` | API, CLI and web interface | The server. Python 3.14.7; Node 24.21 (below 25) and pnpm 12.6.0 to build the web interface |
| `robot` | Feetech serial SDK, OpenCV, NumPy, Pillow and PyArrow in `workers/robot_lab/.venv` | The computer the arm is plugged into. Python 3.14.7 |
| `datasets` | Parquet, HDF5 and image readers in `workers/_cpu_readers/.venv` | The server. Python 3.14.7; FFmpeg and ffprobe installed separately |
| `training` | LeRobot 0.4.4 and SmolVLA training in `workers/smolvla_qlora/.venv` | A supported Linux CUDA machine. Python 3.11 |
| `decision` | CPU Torch, tokenizer and scorer; model files are configured separately | The server. Python 3.12.14 |
| `simulation` | A setup plan for Isaac Sim 6.1 | A supported Linux NVIDIA machine, with Isaac's own Python and container |

Installing needs 10 GiB of free disk plus room for the profile's dependencies (more for `training`). Installing never changes saved projects, credentials, models or dataset paths, and nothing is cleaned up automatically.

## Where each piece runs

With the arm on a Mac or Linux computer and the app on a server, install `robot` on the arm's computer and keep `core`, `datasets` and GPU workers on the server. The companion records LeRobot v3 datasets without Torch or a training stack, and recordings stay on that computer until you import them. Set up its service, camera permission and pairing with the [macOS](../deploy/macos/README.md) or [Linux](../deploy/linux/README.md) companion installer; see [SO-101 Robot Lab](so101-robot-lab.md).

Installing dependencies never assigns motor IDs, calibrates or enables torque; those are explicit steps in Robot Lab.

## Reuse existing environments

Existing optional environments are kept. If one is incomplete or pinned differently, inspect it before choosing a new location with `--environment /absolute/new/environment`. To check an existing interpreter:

```sh
uv run --frozen jensen install datasets --check --environment /existing/reader
uv run --frozen jensen install robot --check --environment "$HOME/.local/share/jensen/robot-lab/runtime"
```

Set `JENSEN_CPU_READER_PYTHON` when the dataset reader lives outside its default directory.

## Simulation and other workers

`jensen install simulation --dry-run` explains the Isaac runtime it needs; `jensen install simulation` does not pull images or pip-install Isaac. Follow the [Isaac setup](../workers/isaac_sim/README.md). Other native workers use their own documented Python 3.12 environments. Installing a worker never starts it or downloads policy weights. Cloud training runs through SkyPilot; see [cloud training](skypilot-training.md).

## Install from the web

Open **Settings → Modules**.

- **Add module** shows a page in this workspace's sidebar; **Hide** removes it without deleting its dependencies or data. An added module may still need a companion, provider or model configured.
- **Runtime packages** names the server and installs Dataset tools, Robot Lab and the CPU Decision scorer, with progress, Cancel and Retry. Dataset readers become available once their dependencies verify. Training uses the Linux NVIDIA worker setup in **Settings → Compute**.

Packages install on the **server**, even when you use the browser on another device. Installing Robot Lab packages on the server does not set up the companion on the arm's computer. Simulation and Teaching need the external Isaac runtime and scene, Augmentation needs provider credentials, and Decision Lab needs model files and license acceptance; the page links to each setup. Nothing here downloads policy weights, starts paid resources or runs jobs.

- Installs continue if you leave the page or close the browser. Stopping the service cancels an install and keeps partial downloads for the retry. After an abnormal shutdown, an installer that is still running blocks a new one.
- Only bundled, allowlisted dependency plans run; the browser cannot supply commands, package indexes or paths. Compatible existing environments are reused; new ones live in the workspace's `module-environments` directory.
- Module management is private and blocked on the public gateway. The server's `JENSEN_MODULES` setting takes precedence and locks these controls.

## Workspace modules

Dashboard, Datasets, the Train and Test pages, Models and Settings always appear. Robot Lab, Teaching, Augmentation, Decision Lab and Cloud runs appear once the server has their configuration or you add them in **Settings → Modules**, which lists what each needs. To choose yourself, set `JENSEN_MODULES` to a comma-separated list such as `robot-lab,teaching`, or to `all`.
