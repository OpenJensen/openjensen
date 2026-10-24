# Install the features you use

OpenJensen separates its web application, local USB companion, dataset readers,
training workers and simulator. Each profile has its own compatible environment.
The existing `uv sync --frozen` Core setup still works.

From an OpenJensen source checkout:

```sh
uv run --frozen jensen install list
uv run --frozen jensen install robot --dry-run
uv run --frozen jensen install robot
uv run --frozen jensen install robot --check
```

An already installed `jensen` command runs `install list`, `--dry-run` and `--check`
without installing or downloading anything. The `uv run --frozen` examples can
first synchronize a missing Core environment and download its dependencies.

`--dry-run` prints commands. `--check`
reads package metadata and checks external tool availability; it does not import
Torch, start a robot, launch a container or contact a cloud provider. An exit code
of 1 means the selected profile still needs setup. Neither command proves physical
robot communication, GPU training or simulator execution.

| Profile | What it installs | Runtime and computer |
| --- | --- | --- |
| `core` | API, CLI and web export | Python 3.14.7; Node 24.21+ below 25 and pnpm 12.6.0 for the web build; application host |
| `robot` | Feetech serial SDK, OpenCV, NumPy, Pillow and PyArrow | Python 3.14.7, isolated `workers/robot_lab/.venv`; computer holding USB devices |
| `datasets` | Parquet, HDF5 and image readers | Python 3.14.7, isolated `workers/_cpu_readers/.venv`; API host; FFmpeg/ffprobe separately |
| `training` | LeRobot 0.4.4/SmolVLA training dependencies | Python 3.11, isolated `workers/smolvla_qlora/.venv`; supported Linux CUDA host |
| `simulation` | Setup plan for external Isaac Sim 6.1 | Supported Linux NVIDIA host; Isaac's bundled Python and separate container |

The Robot Lab companion records LeRobot v3 datasets without installing Torch or
the training environment on the USB computer. With the arm on a Mac and the app
on a GPU server, install `robot` on the Mac and keep Core, dataset tools and GPU
workers on the server. The private connection forwards the app's Robot Lab requests to the Mac;
recordings stay on the Mac until you explicitly import them into a project.

Use the managed [Mac companion setup](../deploy/macos/README.md) for its service,
camera permission and private pairing. Installing dependencies does not assign
motor IDs, calibrate the arm or enable torque. Those are explicit Robot Lab steps.

The installer retains existing optional-profile environments. If a selected
environment is incomplete or has different pins, inspect it before choosing a
new location with `--environment /absolute/new/environment`. Reuse an existing
compatible interpreter for checks:

```sh
uv run --frozen jensen install datasets --check --environment /existing/reader
uv run --frozen jensen install robot --check --environment "$HOME/.local/share/jensen/robot-lab/runtime"
```

Set `JENSEN_CPU_READER_PYTHON` to the reader interpreter when it lives outside the
default worker directory. Profiles do not change saved application data,
credentials, models or dataset paths. Installation requires 10 GiB of free disk
headroom plus an estimated allowance for the selected dependencies; training has
a larger allowance. No cleanup runs automatically.

Simulation remains a separate [Isaac setup](../workers/isaac_sim/README.md).
`jensen install simulation --dry-run` explains the required external runtime;
`jensen install simulation` does not pull images or claim to install Isaac with
pip. Other native workers use their own documented Python 3.12 environments.
Installing a worker does not launch it or download policy weights. Submit
training and scheduled compute through the configured SkyPilot runtime.

## Workspace modules

The sidebar always shows the dataset, training and testing pages. Robot Lab,
Teaching, Augmentation, Decision lab and Cloud runs appear once this host has
their configuration; **Settings → Modules** lists what each one
needs. Set `JENSEN_MODULES` to a comma-separated list such as
`robot-lab,teaching` to choose the modules yourself, or to `all`.
