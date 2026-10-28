# OpenJensen

**An open-source workspace for robot learning.** Record demonstrations on a real arm, inspect datasets, fine-tune and compress robot policies, and test them in simulation, all from one self-hosted web app.

![Robot Lab setting up an SO-101 arm: the Connect, Motor IDs, Calibrate, Teleoperate and Record steps, a 3D view of the arm's numbered motors, and the camera preview](docs/images/robot-lab.png)

[Run it](#run) · [Workspace guide](docs/workspace-guide.md) · [Getting started](docs/getting-started.md) · [SO-101 Robot Lab](docs/so101-robot-lab.md) · [Contributing](CONTRIBUTING.md)

## What you can do

| Page | What it does |
| --- | --- |
| **Robot Lab** | Set up an SO-101 arm from the computer it is plugged into: assign motor IDs, calibrate, teleoperate with on-screen controls or a leader arm, and record LeRobot v3 episodes into a project. |
| **Teaching** | Record demonstrations in Isaac simulation, with joint corrections, failure markers and optional voice. |
| **Datasets** | Inspect public LeRobot v2/v3 datasets from Hugging Face, or import local folders, ZIP and HDF5 files. Play all cameras in sync and read state and action samples. |
| **Augmentation** | Generate appearance variations of recorded clips and review them before reuse. |
| **Fine-tune** | Train SmolVLA, ACT, π₀, π₀.₅, GR00T N1.7, Diffusion Policy and other LeRobot policies on a local NVIDIA GPU or cloud GPUs. Follow loss live, resume runs and download checkpoints. |
| **Distill** | Train a small ACT student from an ACT teacher. |
| **Quantize** | Convert SmolVLA to GGUF (Q8 or Q4) and pack ACT to INT8 or INT4. |
| **Evaluate** | Check a policy's loading, action validity and speed, or measure task success on LIBERO Spatial. |
| **Run** | Run a policy in Isaac simulation, or replay it on recorded observations. |
| **Decision Lab** | Score text criteria with a small local model (experimental). |
| **Models** | Browse every checkpoint and derived model with its history, and start the next step from it. |
| **Cloud runs** | Follow jobs running on cloud GPUs. |

The `jensen` command line and an optional terminal UI work with the same projects and jobs through the same API.

![Datasets page playing a recorded SO-101 episode with its camera view and episode list](docs/images/dataset-explorer.png)

*Dataset: [codywang/so101_pickup_test](https://huggingface.co/datasets/codywang/so101_pickup_test) (Apache-2.0).*

## Run

You need Python 3.14.7, [uv](https://docs.astral.sh/uv/) 0.12.19, Node 24.21.0 and pnpm 12.6.0. No GPU or cloud account is needed to start.

```sh
git clone https://github.com/OpenJensen/openjensen.git
cd openjensen
uv sync --frozen
uv venv --python 3.14.7 workers/_cpu_readers/.venv
uv pip install --python workers/_cpu_readers/.venv/bin/python --no-deps -r workers/_cpu_readers/requirements.txt
pnpm install --frozen-lockfile
pnpm build:web
uv run --frozen jensen serve
```

Open [localhost:8000](http://127.0.0.1:8000), create a project, and try a public dataset:

1. Open **Datasets**, choose **Hugging Face**, and enter `codywang/so101_pickup_test`.
2. Select **Inspect dataset**, pick an episode and press **Play**.

Install FFmpeg on the same machine to import local videos. [Getting started](docs/getting-started.md) covers local data, the CLI and troubleshooting. The app also serves the [workspace guide](docs/workspace-guide.md) at `/guide/` and its API reference at `/docs/`.

## Add what you need

The core app runs on any machine. Robots, GPUs and simulators are added separately, each in its own environment, so you install only what you use. **Settings → Modules** shows what each optional page needs and adds it to the sidebar once it is set up.

- **SO-101 arm:** the Robot Lab companion on the macOS or Linux computer the arm is plugged into. See [SO-101 Robot Lab](docs/so101-robot-lab.md).
- **Training on your GPU:** **Settings → Compute → Local runs** checks the machine and installs the SmolVLA worker (Linux, NVIDIA).
- **Training on cloud GPUs:** connect Google Cloud in **Settings → Compute**; jobs launch through SkyPilot.
- **ACT distillation, packing and replay, SmolVLA quantization, Isaac simulation:** each uses its own worker under `workers/`. Run `jensen install list` to see the installation profiles and `jensen install <profile> --dry-run` to preview one.

SmolVLA is the most complete training path; other policies, distillation, Q4 quantization, Isaac simulation, assembled-arm motor setup and Windows installs are experimental.

## Status and safety

OpenJensen is alpha software for a single user on their own machines. The server listens on localhost; use an SSH tunnel for remote access and read [SECURITY.md](SECURITY.md) before exposing it. Back up and upgrade as described in [upgrading](docs/upgrading.md).

## Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md), [AGENTS.md](AGENTS.md) and [writing documentation](docs/writing-docs.md).

## License

[Apache License 2.0](LICENSE); retained third-party notices are in [NOTICE](NOTICE).
