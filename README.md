# OpenJensen

An open-source workspace for robot learning. Record demonstrations, manage datasets and models, train policies, and run them from a self-hosted web app.

## Run

Use Python 3.14.7, uv 0.12.19, Node 24.21.0 and pnpm 12.6.0.

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

Open [localhost:8000](http://127.0.0.1:8000). The app includes a [workspace guide](docs/workspace-guide.md) at `/guide/` and its API reference at `/docs/`.

## Workspace

- Set up and teleoperate an SO-101 arm, then record demonstrations.
- Inspect and import datasets, manage saved models, and generate appearance variations.
- Fine-tune, distill and quantize policies with configured workers.
- Evaluate policies, replay observations, and run simulation.
- Manage local workers, cloud connections, saved jobs and application updates.

The app, model workers, simulators and robot companion use separate environments. Optional modules are managed in **Settings → Modules**. Project data and configuration stay in the persistent workspace.

## Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md), [AGENTS.md](AGENTS.md) and [SECURITY.md](SECURITY.md).

## License

[Apache License 2.0](LICENSE); retained third-party notices are in [NOTICE](NOTICE).

Calibration files can be transferred between computers and profile names from Initial setup → Devices → a device → Calibration. See [device setup](docs/device-workspace.md). See [calibration transfer](docs/calibration-transfer.md).
