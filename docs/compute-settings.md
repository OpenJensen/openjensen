# Compute settings

Open **Settings & diagnostics → Compute**. Preferences are saved in the workspace's `compute-settings.json`.

## Local worker setup

Use Linux x86-64, a visible NVIDIA GPU, `uv`, and at least 24 GiB free disk space for automatic SmolVLA setup.

1. Open **Local runs** and select **Check this machine**.
2. Select **Set up locally** to install the pinned Python 3.11/CUDA 12.6 environment, or **Add worker** to register an existing installation.
3. Follow installation progress. Use **Cancel setup** to stop it or **Retry setup** to continue a partial installation.
4. Enable local runs and save.
5. Select the added target under **Fine-tune → SmolVLA**.

Automatic setup stores its environment under `local-environments/smolvla-cu126` in the workspace. Installer logs are in `local-environments/install.log`. Install [CPU workers](../workers/local_cpu/README.md) separately for ACT export, distillation, packing and replay.

## Compute preferences API

Use `GET /api/v1/compute-settings` to read preferences. Send either or both `local` and `gcp` objects with `PUT` to save them.

| Setting | Value |
| --- | --- |
| Local `enabled` | Set true to allow local runs. |
| Local `label` | Printable machine label, 1–100 characters. |
| GCP `enabled` | Set true to allow cloud training. |
| GCP `default_gpu` | `L4`, `T4` or `A100`; default `A100`. |
| GCP `disk_size_gb` | 100–2000; default 200. |
| GCP `idle_minutes` | 1–60; default 10. |

Use `GET /api/v1/compute-settings/local/setup` for setup progress, `POST` to start/resume setup, and `POST /api/v1/compute-settings/local/setup/cancel` to stop it.

## Google Cloud through SkyPilot

1. Install SkyPilot 0.13 with GCP support and Google Cloud CLI on the API host; run `gcloud auth application-default login` to configure application-default credentials.
2. [Connect the project and region](cloud-connections.md).
3. Select `L4` (24 GB, `g2-standard-4`), `T4` (16 GB, `n1-highmem-4`) or `A100` (40 GB, `a2-highgpu-1g`).
4. Open **Fine-tune**, review the recipe and start.
5. Follow preparation in the saved job. If it fails, follow its setup instructions and submit a new request.

Use `POST /api/v1/compute-settings/gcp/check` to inspect prerequisites. Review `gcp_status` and `gpu_options` from the preferences response. Cloud runtime IDs are `skypilot-gcp-` followed by the GPU name.

See Google's [GPU machine types](https://docs.cloud.google.com/compute/docs/gpus) and [GPU regions](https://docs.cloud.google.com/compute/docs/regions-zones/gpu-regions-zones), and the [training procedure](skypilot-training.md).

## Existing native workers

Set `FIREBIRD_RUNTIME_CONFIG` to an operator runtime file. Supply each worker's installed interpreter, source root and model assets. Use `provider: local` or `provider: gcp`, and the same provider for entries on one application host. Add an optional `region` for the host. Follow [native runtime setup](policy-workflow.md#configure-a-local-native-execution-host).
