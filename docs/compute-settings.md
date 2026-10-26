# Compute settings

**Settings → Compute** chooses where training, inference and simulation run: your own NVIDIA GPU, Google Cloud GPUs, or workers you configure yourself. Preferences are saved in the workspace's `compute-settings.json`.

You can inspect datasets before setting up any compute. A passing setup check means the configuration is ready, not that a model trained well; see [what runs where](support.md).

## Train on your own GPU

You need Linux x86-64, a visible NVIDIA GPU, `uv`, and at least 24 GiB of free disk for the automatic SmolVLA setup.

1. Open **Local runs** and select **Check this machine**.
2. Select **Set up locally** to install the pinned Python 3.11 / CUDA 12.6 environment, or **Add worker** to register an existing installation.
3. Follow the progress. **Cancel setup** stops it; **Retry setup** continues a partial installation.
4. Enable local runs and save.
5. Choose the new target under **Fine-tune → SmolVLA**.

The environment is installed in the workspace under `local-environments/smolvla-cu126`, with its log in `local-environments/install.log`. ACT export, distillation, packing and replay use the separate [CPU workers](../workers/local_cpu/README.md).

## Train on Google Cloud

Cloud jobs launch through SkyPilot.

1. On the server, install SkyPilot 0.13 with GCP support and the Google Cloud CLI, then run `gcloud auth application-default login`.
2. [Connect the project and region](cloud-connections.md).
3. Choose a GPU:

   | GPU | Memory | Machine type |
   | --- | --- | --- |
   | `L4` | 24 GB | `g2-standard-4` |
   | `T4` | 16 GB | `n1-highmem-4` |
   | `A100` | 40 GB | `a2-highgpu-1g` |

4. Open **Fine-tune**, review the recipe and start.
5. Follow preparation in the saved job. If it fails, follow its setup instructions and submit a new request.

Cloud runtime IDs are `skypilot-gcp-` followed by the GPU name. See Google's [GPU machine types](https://docs.cloud.google.com/compute/docs/gpus) and [GPU regions](https://docs.cloud.google.com/compute/docs/regions-zones/gpu-regions-zones), and [cloud training](skypilot-training.md) for the full procedure.

## Use workers you configure

Set `JENSEN_RUNTIME_CONFIG` to a runtime file listing each worker's installed interpreter, source root and model assets. Use `provider: local` or `provider: gcp`, the same provider for every entry on one server, and an optional `region`. See [configure the execution host](policy-workflow.md#configure-the-execution-host).

## API

| Request | Purpose |
| --- | --- |
| `GET /api/v1/compute-settings` | Read preferences, plus `gcp_status` and `gpu_options` |
| `PUT /api/v1/compute-settings` | Save a `local` and/or `gcp` object |
| `GET /api/v1/compute-settings/local/setup` | Local setup progress |
| `POST /api/v1/compute-settings/local/setup` | Start or resume local setup |
| `POST /api/v1/compute-settings/local/setup/cancel` | Stop local setup |
| `POST /api/v1/compute-settings/gcp/check` | Check Google Cloud prerequisites |

| Setting | Value |
| --- | --- |
| `local.enabled` | `true` allows local runs |
| `local.label` | A printable machine label, 1–100 characters |
| `gcp.enabled` | `true` allows cloud training |
| `gcp.default_gpu` | `L4`, `T4` or `A100`; default `A100` |
| `gcp.disk_size_gb` | 100–2000; default 200 |
| `gcp.idle_minutes` | 1–60; default 10 |
