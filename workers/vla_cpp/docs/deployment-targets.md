# Configure an execution target

Select a target from [configs/targets.yaml](../configs/targets.yaml) before
building or running the configured benchmark:

| Target key | Build backend | Artifact |
| --- | --- | --- |
| `mac_cpu` | CPU | GGUF |
| `rtx_3070` | CUDA | GGUF |
| `gcloud_gpu` | CUDA | GGUF |

For `gcloud_gpu`, set `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_REGION` and
`POLICYKIT_GCLOUD_ACCELERATOR`. Record the selected GPU, driver and container image
digest in the run configuration. Build the native runtime for the selected backend
using the [CPU](cpu-benchmark.md) or [CUDA](gpu-setup.md) recipe.
