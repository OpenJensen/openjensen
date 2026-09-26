# Deployment targets

PolicyKit supports three planned execution lanes: the local Intel Mac CPU, an NVIDIA RTX 3070, and Google Cloud GPU workers. The source of truth is [`configs/targets.yaml`](../configs/targets.yaml).

Each job must select one target before build or evaluation. A run records its target label, backend, accelerator/driver (when CUDA is used), runtime version, and model checksum. Results from different target lanes must never be averaged or ranked against each other: task-success changes may be compared cautiously, while latency and memory are target-specific.

## Deployment selection contract

| Target | Intended use | Runtime | Artifact |
|---|---|---|---|
| `mac_cpu` | Development and CPU fallback | vla.cpp CPU | GGUF |
| `rtx_3070` | Local GPU benchmark and demo | vla.cpp CUDA | GGUF |
| `gcloud_gpu` | Scalable evaluation/training workers | vla.cpp CUDA | GGUF |

For GCloud, deployment must require `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_REGION`, and `POLICYKIT_GCLOUD_ACCELERATOR`; do not assume a particular GPU model. The selected accelerator, CUDA driver, and container image digest belong in the run manifest before results are published.
