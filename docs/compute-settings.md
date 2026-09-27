# Compute settings

**Settings & diagnostics → Compute** connects cloud providers, adds compatible
local training workers, and stores the local compute switch and machine label.
Local runs are disabled by default and require an explicit opt-in. Existing saved
local choices remain intact. These
preferences persist in the workspace's `compute-settings.json`; disabling local
compute blocks new native policy jobs and queued or later workflow stages before
their worker starts. An already running worker is not stopped by this switch.
Dataset inspections and augmentation have their own execution paths and are not
controlled by the native compute preference.

## Add a local training worker

1. Open **Local runs** and select **Check this machine**.
2. Review the detected worker and **App host** name, then select **Add worker**.
3. Select **Enable local runs** and **Save local settings** when ready.
4. Choose the added compute target under **Fine-tune → SmolVLA**.

The check runs on the machine hosting the application, which can differ from the
computer running your browser. Opening Compute does not run a check. **Check again**
refreshes the result; **Added** means the worker is already registered. Adding a
worker preserves the saved local switch and label, including a disabled switch.

The initial discovery adapter supports an existing SmolVLA worker at
`workers/smolvla_qlora/.venv/bin/python` in the application's source checkout,
on Linux x86_64 with an NVIDIA GPU of compute capability 7.0 or newer. It checks
the first GPU visible to that worker, respecting the application's
`CUDA_VISIBLE_DEVICES` selection, and requires a stable GPU UUID to keep the saved
worker bound to that device across restarts. It does not enumerate remote machines, scan
arbitrary Python environments, or add other model families. Packaged app installs
without the worker source are not supported by this discovery path.

Install dependencies separately using the
[pinned SmolVLA installation](../workers/smolvla_qlora/docs/training/smolvla-qlora.md#install-on-the-training-machine),
then check again. The lightweight development install in the worker README is
not enough for CUDA training. Readiness checks require an isolated Python 3.11 or
3.12 environment with LeRobot 0.4.4, Torch 2.7.1, torchvision 0.22.1,
Transformers 4.57.1, PEFT 0.18.0 and bitsandbytes 0.48.2, plus successful imports
and CUDA access. The linked installation uses the pinned Linux dependency lock.

Check and Add perform bounded, offline environment checks; they install nothing,
download no models or datasets, and start no training or inference jobs. Add
rechecks the worker before saving. A ready result describes the installed
environment and GPU capacity, not model access, free VRAM, dataset compatibility
or a successful training run. Missing prerequisites show **Setup needed** with a
link to setup guidance.

Added workers persist in the workspace's `local-workers.json`, separate from
`compute-settings.json` and the operator's `FIREBIRD_RUNTIME_CONFIG`. The app merges
managed workers with operator entries while preserving their configured sources;
equivalent workers are reused and conflicting operator entries take precedence.
The app never rewrites the operator's runtime configuration. Managed workers have
`training_only: true` and accept fine-tuning only. Export, quantization,
evaluation and Run require separately configured compatible workers.

`POST /api/v1/compute-settings/local/check` accepts no body and returns the app-host
identity, check time, candidate readiness and any setup issues.
`POST /api/v1/compute-settings/local/workers` accepts only `candidate_id` from a
recent check and returns the registered runtime, current compute preferences and
updated discovery result. The server determines executable paths and GPU identity;
the browser cannot supply worker paths or commands.

## Compute preferences API

`GET /api/v1/compute-settings` returns local and Google Cloud preferences, safe
runtime descriptions, `gcp_status`, and `gpu_options`. `PUT` accepts either or both
top-level `local` and `gcp` objects; saving one preserves the other. Local labels
are trimmed printable text from 1 to 100 characters. GCP preferences are
`enabled` (default true), `default_gpu` (default `A100`), `disk_size_gb` (default
200, range 100–2000), and `idle_minutes` (default 10, range 1–60). Existing saved
local settings remain compatible. A saved Google Cloud connection is required;
connecting also enables cloud training while preserving the chosen GPU and disk.

Policy options include `compute: {local, gcp}`. Runtime descriptions include
`execution`, `provider`, `provider_label`, `region`, `enabled`, and an optional
`unavailable_reason`. Cloud runtimes distinguish `launchable` (eligible for Start)
from `needs_preparation`; GPU option `available` means setup has actually passed.
Disabled runtimes do not make training models available.
Responses never include worker executables, paths, environment variables, or
credential material.

## Google Cloud through SkyPilot

Connect Google Cloud to choose the project and region, then select a GPU and start
training. There is no separate required preparation or verification action.
The request captures its target using a short local SkyPilot configuration read,
then queues promptly. The background job prepares its project-specific workspace,
checks credentials and regional offerings, and launches the selected GPU.
Preparation can enable required Google Cloud APIs through SkyPilot's supported
workspace API. Existing default and other workspaces are preserved; OPEN JENSEN never
overwrites a mismatched workspace name. A matching existing workspace is reused
and its GCP compute access is checked automatically.

The optional diagnostic `POST /api/v1/compute-settings/gcp/check` remains read-only. It
verifies the installed CLI's Google Cloud dependencies, application-default
credentials, required enabled APIs, regional GPU catalog, and the server's exact
project-pinned workspace. It does not create machines, launch jobs, enable APIs,
or edit SkyPilot configuration. SkyPilot may start its local API server to read
the catalog or workspace. Setup commands are shown only for a failed prerequisite.

Verification lasts for the current application session and is invalidated on
reconnection or when the saved project or region changes. A failed preparation is
saved for diagnosis across restarts, without blocking selection for a new attempt.
Availability means the reviewed machine is
listed in SkyPilot's catalog; it does not guarantee GPU quota, capacity, billing,
or permission to launch. Unverified prerequisites are handled during the queued run.

The one-GPU on-demand choices are `L4` (24 GB, `g2-standard-4`), `T4`
(16 GB, `n1-highmem-4`), and `A100` (40 GB, `a2-highgpu-1g`). T4's host has
4 vCPUs and 26 GB memory, satisfying the cloud recipe's host-memory requirement. Their stable
runtime IDs are `skypilot-gcp-` followed by the GPU name. Synthetic runtime IDs and
execution commands are managed by the application, not the runtime config or API
caller. Each accepted run captures project, region, GPU, machine type, disk size,
idle timeout, workspace identity, and SkyPilot API server address so later edits
cannot redirect that run. The runner re-verifies the saved workspace on that exact
server immediately before launch and refuses to provision if its project or
server differs; it never falls back to a default workspace. Saved server addresses
cannot contain credentials, query strings, or fragments.

The GPU cards are limited to these three choices. Older saved `A100-80GB`
preferences migrate to `A100`; existing job records retain their original target.
T4 uses FP16 compute, floating NF4 storage, FP32 trainable parameters, and dynamic
gradient scaling. L4 and A100 retain native BF16 compute. Checkpoints preserve
their compute precision and scaler state; resuming across different compute
precisions is rejected. GPU memory fit depends on the dataset and batch size.
See Google's [GPU machine types](https://docs.cloud.google.com/compute/docs/gpus)
and [GPU regions](https://docs.cloud.google.com/compute/docs/regions-zones/gpu-regions-zones).

## Existing native workers

Workers beyond the discovery adapter use the operator-owned
`FIREBIRD_RUNTIME_CONFIG`. Each worker
may declare `provider` (`local` or `gcp`) and an optional `region`. Existing
entries default to `local`. The saved local label appears as `provider_label`;
the worker's own label and GPU specifications remain those in its runtime config.
For these native entries, provider metadata identifies where an operator has
installed the app and workers. It does not change their host/container execution
path. SkyPilot targets are configured separately through compute settings.
Use the same provider on all worker entries belonging to one application host.
