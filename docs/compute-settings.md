# Compute settings

Settings stores the local compute switch and a display label for this machine.
Local runs are disabled by default and require an explicit opt-in. Existing saved
local choices remain intact. These
preferences persist in the workspace's `compute-settings.json`; disabling local
compute blocks new native policy jobs and queued or later workflow stages before
their worker starts. An already running worker is not stopped by this switch.
Dataset inspections and augmentation have their own execution paths and are not
controlled by the native compute preference.

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

The dropdown is limited to these three GPU choices. Older saved `A100-80GB`
preferences migrate to `A100`; existing job records retain their original target.
T4 uses FP16 compute, floating NF4 storage, FP32 trainable parameters, and dynamic
gradient scaling. L4 and A100 retain native BF16 compute. Checkpoints preserve
their compute precision and scaler state; resuming across different compute
precisions is rejected. GPU memory fit depends on the dataset and batch size.
See Google's [GPU machine types](https://docs.cloud.google.com/compute/docs/gpus)
and [GPU regions](https://docs.cloud.google.com/compute/docs/regions-zones/gpu-regions-zones).

## Existing native workers

Workers still come from the operator-owned `FIREBIRD_RUNTIME_CONFIG`. Each worker
may declare `provider` (`local` or `gcp`) and an optional `region`. Existing
entries default to `local`. The saved local label appears as `provider_label`;
the worker's own label and GPU specifications remain those in its runtime config.
For these native entries, provider metadata identifies where an operator has
installed the app and workers. It does not change their host/container execution
path. SkyPilot targets are configured separately through compute settings.
Use the same provider on all worker entries belonging to one application host.
