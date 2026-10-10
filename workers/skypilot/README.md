# Run Isaac on SkyPilot

Use Python **3.9–3.13**, Google Cloud CLI, `jq`, and the project resources in
[GCP setup](SETUP.md). Build [the Isaac image](../isaac_sim/README.md#build) and
record its immutable digest. For ACT/SmolVLA policy episodes use
[the rollout launcher](ROLLOUT.md) or [the configured runner account](RUNNER.md).

## Configure locally

From `workers/skypilot`:

```bash
export SIM_PROJECT_ID=your-gcp-project
export SIM_LAUNCHER=user:you@example.com
SIM_PYTHON=python3.12 bash install.sh
bash sky.sh gcloud auth login --update-adc
bash sky.sh gcloud auth application-default set-quota-project "$SIM_PROJECT_ID"
cp config.example.yaml config.yaml
cp task.example.yaml task.yaml
cp ../isaac_sim/demo.yaml ../isaac_sim/demo.local.yaml
```

Edit the copied files:

- Set the project in `config.yaml`.
- Set the project and immutable worker digest in `task.yaml`; retain
  `SIM_MANIFEST: demo.local.yaml`.
- Set the manifest's `outputs.uri` to your private results prefix.

The template selects `us-east4`, `g2-standard-16`, one L4, 64 GB RAM and a
200 GB disk. If changing region/network, update `configure.sh` and both local
YAMLs together. Replace literal placeholders in the YAMLs before checking:

```bash
# Creates IAM/network access; creates no VM. Review SETUP.md first.
bash sky.sh configure
bash sky.sh check gcp
```

Keep `SIM_PROJECT_ID` exported. After changing authentication, stop an older
local API server with `bash sky.sh api stop` before continuing.

## Submit and inspect

Accept [NVIDIA's container license](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim/license), then run:

```bash
# Creates or reuses the cluster and submits the first job.
SIM_MANIFEST=demo.local.yaml ACCEPT_EULA=Y bash launch.sh

# Subsequent job on a running cluster; setup is not repeated.
bash sky.sh exec isaac-sim task.yaml --env ACCEPT_EULA=Y \
  --env SIM_MANIFEST=demo.local.yaml

bash sky.sh queue isaac-sim
bash sky.sh logs isaac-sim
bash sky.sh status --refresh
bash sky.sh down isaac-sim
```

Set `SIM_CLUSTER` on `launch.sh` for another cluster name. Resubmit after source,
manifest or scene edits; rebuild the image after dependency or Isaac changes.

### Custom USDA jobs

Prepare [the scene bundle](../isaac_sim/README.md#custom-scenes) at
`../isaac_sim/jobs/my-scene/`, then submit from this directory:

```bash
# New or stopped cluster.
SIM_MANIFEST=jobs/my-scene/job.local.yaml ACCEPT_EULA=Y bash launch.sh

# Running cluster.
bash sky.sh exec isaac-sim task.yaml --env ACCEPT_EULA=Y \
  --env SIM_MANIFEST=jobs/my-scene/job.local.yaml
```

## Results and cleanup

Read `outputs.uri/<worker UUID>/result.json` and its `video.mp4`. Compare terminal
status, checksum and frame count. Read launcher logs under
`SIM_RESULTS_URI/<launcher UUID>/`; retain both identifiers.

Download logs before cluster cleanup:

```bash
bash sky.sh logs isaac-sim --sync-down
```

Use `bash sky.sh status --refresh` to inspect provisioning and idle resources,
then `bash sky.sh down CLUSTER_NAME` to delete the selected cluster. The job
has a one-hour timeout and cluster idle autodown is 15 minutes; inspect resources
after failures rather than relying on those timers.

For a renderer check on an existing cluster, copy `probe.example.yaml` to
`probe.yaml`, fill its image placeholders and submit with
`bash sky.sh exec isaac-sim probe.yaml --env ACCEPT_EULA=Y`. Preserve the PNGs and
statistics in `~/sim-debug/` before cleanup.

## Local checks

```bash
python3 -m unittest discover -s tests -v
```

References: [task YAML](https://docs.skypilot.ai/en/v0.13.0/reference/yaml-spec.html),
[autodown](https://docs.skypilot.ai/en/v0.13.0/reference/auto-stop.html),
[host image catalog](https://github.com/skypilot-org/skypilot-catalog/blob/master/catalogs/v8/gcp/images.csv).
