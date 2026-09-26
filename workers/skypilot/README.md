# Isaac on SkyPilot

Run the standalone Isaac worker on a GCP L4 from a local SkyPilot client.
This launcher has **no integration with the repository's API or job database**.
Related to [SKY-001 (#26)](https://github.com/sobhanb-eth/firebird-hackathon-codebase/issues/26);
application recipe integration and acceptance remain open.

**Known limitation:** a GPU run produced an all-black MP4. The worker rejects
black builtin-demo frames, but the rendering cause remains unresolved. Passing
host checks and unit tests does not establish successful rendering.

```text
Local task + simulation manifest
              |
        SkyPilot client
              |
       GCP L4 VM + Docker
              |
    Isaac worker -> MP4 + result.json in GCS
```

## Configure locally

Requires Python 3.9–3.13, Google Cloud CLI, `jq`, and the GCP resources in
[SETUP.md](SETUP.md). Docker and Isaac run on the remote VM.

From this directory:

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

Edit the ignored local files before proceeding:

- `config.yaml`: replace `CHANGE_ME_PROJECT_ID` with `SIM_PROJECT_ID`.
- `task.yaml`: replace the project and worker image digest placeholders. Use
  the immutable digest from your build; keep `SIM_MANIFEST: demo.local.yaml`.
- `../isaac_sim/demo.local.yaml`: set `outputs.uri` to your results bucket,
  for example `gs://YOUR_PROJECT-sim-results/runs`.

The task defaults to `us-east4`, `g2-standard-16`, one L4, 64 GB RAM, and a 200 GB
disk. The pinned public SkyPilot image supplies the driver and Docker runtime.
If changing the region or network, update `configure.sh` and both local YAMLs
together. YAML placeholders are literal; shell variables are not substituted.

```bash
# Creates IAM/network access; creates no VM. Review SETUP.md first.
bash sky.sh configure
bash sky.sh check gcp
```

`sky.sh` uses this directory's `.venv`, optional `.tools/google-cloud-sdk`, and
configuration without changing global gcloud settings. Keep `SIM_PROJECT_ID`
exported for subsequent commands. Existing SkyPilot API servers may require
`bash sky.sh api stop` after authentication changes.

## Submit and inspect

After accepting the [NVIDIA container license](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim/license):

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

Set `SIM_CLUSTER` on `launch.sh` to use another cluster name. A local API server
does not automatically adopt clusters created by a different Cloud Shell client.

SkyPilot syncs `../isaac_sim`; each job snapshots that source and mounts it
read-only over `/opt/sim-worker`. Code, manifest, and local scene changes need
another submission. Dependency or Isaac version changes need a new image.
Setup installs a missing Vulkan loader and checks graphics dependencies without
replacing the NVIDIA driver.

## Results and cleanup

The MP4 remains at the manifest's `outputs.uri/<worker UUID>/video.mp4`;
`result.json` in that directory must report `status: succeeded`. The launcher
checks that record, nonempty video, and checksum. Its diagnostics go to
`SIM_RESULTS_URI/<launcher UUID>/`; the two UUIDs differ.

The job times out after one hour. SkyPilot deletes the cluster after 15 idle
minutes. A persistent guest timer also requests deletion 48 hours after VM
creation; this depends on the guest OS and identity and is not a spending cap.
Provisioning failures before setup can require explicit `sky down`.

Download logs before cleanup if GCS publication fails:

```bash
bash sky.sh logs isaac-sim --sync-down
```

`probe.example.yaml` is an **experimental** renderer diagnostic for an existing
cluster. Copy it to `probe.yaml`, fill its image placeholders, then submit with
`bash sky.sh exec isaac-sim probe.yaml --env ACCEPT_EULA=Y`. It writes PNGs and
statistics under `~/sim-debug/` on the VM, not worker results or MP4s in GCS.

## Validation

```bash
python3 -m unittest discover -s tests -v
```

GPU acceptance still requires a visible falling-cube MP4, correct frame count
and checksum, and confirmed teardown. L4 quota does not guarantee capacity.

References: [task YAML](https://docs.skypilot.ai/en/v0.13.0/reference/yaml-spec.html),
[autodown](https://docs.skypilot.ai/en/v0.13.0/reference/auto-stop.html),
[host image catalog](https://github.com/skypilot-org/skypilot-catalog/blob/master/catalogs/v8/gcp/images.csv).
