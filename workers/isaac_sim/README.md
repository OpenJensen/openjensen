# Run the Isaac recording worker

Prepare the GPU/project resources in [GCP setup](../skypilot/SETUP.md).
Use the worker's Isaac **6.1.0** container Python and FFmpeg/ffprobe.
For policy-controlled episodes follow [rollout setup](ROLLOUT.md); for recorded
state projection use [offline calibration](CALIBRATION_OFFLINE.md).

## Manifest and outputs

Copy `demo.yaml` to `demo.local.yaml`. Set `outputs.uri` to a private GCS prefix.
The template requests 180 frames at 1280×720 and 30 fps. Set `scene.uri` to an
existing `.usd`, `.usda` or `.usdc` and `scene.camera` to its camera prim.
Resolve relative scene paths from the manifest directory. Choose even H.264
image dimensions up to 1920×1080, FPS up to 60 and duration up to 600 seconds.

Read the UUID printed by each invocation under `outputs.uri`. Inspect its files:

```text
<uuid>/manifest.json  # normalized input, uploaded first
<uuid>/video.mp4      # H.264, yuv420p, fast-start metadata
<uuid>/result.json    # terminal status, video URI, SHA-256 and frame count
```

Read the terminal `result.json`, then compare its video URI, hash and frame
count to the saved video. Download partial logs if no terminal record appears.

## Custom scenes

Keep the scene and all relative dependencies together:

```text
workers/isaac_sim/jobs/my-scene/
├── job.local.yaml
├── scene.usda
└── assets/           # referenced layers, models and textures
```

From `workers/isaac_sim`, copy the manifest template:

```sh
mkdir -p jobs/my-scene
cp scene.example.yaml jobs/my-scene/job.local.yaml
```

Set `outputs.uri`, then supply your scene and existing camera:

```yaml
scene:
  uri: scene.usda
  camera: /World/Camera
```

Supply lighting, physics and any controller in the scene. Validate locally:

```sh
python -m sim_worker --manifest jobs/my-scene/job.local.yaml --validate-only
```

Submit with `SIM_MANIFEST=jobs/my-scene/job.local.yaml` using
[the launcher commands](../skypilot/README.md#custom-usda-jobs).

## Build

Complete [Cloud Build setup](../skypilot/SETUP.md#optional-cloud-build-setup).
From `workers/isaac_sim`:

```sh
export SIM_PROJECT_ID=your-project-id
export SIM_REGION=us-east4
bash build.sh
export SIM_BUILD_ID="$(cat build-id.txt)"
gcloud builds describe "$SIM_BUILD_ID" \
  --project="$SIM_PROJECT_ID" --region="$SIM_REGION" \
  --format='yaml(status,results.images,logUrl)'
```

Cloud Build uses `sim-builder@<project>.iam.gserviceaccount.com`, the
`<project>-sim-build-source` bucket, an 8-vCPU machine, 200-GB disk and one-hour
build timeout. After status `SUCCESS`, record the immutable image digest:

```sh
SIM_IMAGE_NAME="$(gcloud builds describe "$SIM_BUILD_ID" \
  --project="$SIM_PROJECT_ID" --region="$SIM_REGION" \
  --format='value(results.images[0].name)')"
SIM_IMAGE_DIGEST="$(gcloud builds describe "$SIM_BUILD_ID" \
  --project="$SIM_PROJECT_ID" --region="$SIM_REGION" \
  --format='value(results.images[0].digest)')"
printf '%s@%s\n' "${SIM_IMAGE_NAME%:*}" "$SIM_IMAGE_DIGEST" > worker-image.txt
```

Put that reference in the SkyPilot task. Resubmit after worker/manifest/scene
changes; rebuild the image after dependency or Isaac-version changes.

## Direct container execution

Use a Linux GPU host with a compatible graphics driver, NVIDIA Container
Toolkit, Docker, Artifact Registry authentication and a VM identity with GCS
access. Set `ACCEPT_EULA=Y` after accepting NVIDIA's container license.
Prepare the output owner and run:

```sh
export SIM_IMAGE="$(cat worker-image.txt)"
export SIM_OUTPUT_DIR="$PWD/results"
export SIM_RUNTIME_UID=1234
mkdir -p "$SIM_OUTPUT_DIR"
sudo chown "$SIM_RUNTIME_UID:$SIM_RUNTIME_UID" "$SIM_OUTPUT_DIR"
docker run --rm --gpus all \
  --env ACCEPT_EULA --env NVIDIA_DRIVER_CAPABILITIES=all \
  --mount "type=bind,src=$PWD,dst=/inputs,readonly" \
  --mount "type=bind,src=$SIM_OUTPUT_DIR,dst=/outputs" \
  "$SIM_IMAGE" --manifest /inputs/demo.local.yaml --output-dir /outputs
```

Use [SkyPilot](../skypilot/README.md) to submit remote jobs and manage VM lifetime.
Download output and logs before deleting its cluster.

## Local checks

Use an isolated Python **3.12** environment with PyYAML **6.0.3**:

```sh
python -m sim_worker --manifest demo.yaml --validate-only
python -m unittest discover -s tests -v
```

Install FFmpeg/ffprobe for encoding checks; set `SIM_REQUIRE_VIDEO_TOOLS=1` to
require those tools during the suite.

References: [Isaac container setup](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_container.html),
[Replicator troubleshooting](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/replicator_tutorials/troubleshooting.html),
[Cloud Build](https://docs.cloud.google.com/build/docs/build-config-file-schema).
