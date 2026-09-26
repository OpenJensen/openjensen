# Isaac recording worker

Records simulations with Isaac's RTX real-time renderer (RT2). The bundled
falling-cube scene produced a verified recording on GCP L4. Offline path
tracing is not required.

```text
YAML -> service -> Isaac adapter -> USD physics + RGB frames
               -> video adapter -> FFmpeg -> MP4
               -> artifact adapter -> GCS -> result.json
```

The CLI calls the service; SDK and subprocess details stay in adapters.
[SkyPilot](../skypilot/README.md) owns VM provisioning and cleanup.
The worker uses Isaac's container Python, separate from the application's
Python 3.14 environment. No simulator dependency is added to core, and no
application operation is registered by this transfer.

## Manifest and outputs

`demo.yaml` requests 180 frames at 1280×720/30 fps. Its output bucket is an
example. Copy it to `demo.local.yaml` and set `outputs.uri` to your GCS prefix.
The bundled scene contains a falling cube, ground, light and camera.

For another scene, use an absolute container USD path and camera prim path.
All scene dependencies must be mounted. GCS scene staging and policy execution
are not implemented. Validation rejects unknown/duplicate keys, YAML aliases,
unsupported URIs and invalid capture values. Limits are 1920×1080, 60 fps and
600 seconds; H.264 dimensions must be even.

Each invocation creates a UUID below `outputs.uri`:

```text
<uuid>/manifest.json  # normalized input, uploaded first
<uuid>/video.mp4      # H.264, yuv420p, fast-start metadata
<uuid>/result.json    # terminal status, video URI, SHA-256 and frame count
```

Consumers require a final `succeeded` result. Uploads are create-only; completion
is written last, before Isaac shutdown. Failures attempt to publish a failure
record. VM loss can leave no terminal record. Retries create new UUIDs.

The renderer uses `RealTimePathTracing` (RT2), DLSS quality mode, one capture
subframe, and disabled responsive denoising to avoid speckled output. An
eight-subframe warmup runs at simulation time zero before recording. Startup
and warmup took about eight minutes on the tested L4 host.

The worker rejects entirely black builtin-demo output before publishing
success. This guard does not validate motion or custom scenes, which may be dark.

## Build

Prerequisites are listed in [GCP setup](../skypilot/SETUP.md). From this directory:

```sh
export SIM_PROJECT_ID=your-project-id
export SIM_REGION=us-east4
bash build.sh
export SIM_BUILD_ID="$(cat build-id.txt)"
gcloud builds describe "$SIM_BUILD_ID" \
  --project="$SIM_PROJECT_ID" --region="$SIM_REGION" \
  --format='yaml(status,results.images,logUrl)'
```

Cloud Build uses `sim-builder@<project>.iam.gserviceaccount.com`, the project's
`<project>-sim-build-source` bucket, an 8-vCPU build machine, 200-GB disk and
one-hour timeout. It resolves the Isaac 6.1.0 base tag to a digest, builds, tests
without a GPU, then publishes to Artifact Registry's `simulation` repository.
Direct Python requirements are pinned; OS and transitive dependencies are not
fully locked. Build success does not prove GPU rendering.

After `SUCCESS`, retain the image digest:

```sh
SIM_IMAGE_NAME="$(gcloud builds describe "$SIM_BUILD_ID" \
  --project="$SIM_PROJECT_ID" --region="$SIM_REGION" \
  --format='value(results.images[0].name)')"
SIM_IMAGE_DIGEST="$(gcloud builds describe "$SIM_BUILD_ID" \
  --project="$SIM_PROJECT_ID" --region="$SIM_REGION" \
  --format='value(results.images[0].digest)')"
printf '%s@%s\n' "${SIM_IMAGE_NAME%:*}" "$SIM_IMAGE_DIGEST" > worker-image.txt
```

Use this immutable reference in the SkyPilot task. SkyPilot mounts the current
worker source into the image, so Python, scene and manifest changes need no
rebuild. Dependency or Isaac-version changes do.

## Direct container execution

On a prepared Linux GPU host, provide a compatible graphics driver, NVIDIA
Container Toolkit, Docker, registry authentication and a VM service account
with GCS access. Set `ACCEPT_EULA=Y` only after accepting NVIDIA's license.

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

FFmpeg encodes on CPU. The service verifies MP4 metadata and publishes results
before Isaac's potentially process-terminating shutdown. The worker does not
delete the VM. Use SkyPilot for managed lifetimes and diagnostics.

## Verify

Use an isolated Python 3.12 environment with PyYAML 6.0.3 for local tests:

```sh
python -m sim_worker --manifest demo.yaml --validate-only
python -m unittest discover -s tests -v
```

FFmpeg/ffprobe are required for three encoding tests. Set
`SIM_REQUIRE_VIDEO_TOOLS=1` to require them, as CI and Cloud Build do. SDK mocks
verify adapter contracts and failure publication, not rendering or physics.

GPU acceptance requires visible falling motion, nonblack frames, a six-second
MP4 at 1280×720/30 fps, matching GCS checksum and terminal result, and verified
VM cleanup. Repeat acceptance after renderer, driver or simulator changes.

On 2026-09-26, Isaac 6.1.0 with driver 580.159.04 on GCP L4 produced all 180
visible frames at 1280×720/30 fps with cube motion. The downloaded GCS MP4
matched the terminal result's SHA-256. All 22 worker tests passed in the worker
container, including FFmpeg encoding; all 41 launcher tests passed locally.
Container removal was verified. Full VM teardown remains unverified.

References: [Isaac container setup](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_container.html),
[Replicator troubleshooting](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/replicator_tutorials/troubleshooting.html),
[Cloud Build](https://docs.cloud.google.com/build/docs/build-config-file-schema).
