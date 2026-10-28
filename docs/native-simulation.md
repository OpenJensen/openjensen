# Run ACT or SmolVLA in Isaac

Roll out a saved ACT or SmolVLA policy in an Isaac Sim scene and keep a recording of the run. Rollouts are experimental and run on paid Google Cloud GPUs.

## What you need

- An Isaac runner prepared with the [runner setup](../workers/skypilot/APP_RUNNER.md).
- A complete ACT or SmolVLA policy: safetensors weights (packed ACT weights for a `packed-act-cpu` profile), configuration, saved processors and normalization statistics. To upload one, use a TAR within 4 GiB.
- For a training checkpoint, export or materialize its inference package first.

## Register a scene profile

Save a private profile file and set `JENSEN_SIMULATION_CONFIG` to its absolute path before starting the API. The runner setup shows the file's shape. A profile's optional `policy_runtime` decides which policies it runs:

| `policy_runtime` | Policies | Policy device | Cloud GPU |
| --- | --- | --- | --- |
| `lerobot-cuda` (default) | ACT or SmolVLA with safetensors weights | CUDA | L4 or H100 |
| `packed-act-cpu` | Packed ACT (`model.fbq` and `encoding.json`) | CPU | L4 |

See the [packed ACT profile](../workers/skypilot/ROLLOUT.md#separate-packed-act-cpu-profile) for its task file.

## Start a rollout

1. Open **Run → 3D simulation**.
2. Under **Scene**, choose the configured profile.
3. Under **Policy**, choose **Saved policies** and select a compatible policy. Or choose **Import package**, pick the **Native policy TAR** and select **Upload and validate policy**. Import saves and validates the package on the server; no GPU job starts.
4. After an import, select the newly registered policy explicitly.
5. Set **Timeout (seconds)** from 30 to 7200. The timeout is not a spending cap.
6. Review the paid target and **Cloud limits**, tick **I understand this is an experimental, paid cloud rollout.** and select **Start experimental simulation**.
7. Follow the saved job's status and events. Watch the video or select **Download simulation record** when it is available.

A packed ACT package from **Quantize** opens here through **Prepare simulation**.

## What a rollout shows

- The recording shows execution only: cup pickup success is not measured and calibration is not verified.
- Cancellation is supervised, but resource deletion is not verified.
- The video is served only when the output contains `artifacts/outputs/video.mp4`.

## Use the API

1. Read `GET /api/v1/simulation-options` for the profiles and upload limit.
2. Upload the raw TAR to `POST /api/v1/projects/{id}/model-imports?profile_id=...`. Uploads do not accept an `Idempotency-Key`.
3. Submit `operation: policy.run` to `POST /api/v1/projects/{id}/policy-jobs` with the artifact ID, `runtime_id` set to the profile ID, and `simulation: {profile_id, experimental: true}`.

## Cancel or troubleshoot

- **Cancel:** select **Cancel selected native job**, then **Confirm cancellation**, and inspect the job's cleanup status. The [launcher cleanup commands](../workers/skypilot/ROLLOUT.md#results-and-lifecycle) check for leftover cloud resources.
- **Lost upload or start response:** refresh the project history before another request.
- **Profile or import errors:** check the configured interpreter, source paths and complete policy inventory.
