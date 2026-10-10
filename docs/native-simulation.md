# Run ACT or SmolVLA in Isaac

## Operator and API integration

Prepare an Isaac runner using [runner setup](../workers/skypilot/APP_RUNNER.md). Save a private profile file and set `FIREBIRD_SIMULATION_CONFIG` to its absolute path before starting the API.

Prepare a complete ACT/SmolVLA policy TAR with safetensors weights, configuration, saved processors and normalization statistics. Use an archive within 4 GiB. For a training checkpoint, export/materialize its inference package first.

## Import and start

1. Open **Run → 3D simulation**.
2. Select the configured scene/profile and a saved compatible policy, or choose **Import package** and upload its TAR.
3. After import, select the registered policy explicitly.
4. Set a timeout from 30–7200 seconds, review paid-run consent, and start.
5. Follow the saved job's status/events. Open its video or download the output when available.

For direct clients, read `GET /api/v1/simulation-options`, upload the raw TAR to `POST /api/v1/projects/{id}/model-imports?profile_id=...`, then submit `operation: policy.run` to the policy-job endpoint with the artifact ID and `simulation: {profile_id, experimental: true}`.

## Cancel or troubleshoot

Cancel the exact selected job and inspect its cleanup status. After a lost upload/start response, refresh project history before another request. For profile/import errors, check the configured interpreter, source paths and complete policy inventory. The video is served when output contains `artifacts/outputs/video.mp4`.
