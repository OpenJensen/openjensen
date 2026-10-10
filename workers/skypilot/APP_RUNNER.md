# Configure the application simulation runner

Prepare [the SkyPilot runner](RUNNER.md), its isolated `.venv`, private
`config.yaml`, Isaac image and complete ACT/SmolVLA checkpoint imports.

## Register a profile

Save a private server-side JSON configuration using this shape:

```json
{
  "schema_version": 1,
  "profiles": [{
    "id": "so101-cup-episode-001",
    "label": "SO101 cup · experimental",
    "runner_root": "/private/runner",
    "task": "/private/runner/workers/skypilot/rollout.episode-001.local.yaml",
    "task_sha256": "REPLACE_WITH_THE_EXACT_TASK_SHA256",
    "project_id": "your-google-project",
    "credential_file": "/private/credentials/service-account.json",
    "gcloud_config": "/private/runner-gcloud-state",
    "results_uri": "gs://your-private-results/cup-experiments",
    "accept_eula": true
  }]
}
```

Replace every path/project/hash/result prefix. Match task `SIM_RESULTS_URI` to
`results_uri`; set `accept_eula` only after the operator accepts NVIDIA's license.
Keep the task's scene/calibration files and source bundle under `runner_root`.
Use six native state/action coordinates, one named RGB camera, even image
dimensions from 2 through 1920 and the checkpoint's exact model fingerprint.

Point `credential_file` and `gcloud_config` to the configured private account
files. An optional `sky_api_endpoint` may select an already configured endpoint.
Set the configuration path on the application host before serving:

```sh
export FIREBIRD_SIMULATION_CONFIG=/private/simulation-profiles.json
uv run --frozen firebird serve
```

## Submit and follow

In **Run → 3D simulation**, select the profile and a complete compatible policy,
review the preparation form and launch. Use a new job/output directory and an
integer application timeout from 30 through 7200 seconds. Follow the job's
status, events and video/download outputs in the application.

For SDK integration, use `load_profiles(path)`, then
`admit(profile, checkpoint_metadata)`, and invoke
`run(profile, checkpoint_directory, job_directory, event, timeout_seconds,
expected_profile_sha256=..., expected_model_id=...)`. The event callback accepts
`(stage, message, data_or_none)`; retain returned relative artifact paths,
SHA-256, byte counts and GCS generations with the owning job.

## Cancel and reconcile

Cancel the owning application job rather than submitting a replacement. Retain
its `launch-context.json`, submission/group IDs and result prefix. On app
interruption use `recover(profile, job_directory)` with the unchanged profile
identity. Reconcile changed/missing identities manually. Inspect the saved group
and cloud resource inventory after cancellation, then stop unused controllers
with [the launcher cleanup commands](ROLLOUT.md#results-and-lifecycle).
