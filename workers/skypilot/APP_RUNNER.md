# Application-owned native simulation

The application adapter in `packages/core/src/vla_platform/lifecycle/isaac_runner.py`
uses the existing launcher for complete native ACT or SmolVLA policies. It does
not route these policies through GGUF. This is an **experimental cup rollout**;
completion means both owned tasks finished and their model/manifest-bound video,
trajectory and completion records were collected. Pickup success remains unknown.
The historical dataset/policy prompt may say “cube”; changing that language is a
separate experiment, not an automatic correction to recorded provenance.

An operator registers an isolated, reviewed runner overlay using a private JSON
configuration (paths below are examples, not an installed environment):

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

The task's `SIM_RESULTS_URI` must match `results_uri`. Its manifest must select
the reviewed SO101 cup scene, six native joint coordinates, compatible camera
geometry and experimental calibration. The launcher still performs its complete
local manifest/network/resource/model checks. Its two workers retain the existing
L4 simulator and H100 policy-server topology. This profile is separate from the
application's single-GPU training selection. No new provider configuration or IAM
changes are performed by the adapter. `accept_eula` records the operator's prior
license decision; it must not be enabled on their behalf.

`runner_root` contains the reviewed `workers/skypilot` and `workers/isaac_sim`
sources, the pinned SkyPilot `.venv`, and existing private `config.yaml`. The
optional server-only `sky_api_endpoint` selects an already configured endpoint.
Unrelated inherited API keys, proxies and training endpoint settings are omitted.
`gcloud_config` preserves a configured isolated Cloud SDK state directory.
Credentials and local paths are absent from public profile options.

`load_profiles(path)` reads the bounded server configuration. `profile.python`
points to `workers/skypilot/.venv/bin/python`; imports use that isolated Python.
`admit(profile, checkpoint_metadata)` checks the two supported families, one
named RGB camera with even dimensions within 1920, six state/action coordinates
and a model fingerprint. It records a hash of the visible runner scripts, runtime
configuration and scene assets, excluding credential contents, environments,
hidden temporary files and model weights. This is a compatibility admission,
not a model reload, scene calibration or cloud readiness claim.

The core job owner calls
`run(profile, checkpoint_directory, job_directory, event, timeout_seconds,
expected_profile_sha256=..., expected_model_id=...)`. The checkpoint has already
been copied and inventory-verified by the package importer; the runner verifies
its model fingerprint again. `job_directory` is new and private. The callback
accepts `(stage, message, data_or_none)`. The runner returns a report with exact
relative artifact paths, SHA-256, byte counts and GCS generations for the core to
register. It never invents a score or marks cloud infrastructure deleted.

For this app path, `--receipt-dir` requires experimental mode and a new directory.
Before submission, `launch-context.json` records the unique group, rollout label,
model/manifest hashes and a distinct result prefix beneath the operator's prefix.
`--expected-model-id` rejects a changed checkpoint before cloud submission.
Ordinary CLI receipt behavior remains available. Remote outputs are fetched only
from that job's prefix, with one remote UUID, an exact filename allowlist, fixed
byte limits and generation-bound reads. Completion requires matching model and
manifest identities, not just exit code zero. Raw cloud logs are not streamed
through the app callback.

A job is never automatically resubmitted. Timeout or cancellation stops/reaps
owned local command processes and requests cancellation of the saved job ID, or
its pre-submission unique group if an ID was not returned. Repeated cancellation
waits for that bounded cleanup. A disk error cannot suppress an already-owned
cancellation request. `recover(profile, job_directory)` provides the same bounded
cancellation after an app interruption; it does not adopt late output or retry.
The application must invoke recovery for its interrupted simulation jobs. A changed source/configuration profile or model identity is not used to cancel
an old numeric job ID; operator reconciliation is required. Missing
or inconsistent identity produces `cleanup_unknown` for operator reconciliation.
Even a successful cancellation response means **requested**, not proof of VM
deletion. The 30–7200-second application deadline plus bounded cleanup is not an
exact cloud-runtime or billing cap.

The tests use local processes and generated cloud responses. Actual native
checkpoint loading, GPU execution, cup-task outcome and cloud cleanup must be
established by a separately authorized run; they are not implied by these tests.
