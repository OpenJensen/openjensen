# Run a managed Teaching session

## Explicit operator configuration

Prepare a local Linux Isaac installation and the pinned [Teaching worker](../workers/teaching/README.md). Configure the project's capture root with `FIREBIRD_RECORDING_CONFIG` using [recording setup](recording-preparation.md).

Save a private session profile file and set `FIREBIRD_TEACHING_SESSION_CONFIG` to its absolute path before restarting the API:

```json
{
  "schema_version": 1,
  "profiles": [{
    "id": "local-isaac",
    "label": "Local Isaac teaching",
    "project_id": "existing-project-id",
    "isaac_python": "/operator/isaac/python.sh",
    "worker_root": "/operator/openjensen/workers/teaching",
    "settings_path": "/operator/teaching/settings.json",
    "lease_path": "/operator/teaching/isaac-resource.lock",
    "control_port": 8878,
    "accept_eula": true,
    "max_seconds": 300,
    "max_capture_bytes": 536870912
  }]
}
```

Replace the project, interpreter, worker, settings and lease paths. Supply the worker's strict Teaching settings file and size `max_capture_bytes` for available disk space. Set `accept_eula` after accepting Isaac's terms.

## Start and record

Open **Teaching**, choose the configured **Teaching profile**, set the duration,
review the session consent, and select **Start teaching session**. Select the saved
session to use its recording controls. After finishing the episodes, choose
**Stop and publish**.

For direct API requests:

1. Read `GET /api/v1/projects/{project_id}/teaching/profiles` and select the profile ID/hash.
2. Send `POST /api/v1/projects/{project_id}/teaching/sessions` with `operation: teaching.capture`, `profile_id`, `profile_sha256` and `timeout_seconds`.
3. Read `GET /api/v1/projects/{project_id}/teaching/sessions/{job_id}` until the session is ready.
4. Open Teaching to record and finish episodes through that session's state/frame/command routes.
5. Send `POST /api/v1/projects/{project_id}/teaching/sessions/{job_id}/stop` for graceful finish and capture publication.

Choose a duration of 1–3600 seconds within the profile's `max_seconds`. Finalize every recorded episode before graceful stop.

## Capture publication

After the worker exits, follow its job through capture validation. Published captures appear in the project's recording catalog. Select them in [recording preparation](recording-preparation.md) to create a dataset.

Use job cancellation to abort the session. For interrupted or failed sessions, inspect the saved job's cleanup message and private logs before starting another. Prepare LiveKit and provider credentials separately for [voice](teaching-intelligence.md).
