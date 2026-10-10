# Prepare a dataset from Teaching captures

## Operator configuration

Install the isolated Python 3.12/LeRobot 0.6.2 writer with `workers/teaching/requirements-dataset.txt` or the [CPU reader setup](../workers/local_cpu/README.md). Keep the sibling `workers/isaac_sim` source present and install FFmpeg/ffprobe.

Copy or mount finalized captures to the API host. Give each project a distinct capture root, save this configuration with real absolute paths, and set `FIREBIRD_RECORDING_CONFIG` before restarting:

```json
{
  "schema_version": 1,
  "dataset_python": "/absolute/pinned-reader/bin/python",
  "worker_root": "/absolute/checkout/workers/teaching",
  "ffmpeg": "/absolute/bin/ffmpeg",
  "ffprobe": "/absolute/bin/ffprobe",
  "projects": [
    {"project_id": "existing-project-id", "capture_root": "/absolute/published-captures"}
  ]
}
```

Keep selected capture files unchanged during preparation. Prepare episodes with matching joint order, radians/action schema, camera dimensions and rate. Use at least two source lineage groups for a training split.

## Prepare in the web workspace

1. Open **Teaching** and choose the project.
2. Select finalized sessions/episodes in the preparation panel.
3. Review the selection and choose **Prepare selected recordings**.
4. Follow the saved job through conversion and snapshot creation.
5. Select **View this dataset** or **Train on this dataset** from the completed result.

Choose 1–100 episodes across at most 100 sessions and a 60–1800-second job budget. Keep the selected input within 8 GiB total, 2 GiB per file and 50,000 entries.

## Submit and inspect

Read `GET /api/v1/projects/{project_id}/recordings/options` for the configuration hash and `GET /api/v1/projects/{project_id}/recordings` for session/episode IDs and hashes. Send those exact values to the project's intake endpoint:

```json
{
  "source": "local",
  "snapshot_for_training": true,
  "recordings": {
    "schema_version": 1,
    "configuration_sha256": "64-lowercase-hex-characters-from-options",
    "timeout_seconds": 600,
    "captures": [{
      "session_id": "32-lowercase-hex-characters",
      "session_sha256": "64-lowercase-hex-characters",
      "episodes": [{
        "episode_id": "32-lowercase-hex-characters",
        "receipt_sha256": "64-lowercase-hex-characters"
      }]
    }]
  }
}
```

Follow the returned job using the ordinary jobs/status/events endpoints. Cancel the selected job to stop preparation. On a lost response, reconcile the saved request key using [submission recovery](job-submissions.md). For stale-content errors, refresh the catalog and reselect unchanged captures.
