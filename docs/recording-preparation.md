# Prepare a dataset from Teaching captures

Turn finished Teaching captures into a LeRobot dataset with a verified training copy, ready for **Fine-tune**. Preparation runs on the application host and needs no simulator connection.

## What you need

- A Linux or macOS application host.
- The isolated Python 3.12 writer with pinned LeRobot 0.6.2. Install `workers/teaching/requirements-dataset.txt`, or use the `dataset-reader` environment from the [local CPU runtime installer](../workers/local_cpu/README.md).
- The `workers/teaching` source, with the sibling `workers/isaac_sim` source present.
- FFmpeg and ffprobe.
- Finished captures on the application host. Copy or mount remote Isaac captures there first.

## 1. Configure the capture roots

Save this configuration with real absolute paths, set `JENSEN_RECORDING_CONFIG` to its path, and restart the server:

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

- Give each project its own capture root. Roots must exist and must not overlap.
- The file lists up to 100 projects and must stay under 64 KiB.
- `dataset_python` is the writer's interpreter; `worker_root` is the `workers/teaching` directory.

Setting `JENSEN_RECORDING_CONFIG` also adds **Teaching** to the sidebar.

## 2. Prepare in the browser

1. Open **Teaching** and choose the project. The **Prepare a dataset from recordings** panel lists the published captures.
2. Under **Choose recorded episodes**, tick the episodes you want. After a [managed session](managed-teaching.md), **Select these recorded episodes** fills in that capture for you.
3. Set **Preparation timeout (seconds)**: 60–1800, default 600.
4. Tick the review box and select **Prepare selected recordings**. Keep the selected capture files unchanged until preparation finishes.
5. Follow the job under **Saved recording preparation** through conversion and snapshot creation.
6. When it succeeds, select **View this dataset** or **Train on this dataset**.

## Selection rules

- Episodes must match: joint order, radians state/action schema, camera dimensions and frame rate.
- Choose 1–100 episodes across at most 100 sessions.
- Keep the selected input within 8 GiB total, 2 GiB per file and 50,000 entries.
- Training needs at least two distinct source lineage groups for a held-out split. With only one, the dataset is still prepared, but **Train on this dataset** stays disabled.

## Cancel and recover

- **Cancel:** select **Cancel selected preparation**, then **Confirm preparation cancellation**. The original recordings are kept.
- **Lost response:** reconcile the saved request key with **Check saved request**. See [submission recovery](job-submissions.md); the operation is `dataset.inspect`.
- **Stale content:** select **Refresh**, then **Discard stale recording selection** if offered, and reselect unchanged captures.

## API

1. Read `GET /api/v1/projects/{project_id}/recordings/options` for the configuration hash.
2. Read `GET /api/v1/projects/{project_id}/recordings` for the session and episode IDs and hashes.
3. Send those exact values to `POST /api/v1/projects/{project_id}/intakes`:

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

Follow the returned job with `GET /api/v1/jobs/{job_id}` and `GET /api/v1/jobs/{job_id}/events`. Cancel it with `POST /api/v1/jobs/{job_id}/cancel`.
