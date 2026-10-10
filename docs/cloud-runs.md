# Cloud run monitor

Open **Cloud runs** and select an application job to follow its saved status, target and events. For an external rollout, configure the observer feed below.

## Configure the observer

Set `FIREBIRD_CLOUD_RUNS_DIR` to a dedicated absolute directory outside Git and restart the application. Start the observer separately with its operator-owned runner and key paths:

```sh
bash workers/skypilot/watch-rollout.sh RUNNER_ROOT KEY_FILE JOB_ID GROUP_NAME OUTPUT_DIR [--once]
```

Replace `RUNNER_ROOT`, `KEY_FILE`, `JOB_ID`, `GROUP_NAME` and `OUTPUT_DIR` with the running rollout's values. Use `--once` for one collection. Follow the [rollout observer setup](../workers/skypilot/ROLLOUT.md).

## Snapshot contract, version 1

Write one `<run_id>.json` per run. Write a private sibling temporary file first, then atomically replace the final pathname. Use this schema:

```json
{
  "schema_version": 1,
  "run_id": "example-run-1234",
  "label": "Example operator run",
  "cluster": "example-job-group",
  "job_id": "7",
  "collected_at": "2026-09-27T00:00:00Z",
  "status": "RUNNING",
  "collection_error": null,
  "logs": { "isaac": "Recent Isaac tail\n", "vla": "Recent VLA tail\n" },
  "outcomes": {
    "rollout_completed": null,
    "pickup_success": null,
    "calibration": "unverified"
  }
}
```

Supply a 1–64-character lowercase ASCII `run_id` matching the filename, a timezone-aware collection timestamp, the raw task status and recent log tails. Keep log strings within 48 KiB each and the encoded file within 768 KiB. Store at most 50 snapshots in a directory with at most 200 entries.

Use `null` for unknown outcome values. On collection failure, retain the last collection timestamp/observations and set `collection_error`. Remove secrets before writing logs or snapshots.

## API and freshness

Read `GET /api/v1/cloud-runs` for the feed. Open the page to poll every three seconds; inspect the displayed collection time and errors when data becomes stale. Refresh the observer or correct its connection if a collection is over 90 seconds old.
