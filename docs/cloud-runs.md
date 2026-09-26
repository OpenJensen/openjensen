# Read-only cloud run monitor

The **Cloud runs** workspace view reads snapshots published by a separate operator
monitor. This read-only snapshot view does not load cloud credentials, connect to SkyPilot,
launch jobs, refresh cloud resources, cancel jobs or accept a browser-supplied
filesystem path. Existing local host and origin restrictions apply to its API.

Firebird-managed training and quantization use the separate compute connection
and SkyPilot dispatch path. Follow those jobs in **Fine-tune** and **Quantize**;
this view preserves the standalone operator monitor for runs such as Isaac
rollouts. See [cloud training](skypilot-training.md) for the managed lifecycle.

Monitoring is disabled unless the operator sets `FIREBIRD_CLOUD_RUNS_DIR` to a
dedicated external directory before starting the application. Keep this directory
outside Git. The application only reads it; the operator owns retention and access.
Use a real absolute directory, without symlinks in its path.

The standalone observer is invoked separately:

```sh
bash workers/skypilot/watch-rollout.sh RUNNER_ROOT KEY_FILE JOB_ID GROUP_NAME OUTPUT_DIR [--once]
```

`RUNNER_ROOT` and `KEY_FILE` stay with that operator process. They must never appear
in application configuration, snapshots or logs. The observer must remove secrets
from log tails before publication; the app treats operator snapshots as trusted
content, not as a credential-redaction boundary. This command observes an existing
job only. See its worker documentation for access and collection requirements.

## Snapshot contract, version 1

Publish one UTF-8 JSON file named `<run_id>.json` per run by writing a private
sibling temporary file and atomically replacing the final pathname. Temporary
files should not end in `.json`. A complete example:

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

- `run_id` is 1–64 lowercase ASCII letters, digits, underscores or hyphens; its
  first character is a letter or digit. It must match the filename and identify
  the same immutable job group throughout its lifetime.
- `label` is 1–120 characters; `cluster` is the cluster or job-group name, 1–100
  characters. `job_id` is a numeric string up to 64 digits or `null`.
- `status` is the raw primary-task state reported by SkyPilot, 1–40 characters.
  The app does not guess success semantics or color unknown states as successful.
- `collected_at` is a timezone-aware ISO timestamp of the **last successful remote
  collection**, or `null` if no successful collection exists. On failure preserve
  that timestamp and prior observations and set `collection_error` (at most 2,000
  characters). Future observations more than five seconds ahead are rejected.
- Each log string is limited to **48 KiB of decoded UTF-8**. The producer selects
  recent tails. The app displays literal text, without interpreting HTML or commands.
- Outcome booleans are explicit evidence only. Use `null` until known; neither
  `SUCCEEDED` nor a log message alone establishes pickup or rollout success.
  Calibration is `unknown`, `unverified` or `verified`; these are monitor reports,
  not independent application validation. Nothing here establishes optimizer
  acceptance or a successful trained-policy deployment.

The application rejects unknown fields, unsupported versions, duplicate JSON
keys, invalid types, symlinks and nonregular files. Each encoded snapshot is capped
at **768 KiB**, allowing JSON escaping of the bounded logs. It accepts at most
**50 snapshots** and scans at most **200 directory entries**, including temporary
files. Exceeding either directory limit fails the feed visibly, rather than
silently hiding some runs. Invalid individual snapshots are reported alongside
other valid runs. Reader errors never echo raw validation input or credential paths.

## API and freshness

`GET /api/v1/cloud-runs` returns `enabled`, `server_time`,
`stale_after_seconds`, `runs` and `errors`, with `Cache-Control: no-store`.
Each run adds `age_seconds` and `stale` to the snapshot. No write or execution
endpoint exists. Browser query parameters cannot change the configured directory.

The view polls every three seconds while open. Collection data becomes stale after
90 seconds; an absent collection time is always stale. The browser also ages the
last received observation locally and marks retained data unavailable when API
updates fail. The collection time, remote collection errors and snapshot read
errors remain visible independently of the primary task state.

These are log and status observations. Real policy quality, pickup success,
calibration and hardware acceptance still need their own explicit evidence.
