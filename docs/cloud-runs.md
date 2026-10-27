# Monitor cloud runs

The **Cloud runs** page shows your project's cloud jobs and, optionally, snapshots of external SkyPilot rollouts. It only reads: it never connects to or controls a cloud host.

## Follow application cloud jobs

1. Open **Cloud runs**. If it isn't in the sidebar, open **Settings → Modules** and select **Add module** on **Cloud runs**.
2. Under **Jobs**, pick an **Application cloud job**. The list holds this project's cloud GPU jobs and Isaac simulation jobs.
3. Read its status, execution target, last update and latest 100 events. Select **Open training job** or **Open simulation job** for the full view.

A completed job doesn't prove task success or that its VM was deleted. Check cleanup on the job itself; see [cancel and clean up](skypilot-training.md#cancel-and-clean-up).

## Watch an external rollout

A separate observer process polls a running SkyPilot Job Group and writes one snapshot file per run. The page reads those files.

1. **Choose the snapshot folder.** Either:
   - add the **Cloud runs** module, which creates `cloud-runs/` inside the workspace with owner-only access, or
   - set `JENSEN_CLOUD_RUNS_DIR` to a dedicated absolute directory outside Git, then restart the server. This setting takes precedence.

   The folder path must not contain symlinks.
2. **Start the observer** for a rollout launched with the [rollout launcher](../workers/skypilot/ROLLOUT.md):

   ```sh
   bash workers/skypilot/watch-rollout.sh RUNNER_ROOT KEY_FILE JOB_ID GROUP_NAME OUTPUT_DIR [--once] [--experimental]
   ```

3. **Read it** under **External observations**. Pick the **Cloud run**, check the freshness line, and switch between **Isaac logs** and **VLA logs**.

| Argument | Value |
| --- | --- |
| `RUNNER_ROOT` | Root of the operator-owned runner checkout prepared with [runner setup](../workers/skypilot/RUNNER.md). |
| `KEY_FILE` | The runner's service-account key file. |
| `JOB_ID` | The running rollout's SkyPilot job ID, a positive integer. |
| `GROUP_NAME` | The rollout's Job Group name. It becomes the snapshot's `run_id` and filename. |
| `OUTPUT_DIR` | The snapshot folder from step 1. |
| `--once` | Collect once and exit. Without it, the observer collects every 10 seconds until you stop it. |
| `--experimental` | Record calibration as `unverified` instead of `unknown`. |

## Snapshot format, version 1

Any observer can feed the page if it follows this contract:

- Write one `<run_id>.json` per run. Write a private sibling temporary file first, then atomically replace the final file.
- Use `null` for unknown outcome values.
- If a collection fails, keep the last `collected_at` and observations, and set `collection_error`.
- Remove secrets before writing logs or snapshots.

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

| Field | Rule |
| --- | --- |
| `schema_version` | `1` |
| `run_id` | 1–64 ASCII characters: lowercase letters, digits, `_` and `-`, starting with a letter or digit. Must match the filename. |
| `label` | 1–120 characters. |
| `cluster` | The Job Group name, 1–100 characters. |
| `job_id` | The SkyPilot job ID as a string of digits, or `null`. |
| `collected_at` | Timezone-aware time of the last successful collection, or `null`. Times more than 5 seconds in the future are rejected. |
| `status` | The raw task status, 1–40 characters. |
| `collection_error` | `null`, or up to 2000 characters. |
| `logs.isaac`, `logs.vla` | Recent log tails, at most 48 KiB of UTF-8 each. |
| `outcomes.rollout_completed`, `outcomes.pickup_success` | `true`, `false` or `null`. |
| `outcomes.calibration` | `unknown`, `unverified` or `verified`. |

Limits:

- Each file is at most 768 KiB, a regular file (not a symlink), with no duplicate keys and no extra fields.
- The folder holds at most 50 snapshots and at most 200 entries in total.

## Freshness and API

The page reads `GET /api/v1/cloud-runs` every three seconds. The freshness line shows one of:

- **Current observation**: the latest collection succeeded within the last 90 seconds.
- **Last observation (refresh failed)**: the newest collection failed; its error appears below.
- **Stale or unavailable observation**: the last successful collection is more than 90 seconds old, or the page can't reach the server. Check the displayed collection time and errors, then restart the observer or fix its connection.
