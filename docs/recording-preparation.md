# Prepare a dataset from local teaching captures

The backend can prepare explicitly selected finalized teaching episodes as one saved `dataset.inspect` job. It uses the existing LeRobot 0.6.2 converter, verifies every converted row and video, and registers the existing immutable dataset snapshot. This does not start a simulator, voice provider, training run, or cloud resource. The Teaching page exposes the same project-owned preparation flow beside the live teaching controls. Changing the project does not remount those global controls.

Only finalized captures already present on the API host are available. An operator must copy or mount remote Isaac captures before admission; neither the Teaching relay nor this API transfers remote files. Keep published capture folders quiescent. There is no inferred “session closed” flag: complete source inventories are checked by the converter and rechecked before a result is registered.

## Operator configuration

Set `FIREBIRD_RECORDING_CONFIG` explicitly for the desired application process to a new configuration file. Creating the file alone does not activate it. Existing Hugging Face intake and `FIREBIRD_LOCAL_DATA_ROOT` retain their behavior.

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

The interpreter must be the isolated Python 3.12 / LeRobot 0.6.2 writer runtime already supported by `workers/teaching/requirements-dataset.txt` or the persistent `workers/local_cpu` reader setup. The worker verifies the exact upstream writer source hashes, PyArrow, H.264 decoder/encoder configuration and the configured FFmpeg/ffprobe executables. This is an offline operation: missing packages/assets fail; nothing is installed or downloaded. The sibling `workers/isaac_sim` contract module must be present. No Isaac runtime is imported or required by this adapter.

Configuration, captures, metadata and output paths use no-follow directory traversal. Project capture roots must be disjoint. A configured executable may have a final symlink, as virtualenv interpreters normally do; the original invocation path and resolved target identity are bound and rechecked. Data path links remain forbidden. These are operator-controlled paths, not an adversarial same-user filesystem isolation claim.

`GET /api/v1/projects/{project_id}/recordings/options` returns configuration presence, its identity, limits and setup guidance. `runtime_verified: false` deliberately distinguishes structural setup from a successful native job. No public runtime-ready capability is advertised by this slice.

`GET /api/v1/projects/{project_id}/recordings` returns bounded metadata candidates: session/episode identities, metadata/receipt hashes, declared lineage and origin, joint order, radians/controller/timebase, dimensions/rates and unknown or operator-reported-failure outcome. It returns no capture filesystem paths or images. Unfinished episodes are not candidates. Catalog limits are 100 session directories, 1,000 child entries, 16 MiB aggregate metadata and a 1 MiB response.

## Prepare in the web workspace

Open **Teaching** and choose the project that owns the operator-published captures. The preparation panel lists finalized metadata only; a configured catalog does not prove that the writer is ready. Select the exact episodes, review the unchanged-capture requirement, and explicitly choose **Prepare selected recordings**. Nothing is selected or submitted automatically. Synthetic captures are labelled as synthetic; that label alone does not identify them as software test fixtures.

A saved result shows its exact job, snapshot, episode/frame counts and declared lineage groups. **View this dataset** opens that inspection. **Train on this dataset** opens a new training draft with the exact saved dataset selected; neither action starts training. An explicitly requested dataset or inspection missing from current project history remains unavailable rather than silently selecting another one. A subsequent manual dataset choice takes precedence when later history reads arrive. Single-group snapshots remain inspectable, but their Training handoff is disabled because they cannot supply the required held-out split.

Draft selection, minimal accepted job identity and request uncertainty are stored per project in this browser tab's session storage. They survive reload in that tab; this is not cross-device recovery or server-side idempotency. A lost or invalid acknowledgment requires a fresh successful saved-jobs review and explicit recovery before another request. Storage failures disable preparation and cancellation until the user explicitly restores session recovery. A late response from an earlier mounted panel cannot overwrite a newer attempt, even when both requests selected the same episodes.

Cancellation refreshes the selected saved job and binds the request to that exact identity. Navigating to another record during that check prevents the stale cancellation. Project changes, navigation and result handoffs never submit a preparation, training or control command by themselves.

## Submit and inspect

Use the existing project intake endpoint. The client copies the exact catalog identities; it cannot supply a path, executable, unit conversion, lineage override or output directory.

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

Select 1–100 unique episodes across at most 100 unique sessions. The budget is 60–1,800 seconds after the job acquires the existing local native-work slot. Full input inventory is bounded to 50,000 entries / 8 GiB aggregate / 2 GiB per file. Output verification has the same bounds. All selected captures must have the same existing joint/camera/rate/action schema. This adapter does not normalize radians or infer compatibility with a policy's controller coordinates.

The normal job history, status, events and cancellation endpoints apply. Stale metadata or configuration is rejected before enqueue. Accepted jobs can still fail if full raw content is invalid, changes, or cannot be read by the pinned runtime. Only successful complete snapshot results are selectable as dataset inspections. The result adds `recording_preparation` with exact job/selection identity, source/group counts, readback/source-preservation evidence and `task_success_verified: false`. Legacy intake JSON omits this field and `recordings` when absent.

Multiple sources retain their actual source-session IDs, episode IDs, requested/applied action evidence, declared origin and root lineage groups. Episodes are reindexed only for the new dataset. Two captures with the same declared group remain one group; their snapshot can be saved, but they cannot satisfy the existing held-out training split requirement. Distinct generated fixture groups are software test inputs, not proof of physically independent demonstrations or successful robot behavior.

The supervisor owns the writer and snapshot process groups, waits for bounded cleanup on failure/cancellation, and checks exact output and source inventories before registration. A cancellation never adopts a late result. A hard API crash cannot promise child cleanup: startup marks unfinished jobs interrupted and preserves the existing operator warning; it does not retry, adopt files or kill a PID after restart. Private bounded diagnostic tails and requests remain under the failed job directory. Source captures are never overwritten or deleted.

## Verification scope

Focused core tests cover legacy serialization/intake behavior, project isolation, malformed catalog data, stale identities, symlinks, budgets, exact output/provenance/snapshot bytes and cancellation. Pure teaching adapter tests use generated captures and do not substitute fake data for a writer proof. Genuine pinned writer → immutable snapshot → saved API job evidence is recorded separately for each executed source version. No real recording, robot quality, voice or GPU acceptance is implied.
