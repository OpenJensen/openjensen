# Record a managed simulation session

In a managed session, OpenJensen starts a local Isaac Sim recording process for you, lets you record episodes from the browser, and publishes the capture when you stop. You end up with recordings ready for [dataset preparation](recording-preparation.md). To drive an executor you start yourself, see [Connect the simulation executor](teaching-connections.md).

## What you need

- A Linux application host with a local Isaac Sim 6.1 installation. Managed sessions run only on Linux; there is no remote runner.
- The pinned [Teaching worker](../workers/teaching/README.md) and its strict Teaching settings file.
- A capture root for the project in `JENSEN_RECORDING_CONFIG`, set up as in [recording preparation](recording-preparation.md#1-configure-the-capture-roots). It must not overlap the workspace's `jobs/` folder.
- Accepted Isaac license terms.

Voice and advice are not available in managed sessions; [set them up](teaching-intelligence.md) with an existing executor instead.

## 1. Write a session profile

Save a private session profile file, set `JENSEN_TEACHING_SESSION_CONFIG` to its absolute path, and restart the server:

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

Replace the project, interpreter, worker, settings and lease paths with your own.

| Field | What to set |
| --- | --- |
| `id` | 1–64 letters, digits, `_` or `-`. Unique within the file. |
| `label` | Name shown in **AI teleoperation profile**, up to 100 characters. |
| `project_id` | The existing project allowed to use this profile. |
| `isaac_python` | Isaac's Python launcher. |
| `worker_root` | The `workers/teaching` directory. Keep the sibling `workers/isaac_sim` source present. |
| `settings_path` | The worker's strict Teaching settings file. Its `scene` path is resolved relative to this file. |
| `lease_path` | A private lock file that stops two sessions from using Isaac at once. Keep it apart from the settings file and outside the capture root. |
| `control_port` | Local port for the session, 1024–65535. |
| `accept_eula` | Must be `true`. Set it only after accepting Isaac's terms. |
| `max_seconds` | Longest allowed session, 1–3600 seconds. Default 300. |
| `max_capture_bytes` | Capture size cap, 1 KiB to 8 GiB. Default 512 MiB. Size it for your free disk space. |

The file holds 1–32 profiles and must stay under 64 KiB. Each profile has a hash that covers both configuration files, its settings and scene, the worker source and the Isaac interpreter. If any of them change, refresh the page before starting.

## 2. Start and record

1. Select a project and open **ROBOT LAB → AI teleoperation**. Choose **Simulation** under **Agent target**, then **Managed session** under **Connection**.
2. Choose an **AI teleoperation profile** and set **Session time limit (seconds)**: 1–3600, and no more than the profile's `max_seconds`.
3. Tick **Start this session and record simulator observations.** and select **Start simulation session**. Isaac readiness is checked only after you start.
4. Choose the session under **Saved simulation session**. Once it is ready, the recording controls appear. Record and finish each episode as described in [Record episodes](teaching-connections.md#2-record-episodes).
5. Select **Stop and publish**, then **Confirm stop and publish**.

> **Stop before the time limit.** Only a graceful stop publishes the recordings. If the time limit runs out, nothing is published. Finish every episode before you stop: the capture must hold 1–100 finished episodes.

Leaving the page never stops a saved session. Use **Refresh simulation sessions** to update the list.

To plan a goal in that simulator, select **Simulation** under **Agent target** in **Agent Control**, then enter the running job's ID. Follow [abstract-task planning](teaching-connections.md#plan-an-abstract-task) to review skills, observations and run limits. The managed session time limit also bounds the recording process.

## 3. Prepare the dataset

After the stop, the job validates the capture and publishes it to the project's recording catalog. When the session has succeeded, select **Review for dataset preparation** and continue with [recording preparation](recording-preparation.md).

## Cancel or recover

- **Abort without publishing:** cancel the job with `jensen jobs cancel JOB_ID` or `POST /api/v1/jobs/{job_id}/cancel`.
- **Failed or interrupted session:** read the saved job's error and cleanup message and the private logs before starting another. Raw evidence stays in the private job directory.
- **Lost start response:** select **Check saved simulation request**, and **Retry same simulation request** if it is offered. See [submission recovery](job-submissions.md).

## API

1. `GET /api/v1/projects/{project_id}/teaching/profiles` returns each profile's `id` and `profile_sha256`.
2. `POST /api/v1/projects/{project_id}/teaching/sessions` with `operation: teaching.capture`, `profile_id`, `profile_sha256` and `timeout_seconds` starts a session. It accepts an `Idempotency-Key` header.
3. Poll `GET /api/v1/projects/{project_id}/teaching/sessions/{job_id}` until `ready` is true.
4. Record through that session's routes under `/api/v1/projects/{project_id}/teaching/sessions/{job_id}`: `GET state`, `GET frame`, `POST commands` and `GET commands/{command_id}`. The **ROBOT LAB → AI teleoperation** page uses the same routes.
5. `POST /api/v1/projects/{project_id}/teaching/sessions/{job_id}/stop` finishes gracefully and publishes the capture.
