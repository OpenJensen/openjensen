# Gemini Omni appearance augmentation

Under **Dataset → Augmentation**, choose a successful public Hugging Face
inspection, a video camera and up to four episode indices. Choose **Change
lighting**, **Change textures**, or a custom appearance prompt. Start time is
relative to each episode; duration is 1–10 seconds. The whole selection is
checked and trimmed before the first generation request.

This first augmentation method uses Google's Gemini Omni video editing API.
Google Cloud login uses `gemini-omni-1.1-flash-preview`; the Gemini API key route
uses `gemini-omni-1.1-flash`. Additional augmentation models are planned.

## Setup

Connect **Google Cloud in Settings** using the Google Cloud CLI login and your
project. Install `gcloud`, `ffmpeg` and `ffprobe` on the **Python application
server** PATH. A saved Google Cloud project takes priority over a Gemini API key.
The augmentation page shows the selected project before generating. The server
obtains a short-lived bearer token with `gcloud auth print-access-token --quiet`
before each clip request. The CLI manages sign-in and token refresh. There is
no second ADC login and no Gemini API key requirement for this route.

The Cloud route uses the global Interactions endpoint, independently of the
compute region in Settings. Enable the Vertex AI API (`aiplatform.googleapis.com`),
billing and suitable model permissions on that project. Cloud login and project
verification alone do not prove Omni access or quota. See Google's
[Cloud video-editing guide](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/video/edit-videos),
[Omni 1.1 model availability](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/omni-1-1-flash),
and [Interactions API authentication](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/models/interactions-api).

Alternatively, without a saved Cloud project, set `GEMINI_API_KEY` in the server
environment and restart it. This uses the
[Gemini Developer API](https://ai.google.dev/gemini-api/docs/omni).
Disconnecting Google Cloud restores this fallback when a key is present.
Failures on the selected route never silently retry through another billing
account. Credentials are never requested by the frontend or saved in recipes,
manifests, job records or browser storage. No new Python dependencies or GPU
worker are required.

`GET /api/v1/augmentation-options` reports local configuration and presets.
Options include the selected auth mode, Cloud project and exact model.
Configured means that a saved Cloud project and CLI (or a key), plus media
binaries, are present; it does not claim successful model access or a still-valid
login. An actual run sends the selected video clips
and prompt to Google and may incur API charges. Provider access, quota, and
regional restrictions still apply. The Gemini Developer API currently excludes
uploaded-video editing in the EEA, Switzerland and UK. Generated media includes SynthID.

## Outputs and scope

Jobs appear with queued/running/terminal status, preparation/generation progress,
and cancellation. Successful runs show the original and edited clips, plus a ZIP
download containing both MP4s and `manifest.json`. The manifest records the
exact resolved prompt, request, source inspection, immutable Hub revision,
metadata hash, episode and camera identity, **absolute camera-file timestamps**,
clip hashes, authentication route, Cloud project when applicable, exact model,
and interaction ID when returned.

The input is re-encoded to H.264 without audio, fitting within 1280×720 while
preserving its aspect ratio. Gemini is requested to edit appearance while
preserving robot/object geometry, motion, contact, camera viewpoint and timing.
The application checks that the generated MP4 has a video stream and that its
duration matches the requested interval within 150 ms. This checks gross timing
errors; it **does not establish frame alignment or valid robot actions**.

Review the clips before reusing source action labels. They are exported as
augmentation candidates and are not automatically registered as a trainable
LeRobot dataset. Original camera files, action/state records and metadata are
never overwritten. Multi-camera consistency, full-dataset augmentation, embedded
image columns, local datasets and automatic merging into training are outside
this first implementation.

Limits are four clips/job, one active augmentation job, 10 seconds/clip,
48 MiB per source camera file, 14 MiB per prepared upload and 40 MiB per output.
Oversize source files fail explicitly rather than loading an entire large
LeRobot v3 video shard. Jobs have a 30-minute deadline and individual provider
requests have bounded timeouts. There are no automatic paid-request retries.
Cancel/restart stops local work and prevents publication; a request Google
already accepted may still complete and incur charges. Restart marks unfinished
jobs interrupted and requires an explicit new job.

Both routes send inline input and request `store=false`. Cloud requests inline
video output and requires no Cloud Storage bucket. The Developer API route
downloads returned generated Files immediately. It accepts only validated Google
Files references, never arbitrary model-returned URLs, and never forwards the API
key to storage redirects. Cloud bearer tokens are sent only to the fixed
`aiplatform.googleapis.com` endpoint. Provider and CLI errors are sanitized before
persistence.

## CLI and API

Save a recipe as `augmentation.json`:

```json
{
  "operation": "dataset.augment",
  "source_job_id": "INSPECTION_JOB_ID",
  "episode_indices": [0, 1],
  "camera_key": "observation.images.front",
  "preset": "lighting",
  "prompt": "Keep the shadows mild and the work surface clearly visible.",
  "start_seconds": 0,
  "duration_seconds": 5
}
```

```sh
firebird augmentation options
firebird augmentation submit PROJECT_ID augmentation.json
firebird jobs show JOB_ID
firebird jobs cancel JOB_ID
```

The CLI uses the same persistent job service as the UI:

- `POST /api/v1/projects/{project_id}/augmentations`
- `GET /api/v1/jobs/{job_id}`
- `POST /api/v1/jobs/{job_id}/cancel`
- `GET /api/v1/jobs/{job_id}/augmentation/clips/{index}?original=true`
- `GET /api/v1/jobs/{job_id}/augmentation/clips/{index}`
- `GET /api/v1/jobs/{job_id}/augmentation/download`

Tests use mocked Google responses and deterministic media fixtures. Live paid
model access and real augmentation quality require a configured account and
are not established by those tests.
