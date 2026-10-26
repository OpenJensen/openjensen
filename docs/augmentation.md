# Change dataset appearance with Gemini

Augmentation sends short clips from one camera of a Hugging Face dataset to Google's Gemini Omni, which changes their lighting or textures. You get each original and edited clip back to compare and download. The source dataset is not changed, and the outputs are not added as training data.

> **Billing:** clips and prompts are sent to Google. Charges apply to your connected Google Cloud project, and may apply on the Gemini API key route.

## What you need

- `ffmpeg` and `ffprobe` on the application host.
- A public Hugging Face dataset with a video camera, inspected in **Datasets**.
- Access to Gemini Omni through one of the routes below.

## Connect Gemini

Set up **one** route:

| Route | Setup | Model |
| --- | --- | --- |
| Google Cloud | Install `gcloud` on the application host and sign in. Connect Google Cloud in **Settings → Compute** ([steps](cloud-connections.md#provider-setup)). Enable the Vertex AI API (`aiplatform.googleapis.com`), billing and Gemini Omni permissions. | `gemini-omni-1.1-flash-preview` |
| Gemini API key | Disconnect any saved Cloud project, set `GEMINI_API_KEY` in the server environment, and restart the application. | `gemini-omni-1.1-flash` |

- On the Cloud route, the server gets short-lived access tokens with `gcloud auth print-access-token --quiet`. No Cloud Storage bucket is needed.
- On the key route, Google's Developer API does not offer uploaded-video editing in the EEA, Switzerland or the UK.

The **Augmentation** page appears once either route is set up, or after you add it in **Settings → Modules**. Google's references: [video-editing setup](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/video/edit-videos), [model access](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/omni-1-1-flash), [authentication](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/models/interactions-api) and the [Gemini Developer API](https://ai.google.dev/gemini-api/docs/omni).

## Generate clips

1. Inspect a public Hugging Face dataset that has a video camera.
2. Open **Augmentation**, or select **Augment this dataset** from its inspection.
3. Choose the dataset and a camera. Enter up to four **Episode numbers**, a **Start at (seconds)** time and a **Clip length (seconds)** of 1–10 seconds.
4. Under **Appearance**, choose **Change lighting**, **Change textures** or **Custom edit**. Add instructions; **Custom edit** requires them. Instructions are limited to 2000 characters.
5. Review the provider disclosure and clip summary, then select **Generate augmented clips**.
6. Follow the saved job. Compare each **Original** and **Augmented** clip, then download one clip or select **Download review bundle** for a ZIP with the MP4s and `manifest.json`.

Review the clips before reusing their action labels: a generated appearance can change geometry, motion or contact.

## Limits

- Each source camera file can be up to 48 MiB, and each trimmed clip must stay under 14 MiB.
- Every clip is trimmed and checked before the first paid request is sent.
- One augmentation runs at a time on the server, and each job has a 30-minute limit.

## Cancel or recover

- Select **Cancel augmentation** on the saved job to stop local processing and any further requests. A Gemini request that was already sent may still run and incur charges.
- If a submission's outcome is uncertain, inspect it before submitting another. See [submission recovery](job-submissions.md).

## CLI and API

Save this recipe as `augmentation.json`, replacing the inspection and camera IDs:

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

`preset` is `lighting`, `texture` or `custom`; `custom` requires a `prompt`.

```sh
jensen augmentation options
jensen augmentation submit PROJECT_ID augmentation.json
jensen jobs show JOB_ID
jensen jobs cancel JOB_ID
```

| Request | Use |
| --- | --- |
| `GET /api/v1/augmentation-options` | Setup status, model and presets. |
| `POST /api/v1/projects/{project_id}/augmentations` | Submit the recipe. |
| `GET /api/v1/jobs/{job_id}` | Job status. |
| `GET /api/v1/jobs/{job_id}/augmentation/download` | The ZIP. |
| `GET /api/v1/jobs/{job_id}/augmentation/clips/{index}` | One augmented clip; add `?original=true` for the original. |
| `POST /api/v1/jobs/{job_id}/cancel` | Cancel. |
