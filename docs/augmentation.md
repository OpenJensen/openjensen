# Run appearance augmentation

## Setup

Install `gcloud`, `ffmpeg` and `ffprobe` on the API host. Connect Google Cloud in **Settings & diagnostics → Compute**, enable the Vertex AI API (`aiplatform.googleapis.com`), billing and Gemini Omni permissions. The server obtains access tokens with `gcloud auth print-access-token --quiet`.

Alternatively, disconnect the saved Cloud project, set `GEMINI_API_KEY` in the server environment and restart the application. The Cloud route uses `gemini-omni-1.1-flash-preview`; the key route uses `gemini-omni-1.1-flash`.

See Google's [video-editing setup](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/video/edit-videos), [model access](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/omni-1-1-flash), [authentication](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/models/interactions-api), and [Gemini Developer API](https://ai.google.dev/gemini-api/docs/omni).

## Generate and download

1. Inspect a public Hugging Face video dataset.
2. Open **Augmentation** or select **Augment this dataset** from its inspection.
3. Choose a camera, up to four episodes and a 1–10-second interval per clip.
4. Select **Change lighting**, **Change textures** or **Custom edit** and enter instructions.
5. Review the provider disclosure and clip summary, then start generation.
6. Follow the saved job, compare clips, and download the ZIP containing MP4s and `manifest.json`.

Use source camera files up to 48 MiB and review the clips before reusing their action labels. Select the saved job's cancellation action to stop local processing. Inspect an uncertain request before submitting another.

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

```sh
firebird augmentation options
firebird augmentation submit PROJECT_ID augmentation.json
firebird jobs show JOB_ID
firebird jobs cancel JOB_ID
```

Direct clients use `POST /api/v1/projects/{project_id}/augmentations`. Read `GET /api/v1/jobs/{job_id}` for status and `GET /api/v1/jobs/{job_id}/augmentation/download` for the ZIP. Use `POST /api/v1/jobs/{job_id}/cancel` to cancel.
