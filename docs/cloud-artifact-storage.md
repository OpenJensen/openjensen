# Cloud checkpoint storage

## Configure storage

Connect the Google Cloud project and prepare SkyPilot on the API host using [compute setup](compute-settings.md). Cloud jobs use a private bucket under `gs://firebird-artifacts-<project>/jobs/<job-id>/<stage>/`. Use uniform bucket access and public access prevention, and keep credentials in the configured Google Cloud/SkyPilot environment.

## Download or resume

1. Open the saved **Fine-tune** run.
2. Select the exact published checkpoint.
3. Choose **Download** to stream its TAR to the browser, or **Resume** to start from that checkpoint.
4. For SmolVLA packing, continue with its **Quantize** action. For ACT, follow [CPU export setup](cloud-act-export.md).

Checkpoint files stay in GCS. The application stores their manifests, recipes, metrics and logs in its workspace. Each artifact record contains its project ID, optimizer step and cloud location.

## Finish and clean up

Follow the saved job through checkpoint publication and GPU cleanup. If cleanup is uncertain, inspect the cluster command shown on that job. Manage retained checkpoint objects in the private bucket separately from GPU teardown.
