# Google Cloud training through SkyPilot

## Configure the host

Install SkyPilot 0.13 with GCP support and Google Cloud CLI on the API host. Run `gcloud auth application-default login` and connect the project/region in **Settings & diagnostics → Compute**. Save Hugging Face access there when the selected model needs it, and accept that model's access terms on the Hub.

Use the separate pinned environments for SmolVLA (Python 3.11), native LeRobot models (Python 3.12), and Psi-Zero (Python 3.11). Follow [worker setup](../workers/smolvla_qlora/docs/training/smolvla-qlora.md) and [model-specific inputs](native-training.md).

## Start training

1. Inspect and select a dataset.
2. Open **Fine-tune** and choose cameras, model and method.
3. Select the cloud GPU, review the recipe and paid-job consent, and start.
4. Open the saved run to follow preparation, optimizer progress, loss, events and checkpoints.

The job captures its project, region, GPU, machine type, disk size, idle interval, SkyPilot server and workspace. Model/dataset files download on that worker. Checkpoints and final artifacts are written to private GCS; manifests, recipes, logs and metrics are returned to the application.

## Download, resume or quantize

Select a published checkpoint in the run, then choose **Download**, **Resume**, or **Quantize**. The default recipe publishes five checkpoints including its final step. Resume selects a committed checkpoint and its saved recipe. Follow [checkpoint storage](cloud-artifact-storage.md) for retained objects and [ACT export](cloud-act-export.md) for the CPU download/export route.

For SmolVLA, choose Q8 or Q4 in **Quantize**, review the recipe, and start. Open the quantization job to download its GGUF.

## Cancel and recover

Cancel the selected job and watch its cleanup status. On an interrupted run, inspect saved checkpoints before selecting Resume. Review the exact cluster command when the job reports unconfirmed cleanup. Correct reported credentials, quota or capacity errors before starting another request.

## Evaluate and Run on cloud GPUs

Select the saved GGUF/package under **Evaluate** or **Run → Check inference**, choose the cloud runtime, review the recipe and start. Follow [cloud inference steps](cloud-inference.md).

For external rollout observation, configure the [Cloud runs monitor](cloud-runs.md).
