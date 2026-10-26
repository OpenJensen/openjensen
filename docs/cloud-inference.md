# Test a policy on a cloud GPU

Load a quantized SmolVLA policy on a Google Cloud GPU and measure native inference, from **Evaluate** or **Run → Check inference**. These checks use synthetic inputs: they measure loading, action validity, speed and memory, not robot task success.

## What you need

- Google Cloud connected and SkyPilot set up. See [Set up the API host](skypilot-training.md#set-up-the-api-host) and [compute settings](compute-settings.md).
- A saved SmolVLA GGUF artifact or package in the project. To make one, follow [SmolVLA quantization](policy-workflow.md).
- A T4, L4 or A100 (40 GB) GPU. The A100-80GB target has no inference build profile; a job on it fails before a VM starts.

## Run a check

> **Cloud jobs cost money.** Google Cloud bills your project from the moment the VM is provisioned until it is deleted.

1. Open **Evaluate**, or open **Run** and choose **Check inference**.
2. Under **My model**, select the exact GGUF or package.
3. Under **Compute**, select a cloud GPU (marked **Cloud · GPU**).
4. Optional: open **Engine check settings** and set **Warmup predictions** (1–100, default 3) and **Timed predictions** (2–1000, default 10).
5. Select **Start evaluation** in Evaluate, or **Reload and run** in Run.
6. Open the saved job. **Results** shows timing and memory, **Output** has a **Download** link for each output, and **Details and logs** holds events and reports.

| Page | Selected policy | What runs on the VM |
| --- | --- | --- |
| **Evaluate** | GGUF or package | Measures native CUDA inference of that policy. |
| **Run → Check inference** | GGUF | Packages the GGUF and runs the package in a fresh offline process. |
| **Run → Check inference** | Package | Runs that exact saved package in a fresh offline process. |

The VM builds the pinned native binaries with CUDA 12.8, which needs an Ubuntu 22.04 or 24.04 VM image. Outputs are stored in private Cloud Storage. The application keeps bounded reports (up to 256 KiB each), logs and artifact descriptors. **Download** streams the package from Cloud Storage.

## Use the API or CLI

Submit `policy.evaluate` or `policy.run` with the artifact ID and a cloud runtime ID: `skypilot-gcp-T4`, `skypilot-gcp-L4` or `skypilot-gcp-A100`. Set `evaluation.mode=engine` with the default suite (`libero_object`), plus `warmups` and `repetitions`. Cloud checks accept no training recipe and no task-success limits. The recipe format is in [Use the CLI](policy-workflow.md#use-the-cli).

## Cancel or troubleshoot

To stop a check, open the job and select **Cancel run**, then follow its cleanup status. If the job reports a cluster cleanup command, run it on the API host; see [cancel and clean up](skypilot-training.md#cancel-and-clean-up).

| Problem | What to do |
| --- | --- |
| This policy is stored on Google Cloud. Choose a cloud target. | Select a cloud GPU under **Compute**. |
| Cloud engine checks currently support SmolVLA GGUF policies. | Select a SmolVLA GGUF or package. |
| Artifact admission fails | Select a complete, registered SmolVLA GGUF or package and check its manifest. To use a training checkpoint, quantize it first. |
| This GPU has no prepared native CUDA inference build profile. | Choose a T4, L4 or A100 (40 GB). |
| Setup or startup fails | Read the error and events under **Details and logs**. The full worker log is `worker.log` in the job's folder under `<workspace>/jobs/<job-id>/`. Then check the [SkyPilot setup](skypilot-training.md). |
