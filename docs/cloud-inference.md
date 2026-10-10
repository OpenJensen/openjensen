# Evaluate and Run on Google Cloud

## Prepare

1. [Configure Google Cloud and SkyPilot](compute-settings.md).
2. Select a saved SmolVLA GGUF artifact or package in the project. To create one, follow [SmolVLA quantization](policy-workflow.md).
3. Choose a cloud GPU target. The worker builds the pinned native binaries with CUDA 12.8 on its Ubuntu VM.

## Start and follow

1. Open **Evaluate** or **Run → Check inference**.
2. Select the exact artifact and runtime.
3. For Evaluate, choose `evaluation.mode=engine`, warmups and repetitions.
4. Review the recipe and paid-job consent, then start.
5. Open the saved job for status, events, timing/memory reports and downloadable output.

Run packages a GGUF or executes the selected saved package. Outputs are stored in private GCS; the application retains bounded reports, logs and artifact descriptors. Select **Download** from the saved result to retrieve the package.

## Cancel or troubleshoot

Cancel the exact selected job and follow its cleanup status. Inspect any reported cluster cleanup command. If artifact admission fails, select a complete registered SmolVLA GGUF/package and check its manifest. If startup fails, review the saved worker logs and [SkyPilot setup](skypilot-training.md).
