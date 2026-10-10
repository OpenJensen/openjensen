# Workspace guide

Open the project menu and select or create a project. Use the sidebar to open a page; bookmark its address to return directly. Open **Guide** at `/guide/` and the API reference at `/docs/`.

## Dashboard

Select **Check resources**, then open **My models**, **My datasets**, or a compute setup card.

## My models

Search the collection, open an exact model version, then choose **Distill**, **Quantize**, **Evaluate**, **Replay observations**, or **Run in simulation**. To export an ACT training checkpoint, choose **Open this checkpoint's training run and export**. Use **Import a model** to upload an ACT/SmolVLA policy archive with a configured import profile.

## Dataset

Open **Sources**, choose a starter or enter a Hugging Face repository, then select **Inspect dataset** and **Load visual preview**. For local data, choose **Local files**, upload a folder, ZIP or HDF5 file, and review the conversion settings. Use **Create example** to try the workflow with generated data. Save camera names and labels before exporting.

[Dataset setup](getting-started.md#inspect-a-public-dataset) · [Local imports](dataset-import.md) · [Local training](local-training.md)

## Augmentation

Choose a dataset, camera and episodes, set the clip interval, select an appearance edit, review billing, and start generation. Compare the returned clips before downloading them. [Setup](augmentation.md)

## Teaching

Start the configured executor, open Teaching, wait for a camera frame, set an instruction, and select **Start recording**. Use pause, joint corrections or failure markers as needed. Select **Finish episode**, then prepare the recording as a dataset. [Executor setup](teaching-connections.md) · [Recording](local-training.md#teach-in-simulation) · [Voice and advice](teaching-intelligence.md)

## Fine-tune

Choose data, cameras, model, method and compute. Review the recipe and start training. Open the saved run to follow progress, download a checkpoint, resume, or export it. [Worker setup](native-training.md) · [Recipes and commands](policy-workflow.md)

## Distill

Install the [CPU workers](../workers/local_cpu/README.md). Select an ACT teacher, compatible local observations, and separate training/validation/final episode groups. Review the ACT256 recipe and submit it. [Adapter setup](../workers/policy_distillation/README.md)

## Quantize

Choose the SmolVLA or ACT card and select a saved model. For ACT, export its training checkpoint first. Select Q8/Q4 or INT8/INT4, review the recipe, and start. Open the saved job to download the output. [SmolVLA setup](policy-workflow.md) · [ACT setup](native-quantization.md)

## Evaluate

Select the policy, runtime and check type, review the recipe, and start. For LIBERO, prepare the suite assets and task/state settings first. [Diagnostics setup](policy-workflow.md) · [LIBERO Spatial setup](spatial-workflow.md)

## Run

Choose **3D simulation**, **Replay observations**, or **Check inference**. Select the configured runtime and inputs, review consent, and start. Open the saved job for status, video or downloadable output. [Isaac](native-simulation.md) · [Replay](../workers/isaac_sim/NATIVE_REPLAY.md) · [Cloud inference](cloud-inference.md)

## Decision lab

Install the [Muose worker](../workers/decision/README.md), configure its model directory and license acceptance, then enter a state, instructions and 2–8 unique criteria. Keep each comparison within 512 tokens and select **Score criteria**.

## Cloud runs

Open an application job to follow its status and events. For an external rollout, configure the observer directory and start the observer command. [Monitor setup](cloud-runs.md)

## Settings & diagnostics

Use **Compute** to connect Google Cloud, save Hugging Face access, or add a local worker. For local training, select **Check this machine → Add worker**, enable local runs, and save. Use **Workflow settings** to edit recipes and **Diagnostics** to launch project checks. [Local setup](compute-settings.md#local-worker-setup) · [Cloud connection](cloud-connections.md) · [Diagnostics](policy-workflow.md)
