# Workspace guide

Open the project menu and select or create a project. Use the sidebar to open a page; bookmark its address to return directly. Robot Lab, Teaching, Augmentation, Decision Lab and Cloud runs appear once they are set up on the app host; **Settings → Modules** lists what each needs. Open **Guide** at `/guide/` and the API reference at `/docs/`.

## Dashboard

Select **Check resources**, then open **Models**, **Datasets**, or a compute setup card.

## Robot Lab

Connect the computer beside your SO-101, preview its camera, and work through **Connect → Motor IDs → Calibrate → Teleoperate → Record**. USB access stays on that computer; captures stay there until you add them to a project.

Select a joint in the SO-101 view to identify it; the view never moves the real arm. Assign all six motor IDs, then calibrate by capturing the middle pose and moving each joint through its range. Enable movement explicitly and keep **Stop** within reach; a lost control heartbeat stops the session. To record, name the dataset and task, choose the camera and frame rate, then save or discard each episode. Robot Lab is available only in the private workspace. [SO-101 setup](so101-robot-lab.md)

## Teaching

Start the configured executor, open Teaching, wait for a camera frame, set an instruction, and select **Start recording**. Use pause, joint corrections or failure markers as needed. Select **Finish episode**, then prepare the recording as a dataset. [Executor setup](teaching-connections.md) · [Recording](local-training.md#teach-in-simulation) · [Voice and advice](teaching-intelligence.md)

## Datasets

Open **Sources**, select **Hugging Face**, and enter `codywang/so101_pickup_test` in **Dataset repository**. For the pinned example, expand **Revision (optional)** and enter `ecef85bc07005f771ad86deeff1427f9d72953ed`; leave it blank to use the latest `main` revision. Select **Inspect dataset**, then choose an episode, use **Play**, and open **Frame samples** to inspect state or action values.

For local data, choose **Local files**, upload your folder, ZIP or HDF5 file, and review the conversion settings. Save camera names and labels before exporting.

[Dataset setup](getting-started.md#inspect-a-public-dataset) · [Local imports](dataset-import.md) · [Local training](local-training.md)

## Augmentation

Choose a dataset, camera and episodes, set the clip interval, select an appearance edit, review billing, and start generation. Compare the returned clips before downloading them. [Setup](augmentation.md)

## Fine-tune

Choose a recent dataset or use **Browse all datasets** to select up to eight compatible datasets. Then choose cameras, model, method and compute, review the recipe, and start training. Open the saved run to follow progress, download a checkpoint, resume, or export it. [Worker setup](native-training.md) · [Recipes and commands](policy-workflow.md)

## Distill

Install the [CPU workers](../workers/local_cpu/README.md). Select an ACT teacher, compatible local observations, and separate training/validation/final episode groups. Review the ACT256 recipe and submit it. [Adapter setup](../workers/policy_distillation/README.md)

## Quantize

Choose the SmolVLA or ACT card and select a saved model. For ACT, export its training checkpoint first. Select Q8/Q4 or INT8/INT4, review the recipe, and start. Open the saved job to download the output. [SmolVLA setup](policy-workflow.md) · [ACT setup](native-quantization.md)

## Evaluate

Select the policy, runtime and check type, review the recipe, and start. For LIBERO, prepare the suite assets and task/state settings first. [Diagnostics setup](policy-workflow.md) · [LIBERO Spatial setup](spatial-workflow.md)

## Run

Choose **3D simulation**, **Replay observations**, or **Check inference**. Select the configured runtime and inputs, review consent, and start. Open the saved job for status, video or downloadable output. [Isaac](native-simulation.md) · [Replay](../workers/isaac_sim/NATIVE_REPLAY.md) · [Cloud inference](cloud-inference.md)

## Decision Lab

Install the [Muose worker](../workers/decision/README.md), configure its model directory and license acceptance, then enter a state, instructions and 2–8 unique criteria. Keep each comparison within 512 tokens and select **Score criteria**.

## Models

Search the collection, open a model version, then choose **Distill**, **Quantize**, **Evaluate**, **Replay observations**, or **Run in simulation**. To export an ACT training checkpoint, choose **Open this checkpoint's training run and export**. Use **Import a model** to upload an ACT/SmolVLA policy archive with a configured import profile.

## Cloud runs

Open an application job to follow its status and events. For an external rollout, configure the observer directory and start the observer command. [Monitor setup](cloud-runs.md)

## Settings

Use **Modules** to add pages to the sidebar and install optional runtime packages on the named application server. Installation progress continues after leaving the page; companion pairing, provider credentials and external simulator setup remain separate.

Use **Compute** to connect Google Cloud, save Hugging Face access, or add a local worker. For local training, select **Check this machine → Add worker**, enable local runs, and save. **Modules** shows which optional pages are set up. Use **Workflow settings** to edit recipes and **Diagnostics** to launch project checks. [Local setup](compute-settings.md#local-worker-setup) · [Cloud connection](cloud-connections.md) · [Diagnostics](policy-workflow.md) · [Modules](installation-profiles.md#workspace-modules)
