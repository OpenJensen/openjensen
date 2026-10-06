# Workspace guide

Use the project menu to select an existing project or choose **Create project…** to add one. Then choose a destination under **Models**, **Data**, **Train**, **Test** or **Workspace**. **Guide** opens `/guide/`; `/docs/` remains the separate API reference. Selecting a card does not start a job. Configuration, completed jobs and downloaded artifacts do not by themselves establish robot task quality.

## My models

Open **My models** to find saved checkpoints, imported policies, distilled students and quantized versions across projects. Search by name, dataset or model ID, or show only the current project. Execution records and simulation videos appear as work using a model.

Open a model to trace its recorded parents and producing runs. The history shows available dataset revisions and manifests, episode partitions, worker-reported hardware, recorded compute selection and recipes. Missing ancestry and unrecorded details are shown explicitly. Episodes outside recorded partitions are computed only when the episode count and membership lists are available; the record may not explain why they were excluded.

Continue with a model's **Distill**, **Quantize**, **Evaluate**, **Replay observations** or **Run in simulation** action. The application switches to its owning project and selects that exact version. Compatibility and worker setup still apply. ACT training checkpoints require an inference export before distillation or packing. **Open this checkpoint's training run and export** preserves the selected checkpoint.

**Train a model** opens training setup. **Import a model** opens the existing ACT/SmolVLA archive importer and requires a configured policy profile; importing does not launch simulation. Models appear in the collection after an operation registers their artifact.

## Dataset

Choose a source and **Inspect dataset**, then open **Load visual preview** to browse recorded episodes, cameras and sampled actions. Local data requires **Prepare immutable training copy** before supported training. Preview samples are not a dataset-wide audit. [Dataset setup](getting-started.md#inspect-a-public-dataset) · [Local training](local-training.md)

## Augmentation

Choose a video dataset, camera, up to four episodes and 1–10 seconds per clip. Select an appearance edit, review Google billing disclosure, then generate and compare results. Validate motion and action labels before reuse; clips are not added to training automatically. [Setup and limits](augmentation.md)

## Teaching

Connect a teaching executor, wait for a fresh camera view, set an instruction and **Start recording**. Pause, correct joints or mark failures, then **Finish episode** and import the recording. Voice is optional; a cloud connection does not start a simulator. [Executor setup](teaching-connections.md) · [Recording](local-training.md#teach-in-simulation) · [Optional AI services](teaching-intelligence.md)

## Fine-tune

Open a saved run or start a new one, then choose data, cameras, model and compatible compute. Select a model explicitly, review the recipe, start training and follow loss, checkpoints and status. Download, resume or export where supported; model and dataset requirements differ by adapter. [Training adapters](native-training.md) · [Policy workflow](policy-workflow.md)

## Distill

Choose an exact saved teacher in **Distill** or continue from its action in **My models**. The resulting student retains a recorded link to that teacher. Only compatible local ACT → ACT256 distillation is implemented; unsupported models show the reason. [Distillation setup](../workers/policy_distillation/README.md) · [CPU workers](../workers/local_cpu/README.md)

## Quantize

Choose a saved model in **Quantize** or continue from **My models**. Its format determines the workflow: SmolVLA checkpoints use GGUF conversion; compatible ACT inference models use local packing. Defaults are Q8 and INT8; Q4 and INT4 require explicit selection. Review size, drift and reload results. Saved quantization work remains accessible when its source model is missing. [SmolVLA](policy-workflow.md) · [ACT packing](native-quantization.md)

## Evaluate

Select a policy, compute and check type, review the recipe, then start. Inference checks measure loading, finite actions, timing and memory. Configured LIBERO benchmarks measure task outcomes with the required assets and protocol; replay and Isaac recordings do not provide scored success. [Diagnostics](policy-workflow.md) · [LIBERO Spatial](spatial-workflow.md)

## Run

Choose **3D simulation** for an ACT/SmolVLA Isaac cup recording, **Replay observations** for offline ACT comparison, or **Check inference** for GGUF. Select the required inputs and review launch consent. Isaac pickup success remains unmeasured; replay and inference do not establish closed-loop success. [Isaac](native-simulation.md) · [Replay](../workers/isaac_sim/NATIVE_REPLAY.md) · [Inference](cloud-inference.md)

## Decision lab

Enter a state, instructions and 2–8 unique criteria, then **Score criteria** using the configured local Muose worker. Each comparison must fit 512 tokens. Scores are uncalibrated advice and never execute actions; the operator must accept the model's noncommercial license. [Scorer setup](../workers/decision/README.md)

## Cloud runs

Review jobs with saved cloud targets or inspect separate external rollout observations. Open linked jobs for details and check timestamps and stale-state notices. The monitor reads status; it does not create compute, control Teaching or verify resource cleanup. [Cloud monitoring](cloud-runs.md)

## Settings & diagnostics

Use **Compute** for Google Cloud, Hugging Face access and local workers. For an installed SmolVLA CUDA worker, choose **Check this machine → Add worker**, then enable local runs and save. This checks the app host; it does not install dependencies or start jobs. Use **Workflow settings** for recipes and **Diagnostics** for policy checks. [Local setup and limits](compute-settings.md#add-a-local-training-worker) · [Cloud connections](cloud-connections.md) · [Diagnostics](policy-workflow.md)
