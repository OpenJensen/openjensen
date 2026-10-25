# Train, quantize and evaluate policies

## The workflow

1. **Train.** Inspect a dataset and select **Train on this dataset**, or open **Fine-tune**. Choose the model, method, cameras and compute, review the recipe and start.
2. **Pick a checkpoint.** Open the saved run and select a checkpoint.
3. **Compress it.**
   - SmolVLA: open **Quantize → SmolVLA** and choose Q8 or Q4.
   - ACT: [export the checkpoint](cloud-act-export.md), then [pack it locally](native-act-quantization.md) to INT8 or INT4.
   - ACT can also be [distilled](#distill-act) into a smaller student first.
4. **Evaluate.** In **Evaluate** (or **Settings → Diagnostics**), choose the policy, a registered runtime and engine checks or LIBERO, review the protocol and start. For LIBERO Spatial, [prepare its assets](spatial-workflow.md); for cloud engine checks, see [cloud inference](cloud-inference.md); for Isaac, see [simulation setup](native-simulation.md).
5. **Download.** Each job's page shows its stages and outputs.

Every job keeps its recipe, stage requests, logs and outputs in its workspace directory. To stop a job, open it and cancel. To continue an interrupted training run, open it, select **Resume** and the last complete checkpoint; after a hard server stop, check any reported leftover worker or cluster first.

## Configure the execution host

### Local GPU training

Install the [pinned SmolVLA training environment](../workers/smolvla_qlora/docs/training/smolvla-qlora.md#install-on-the-training-machine) in `workers/smolvla_qlora/.venv` on the GPU machine. Then in **Settings → Compute → Local runs**, select **Check this machine → Add worker**, enable local runs and save. See [compute settings](compute-settings.md).

### Cloud GPU training

[Connect Google Cloud](compute-settings.md), then choose a cloud GPU in Fine-tune. Save Hugging Face access if the model requires it. [Cloud training](skypilot-training.md) covers preparation and cleanup.

### GGUF quantization and diagnostics

Install the [native worker](../workers/vla_cpp/README.md) and its [GPU build](../workers/vla_cpp/docs/gpu-setup.md). To export trained checkpoints, also install its conversion dependencies from `workers/vla_cpp`:

```sh
uv pip install --python .venv/bin/python -e '.[quantize,convert]'
```

Copy [runtime.example.json](runtime.example.json), replace every absolute path and source SHA-256, and set:

| Key | Value |
| --- | --- |
| `worker_root` / `training_root` | `workers/vla_cpp` / `workers/smolvla_qlora` |
| `vendor`, `build` | The pinned native checkout and build |
| `tokenizer` | The pinned local tokenizer directory |
| `conversion_vendor` | The conversion checkout, if built separately |
| `LD_LIBRARY_PATH` | Include the build's `bin` directory |
| `gpu_name`, `gpu_memory_mib` | Optional: the target GPU's name and memory |

For containers, set `image` and `mounts` for evaluation, `conversion_python` or `conversion_image` for conversion, and `training_python` or `training_image` for training. Mounted asset paths must be identical inside the container.

Start the server with the configuration:

```sh
export JENSEN_RUNTIME_CONFIG=/absolute/path/runtime-local.json
export JENSEN_DATA_DIR=/absolute/path/jensen-workspace
uv run --frozen jensen serve
```

### Custom training adapters

Install the adapter's Python module in the training environment and register it with the model IDs it trains:

```json
{
  "training_module": "my_openvla_adapter.application",
  "training_model_ids": ["openvla"]
}
```

## Distill ACT

Install the [ACT distillation adapter](../workers/policy_distillation/README.md) and the [CPU workers](../workers/local_cpu/README.md). In **Distill**, select an ACT teacher and local observations with the same camera, saved processors, coordinate order and units. Choose separate training, validation and final episode groups, review the recipe and submit.

## Use the CLI

Save a recipe, replacing the runtime and source IDs with values from `jensen policy options`:

```json
{
  "operation": "policy.workflow",
  "runtime_id": "local-gpu",
  "source_id": "smolvla-libero",
  "candidates": [{"language": "Q4_0"}, {"language": "Q8_0"}],
  "evaluation": {"mode": "engine", "warmups": 3, "repetitions": 10}
}
```

```sh
uv run --frozen jensen policy options
uv run --frozen jensen policy submit PROJECT_ID recipe.json
uv run --frozen jensen jobs show JOB_ID
uv run --frozen jensen jobs events JOB_ID
uv run --frozen jensen policy artifacts PROJECT_ID
```

For training, use `"operation": "policy.finetune"` with `dataset_job_id`, `training_method` and `training`. For export or quantization, reference the saved artifact.
