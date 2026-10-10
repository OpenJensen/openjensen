# Set up and run policy jobs

## Configure a local native execution host

Install workers on the API/GPU host. For SmolVLA, install the [pinned training environment](../workers/smolvla_qlora/docs/training/smolvla-qlora.md#install-on-the-training-machine) under `workers/smolvla_qlora/.venv`. In **Settings & diagnostics → Compute → Local runs**, select **Check this machine → Add worker**, enable local runs and save.

For GGUF conversion, packing and diagnostics, install the [native worker](../workers/vla_cpp/README.md) and [instrumented GPU build](../workers/vla_cpp/docs/gpu-setup.md). From `workers/vla_cpp`, install conversion dependencies with `uv pip install --python .venv/bin/python -e '.[quantize,convert]'` when exporting a trained checkpoint.

Copy [runtime.example.json](runtime.example.json) and replace all absolute paths/source SHA-256 values. Supply:

- `worker_root`: `workers/vla_cpp`; `training_root`: `workers/smolvla_qlora`.
- `vendor` and `build`: pinned native checkout/build paths.
- `tokenizer`: pinned local tokenizer directory.
- `conversion_vendor`: the conversion checkout when using a separate build.
- `LD_LIBRARY_PATH`: include the build's `bin` directory for shared libraries.
- Optional `gpu_name` and positive `gpu_memory_mib`: the target GPU's specifications.

Start the server with that configuration:

```sh
export FIREBIRD_RUNTIME_CONFIG=/absolute/path/runtime-local.json
export FIREBIRD_DATA_DIR=/absolute/path/firebird-workspace
uv run --frozen firebird serve
```

For a custom training adapter, install its Python module in the training environment and register it with its model IDs, for example:

```json
{
  "training_module": "my_openvla_adapter.application",
  "training_model_ids": ["openvla"]
}
```

Set `image`/`mounts` for Docker evaluation, `conversion_python`/`conversion_image` for a separate conversion environment, and `training_python`/`training_image` for a separate training environment. Keep mounted asset paths identical inside the container.

## Cloud GPU setup

[Connect Google Cloud](compute-settings.md), then select the GPU in Fine-tune. Save Hugging Face access when required by the model. Follow [cloud training](skypilot-training.md) for preparation and cleanup.

## Train, export and quantize

1. Inspect the dataset and select **Train on this dataset** or open **Fine-tune**.
2. Select model, method, cameras and compute; review the recipe and start.
3. Open the saved run and select a checkpoint.
4. For SmolVLA, continue to **Quantize → SmolVLA** and choose Q8 or Q4. For ACT, [export the checkpoint](cloud-act-export.md), then [pack it locally](native-act-quantization.md).
5. Open the new job to follow stages and download its artifact.

## Distillation

Install the [ACT distillation adapter](../workers/policy_distillation/README.md) and [CPU workers](../workers/local_cpu/README.md). Select an ACT teacher and compatible local observations, matching camera, saved processors, coordinate order and units. Choose separate training, validation and final episode groups, review the ACT256 recipe, and submit.

## Evaluate and run

In **Evaluate** or **Settings & diagnostics → Diagnostics**, select the project policy and registered runtime, choose engine or LIBERO checks, edit the protocol and start. For LIBERO Spatial, [prepare its assets and parity settings](spatial-workflow.md). For cloud engine checks, follow [cloud inference](cloud-inference.md). For Isaac, follow [simulation setup](native-simulation.md).

## CLI and persistence

Save a JSON recipe and replace runtime/source IDs with values from `policy options`:

```json
{
  "operation": "policy.workflow",
  "runtime_id": "rtx3070",
  "source_id": "smolvla-libero",
  "candidates": [{"language": "Q4_0"}, {"language": "Q8_0"}],
  "evaluation": {"mode": "engine", "warmups": 3, "repetitions": 10}
}
```

```sh
uv run --frozen firebird policy options
uv run --frozen firebird policy submit PROJECT_ID recipe.json
uv run --frozen firebird jobs show JOB_ID
uv run --frozen firebird jobs events JOB_ID
uv run --frozen firebird policy artifacts PROJECT_ID
```

Use `policy.finetune` with `dataset_job_id`, `training_method` and `training` for training, or select the saved artifact for an export/quantization request. Recipes, stage requests, logs and outputs are retained in the job's workspace directory.

## Job history and creation

Select a saved job for status/events and downloads. Open its new-job action to submit another recipe. Cancel that selected job to stop its worker. For an interrupted training run, select Resume and the last complete checkpoint. Inspect any reported orphan worker/cluster before resuming work after a hard server stop.
