# Prepare LIBERO Spatial inputs

## Prepare exact local assets

Use Linux CUDA with Python 3.11, the
[benchmark requirements](../../benchmark_gpu/requirements-native.txt) and the
native worker's `quantize` extra. Cache these snapshots in directories named by
their revision:

| Resource | Snapshot revision |
|---|---|
| `lerobot/smolvla_libero` | `31d453f7edd78c839a8bbc39744a292686daf0de` |
| `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |
| `lerobot/libero-assets` | `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6` |

Use native weights with SHA256
`9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`.
Set the derived converter config's state dimension to eight while preserving the
original checkpoint. Prepare the floating GGUF with seven real action channels,
50-step chunks, 512px model images and ten denoising steps.

Capture 1–64 raw NPZ fixtures containing two FP32 RGB 360×360 images, eight state
coordinates and FP32 `1×50×32` noise. With these files already local, run:

```sh
python -m policykit.spatial_protocol \
  --policy /local/snapshots/31d453f7edd78c839a8bbc39744a292686daf0de \
  --backbone /local/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467 \
  --fixtures /local/frozen-fixtures \
  --gguf /local/smolvla-bf16.gguf \
  --output /local/new-spatial-bundle
```

Use the output path as `source.evaluation_bundle` and the printed
`spatial-assets.json` digest as `source.evaluation_bundle_sha256`. Supply the
source GGUF path/hash and `task: libero_spatial`.

## Configure the application runtime

Set `worker_root` to this checkout and `evaluation_python` or `evaluation_image`
to the prepared environment. Point `runtime.simulator_lane` to a directory with
LIBERO `config.yaml` and `assets.json`, prepared with
[`prepare_libero_assets.py`](../../benchmark_gpu/scripts/prepare_libero_assets.py).
Use the pinned asset revision above.

Build `vla-server`, `vla-bench` and `tests/vla_predict_check`, including the serving
protobuf and packed-loader patch in `runtime.vendor`. Select NVIDIA GPU index 0
or its matching `CUDA_VISIBLE_DEVICES` mapping.

Set `evaluation.suite: libero_spatial`, optional `task_ids`, disjoint
`initial_states`/`final_states`, and `parity_limits` with `profile`, `max_rmse` and
`max_abs_error` for the run.

## Run worker tests

From `workers/vla_cpp`:

```sh
uv sync --locked --extra quantize --extra test
uv run --locked --extra quantize --extra test pytest -q -rs
```
