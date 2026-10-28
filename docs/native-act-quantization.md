# Set up and submit ACT packing

Pack a local ACT inference policy to INT8 or INT4 on the CPU. This page covers the worker setup, the API recipe and what the result proves. For the steps in the web app, see [Quantize ACT in the web app](native-quantization.md).

## What you need

- A POSIX host, such as Linux or macOS.
- The pinned [ACT packing worker](../workers/jensen_quant/NATIVE_ACT.md) in an isolated interpreter, with its source in `workers/jensen_quant` beside `workers/act_optimizer`.
- A source policy that meets the [requirements below](#prepare-the-source-policy).

## Register the worker

Add this entry to the `runtimes` list in your runtime configuration, replacing the absolute paths:

```json
{
  "id": "native-act-cpu",
  "label": "Local ACT packing",
  "native_quantization_only": true,
  "native_quantization_python": "/absolute/isolated-act-environment/bin/python",
  "native_quantization_root": "/absolute/openjensen/workers/jensen_quant"
}
```

Set `JENSEN_RUNTIME_CONFIG` to the file, restart the API, and enable local runs in **Settings → Compute → Local runs**.

## Prepare the source policy

Use a local ACT `inference_export`, or an imported `native_checkpoint`, saved in the project. For a training checkpoint, [export it first](cloud-act-export.md). The policy needs:

- FP32 weights, `config.json`, both saved processors and their normalization statistics.
- A ResNet18 backbone and `use_vae=false`.
- One RGB camera and six state/action coordinates.
- Saved horizons with `1 <= execution <= prediction <= 1024`, one observation step and no temporal ensembling.

## Submit

Save this recipe, replacing the runtime and registered artifact IDs:

```json
{
  "operation": "policy.quantize",
  "runtime_id": "native-act-cpu",
  "artifact_id": "registered-source:operation",
  "native_quantization": {"format": "firebird_quant", "bits": 8, "group_size": 64},
  "timeout_seconds": 600
}
```

| Key | Allowed values |
| --- | --- |
| `bits` | 4 or 8 |
| `group_size` | 64 |
| `timeout_seconds` | An integer from 30 to 600 seconds |

Submit it with `jensen policy submit PROJECT_ID recipe.json`, or `POST /api/v1/projects/{project_id}/policy-jobs`.

The `format: "firebird_quant"` value is a compatibility discriminator stored in packages. Worker imports and paths use `jensen_quant`. Do not rename the discriminator inside an existing package.

## Follow, download and replay

- Open the saved job for its events and its registered `native_quantized` output, and download the package from the result. The [worker guide](../workers/jensen_quant/NATIVE_ACT.md) lists the package layout.
- To replay it, install the [replay worker](../workers/isaac_sim/NATIVE_REPLAY.md), choose **Replay recorded observations** and select the observations.
- Cancel the selected job to stop the packing process.
- If admission fails, check the source manifest, saved processor files and runtime paths before submitting a new request.

## What the result proves

- A fresh CPU reload establishes agreement with the packed candidate.
- Drift from FP32 is measured on generated inputs only.
- Generated-input drift and reduced storage do not establish calibrated task quality, physical units, GPU memory savings or inference speedups. See the [support matrix](support.md).
