# Run local ACT quantization

## Operator configuration

Install the pinned [ACT packing worker](../workers/jensen_quant/NATIVE_ACT.md). Keep its source beside `workers/act_optimizer` and supply the isolated interpreter in the runtime configuration:

```json
{
  "id": "native-act-cpu",
  "label": "Local ACT packing",
  "native_quantization_only": true,
  "native_quantization_python": "/absolute/isolated-act-environment/bin/python",
  "native_quantization_root": "/absolute/openjensen/workers/jensen_quant"
}
```

Replace the absolute paths, set `JENSEN_RUNTIME_CONFIG`, restart the API and enable local compute in Settings.

Prepare a local ACT `inference_export` or imported `native_checkpoint` with FP32 weights, config, saved processors/statistics, ResNet18, `use_vae=false`, one RGB camera and six state/action coordinates. Saved horizons require `1 <= execution <= prediction <= 1024`; temporal ensembling is unsupported. For a training checkpoint, [export it first](cloud-act-export.md).

## Submit

Save this recipe, replacing its runtime and registered artifact IDs:

```json
{
  "operation": "policy.quantize",
  "runtime_id": "native-act-cpu",
  "artifact_id": "registered-source:operation",
  "native_quantization": {"format": "firebird_quant", "bits": 8, "group_size": 64},
  "timeout_seconds": 600
}
```

Use bits 4 or 8, group size 64 and a timeout from 30–600 seconds. Submit it through `jensen policy submit PROJECT recipe.json` or `POST /api/v1/projects/{project_id}/policy-jobs`.

The saved `format: "firebird_quant"` discriminator remains a compatibility protocol. Current worker imports and paths use `jensen_quant`; do not rename the discriminator inside an existing package.

## Follow and download

Open the saved quantization job for events and its registered `native_quantized` output. Download the package from the result. To replay it, install the [replay worker](../workers/isaac_sim/NATIVE_REPLAY.md), choose **Replay recorded observations** and select the observations.

Cancel the selected job to stop the packing process. If admission fails, check the source manifest, saved processor files and runtime paths before submitting a new request.

Fresh CPU reload checks establish agreement with the packed candidate. Generated-input drift and reduced storage do not establish calibrated task quality, physical units, GPU memory savings or inference speedups. See the [support matrix](support.md).
