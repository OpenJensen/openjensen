# Run local ACT quantization

## Operator configuration

Install the pinned [ACT packing worker](../workers/firebird_quant/NATIVE_ACT.md). Keep its source beside `workers/act_optimizer` and supply the isolated interpreter in the runtime configuration:

```json
{
  "id": "native-act-cpu",
  "label": "Local ACT packing",
  "native_quantization_only": true,
  "native_quantization_python": "/absolute/isolated-act-environment/bin/python",
  "native_quantization_root": "/absolute/openjensen/workers/firebird_quant"
}
```

Replace the absolute paths, set `FIREBIRD_RUNTIME_CONFIG`, restart the API and enable local compute in Settings.

Prepare a local ACT `inference_export` or imported `native_checkpoint` with FP32 weights, config, saved processors/statistics, `use_vae=false`, one camera, six state/action coordinates and 100-action chunks. For a training checkpoint, [export it first](cloud-act-export.md).

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

Use bits 4 or 8, group size 64 and a timeout from 30–600 seconds. Submit it through `firebird policy submit PROJECT recipe.json` or `POST /api/v1/projects/{project_id}/policy-jobs`.

## Follow and download

Open the saved quantization job for events and its registered `native_quantized` output. Download the package from the result. To replay it, install the [replay worker](../workers/isaac_sim/NATIVE_REPLAY.md), choose **Replay recorded observations** and select the observations.

Cancel the selected job to stop the packing process. If admission fails, check the source manifest, saved processor files and runtime paths before submitting a new request.
