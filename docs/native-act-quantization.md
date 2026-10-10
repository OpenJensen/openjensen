# Local native ACT quantization

An explicit `policy.quantize` request with `native_quantization` uses the isolated
`firebird_quant` CPU worker. The existing SmolVLA GGUF workflow is unchanged.
The result is a downloadable `native_quantized` package, not an Isaac, evaluation,
training-resume or deployment-ready artifact.

The first supported source is an application-owned, local ACT `inference_export`
or `native_checkpoint` containing one complete flat inference policy: saved FP32
weights, config, both saved processors and their statistics; `use_vae=false`,
one camera, six state/action coordinates and 100-action chunks. A complete ACT
training checkpoint must first pass the existing inference-export operation.
Remote descriptors, bare weights and other model families are rejected before
launch. An imported policy still undergoes the worker's full pinned-runtime
admission; a format label alone is not evidence of compatibility.

## Operator configuration

Register a dedicated entry in the existing runtime catalog. Paths are operator
configuration and are never accepted from browser requests:

```json
{
  "id": "native-act-cpu",
  "label": "Local ACT packing",
  "native_quantization_only": true,
  "native_quantization_python": "/absolute/isolated-act-environment/bin/python",
  "native_quantization_root": "/absolute/openjensen/workers/firebird_quant"
}
```

The worker root must sit beside `workers/act_optimizer`; both source modules and
an executable interpreter must exist. The isolated interpreter uses the worker's
pinned ACT consumer tuple. See `workers/firebird_quant/NATIVE_ACT.md`. No runtime
installation or cloud configuration occurs through this API. Enable local compute
through the existing compute settings. `native_quantization=true` means the
configured files allow an attempt, not that any selected model was validated.

```json
{
  "operation": "policy.quantize",
  "runtime_id": "native-act-cpu",
  "artifact_id": "registered-source:operation",
  "native_quantization": {"format": "firebird_quant", "bits": 8, "group_size": 64},
  "timeout_seconds": 600
}
```

Submit to the existing project `policy-jobs` endpoint. Bits are explicitly 4 or
8; the group size is 64. The default deadline for this recipe is 600 seconds,
with an accepted range of 30–600. It bounds the operation; process cleanup can
add up to eight seconds plus bounded local receipt cleanup. Cancellation uses
the same job endpoint and drains the owned worker before publishing any result.
No automatic resubmission or retry occurs. Worker children are local CPU
processes with an offline environment, not an operating-system sandbox.

## What the receipt proves

The application verifies the registered source manifest, complete inventory and
metadata before execution and checks them again after exit or cancellation. The
worker uses a private snapshot, packs the supported weights and performs a fresh
packed-only CPU reload. The application then independently hashes the exact
published files, recomputes the packed policy identity and binds the recipe,
lineage and complete finite generated action chunks to the source. Config and
processor bytes must remain unchanged; no floating master may be included.

`drift_from_fp32` reports raw and postprocessed differences on two generated
observations. Exact fresh reload means the packed candidate matches its fresh
packed reload; it does not mean the packed candidate matches FP32. These
observations do not measure task success, calibration, GPU memory or speedup.
The package records those claims as false or null. Compatible saved results can
open **Run → Replay observations** through **Replay recorded observations**,
using the separate [offline replay worker](../workers/isaac_sim/NATIVE_REPLAY.md).
That action prepares a replay; downloading a package does not establish simulator
support or start a job.
