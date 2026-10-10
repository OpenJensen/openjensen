# Pack a native ACT policy

## Prepare the runtime and policy

Use a POSIX host and the [ACT CPU environment](../act_optimizer/README.md):
Python 3.12, LeRobot 0.6.1, Torch 2.11.0, torchvision 0.26.0 and safetensors 0.8.0.
Prepare a complete FP32 ACT inference policy with ResNet18, `use_vae=false`, one
RGB camera, six state/action coordinates and
`1 <= execution <= prediction <= 1024`. Preserve its processors and statistics.

## Run

From the repository root:

```sh
PYTHONPATH=workers/firebird_quant/src:workers/act_optimizer/src \
  workers/act_optimizer/.venv/bin/python -m firebird_quant.native_application \
  /absolute/request.json /absolute/new-result.json
```

Save the request with these fields. Replace the illustrated `files` entry with
the complete flat source inventory. Use the direct policy manifest's SHA256 for
`manifest_sha256`, or `null` when absent. Supply a new `output_dir` and result
path. Select integer `bits` 4 or 8 and `group_size` 64.

```json
{
  "schema_version": 1,
  "job_id": "caller-owned-job-id",
  "operation": "policy.quantize",
  "source": {
    "path": "/absolute/resolved/policy",
    "artifact_id": "caller-owned-artifact-id",
    "artifact_manifest_sha256": "<outer registered artifact SHA256>",
    "manifest_sha256": null,
    "files": {"<every flat source filename>": {"sha256": "<SHA256>", "bytes": 123}}
  },
  "output_dir": "/absolute/job/operation",
  "native_quantization": {"format": "firebird_quant", "bits": 8, "group_size": 64},
  "timeout_seconds": 600
}
```

The package is written to `output_dir/native-quantized/`:

```text
manifest.json              # schema1, metadata, all other relative files -> SHA256
lineage.json               # source IDs, complete source hashes/sizes
source-manifest.json       # exact source manifest bytes, only when one existed
verification.json          # actual generated-input baseline/candidate measurements
policy/
  config.json              # original bytes
  policy_preprocessor.json # original bytes
  policy_postprocessor.json # original bytes
  <referenced statistics>  # original bytes
  temporal-contract.json  # original bytes, when present
  control-contract.json   # original canonical bytes, when present
  model.fbq                # packed parameters and retained tensors, no float master
  encoding.json            # exact recipe/runtime/format discriminator
```

## Run tests

Use the pinned ACT interpreter:

```sh
FIREBIRD_TEST_NATIVE_ACT=1 \
PYTHONPATH=workers/firebird_quant/src:workers/act_optimizer/src \
  workers/act_optimizer/.venv/bin/python -m pytest workers/firebird_quant/tests
```

To serve the resulting policy, follow the [CPU policy-server recipe](../isaac_sim/PACKED_ACT.md).
For generated horizon checks, use the [ACT fixture commands](../act_optimizer/README.md#verify-independent-prediction-and-execution-horizons).
