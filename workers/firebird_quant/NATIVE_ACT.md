# Native ACT packed policy operation

This optional worker turns one complete, inference-ready ACT policy into a packed
policy directory. It runs on the existing isolated ACT CPU environment; the base
Firebird Quant library and SmolVLA GGUF paths are unchanged. It does not itself
register an application capability or make the package runnable in Isaac.

The first accepted recipe is exactly the existing ACT export validator: LeRobot
0.6.1, Torch 2.11.0, torchvision 0.26.0 and safetensors 0.8.0; FP32, ResNet18,
`use_vae=false`, one RGB camera, six state/action coordinates, and saved independent
prediction/execution horizons with `1 <= execution <= prediction <= 1024`. Images may use the validator's supported dimensions up to 1920×1080.
This is a bounded tested ACT recipe, not a general restriction on ACT or a
claim of native SmolVLA support. No environment or dependency pin is changed.

## Request and execution

Use a trusted interpreter that already has the exact ACT runtime. From the
repository root, expose both worker source packages without installing either
into the API environment:

```sh
PYTHONPATH=workers/firebird_quant/src:workers/act_optimizer/src \
  workers/act_optimizer/.venv/bin/python -m firebird_quant.native_application \
  /absolute/request.json /absolute/new-result.json
```

The request requires exactly these fields:

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

`files` must enumerate every source file, not just the illustrated entry. The
source is a resolved complete policy directory, so an imported wrapper and an
ACT export wrapper do not need the same parent layout. `manifest_sha256` is the
exact hash of a direct policy `manifest.json`, or null only when it is absent.
The worker checks direct ACT export or generic inventory manifests when present.
The parent application must verify its registered outer artifact and pass the
corresponding ID/hash; the worker records that binding and verifies the complete
supplied inner inventory, without pretending to authenticate a caller's registry.

Only integer 4/8-bit, group size 64 and the default tensor-selection recipe are
accepted. Unknown fields, booleans masquerading as integers, nonfinite JSON,
nonlocal/traversing paths, symlinks, nonregular files, changed/missing/extra bytes,
unsupported policies and an existing destination fail explicitly. Each policy
has at most 16 flat files, the weight file at most 512 MiB, each metadata/statistics
file at most 1 MiB, and all files at most 768 MiB. These are narrower bounds than
the generic Isaac archive importer. A training checkpoint must first pass the
existing inference-export step; the quantizer does not modify resume state.

On POSIX, one owner supervises separate conversion and fresh-reload processes.
Its signal handlers record interruption even during spawn registration; cleanup
terminates the entire owned process group, including descendants after leader
exit, and restores prior handlers. A shared 1..600-second deadline bounds child
execution, with at most five seconds of final leader reap. It never retries.
Regular-file admission/publication uses bounded reads and checks the deadline
between phases; an OS filesystem stall is not a hard real-time guarantee.
Windows ownership is unsupported by this first native worker.

The children do not inherit cloud/provider credentials. They force CPU/offline
settings, disable network through a Python audit hook, use trusted LeRobot code,
and never load pickle or supplied architecture code. These are software guards,
not an OS sandbox. The final reload additionally denies every Python file-open
named `model.safetensors`; the packed directory contains no floating master.

## Published artifact

The new destination is `output_dir/native-quantized/`:

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

Publication is an atomic no-replace directory rename after source revalidation
and both runtime checks. Failure or interruption cannot produce a success result;
a process killed after the publication boundary may leave an intact unregistered
artifact for the owner to inspect. The operation never overwrites the source or
an earlier destination. The result contains `schema_version`, `job_id`,
`operation`, `artifact` (`path`, `format=native_quantized`, `label`) and `report`.
Its outer manifest matches the existing application's artifact convention.

`encoding.json` fixes schema/format version 1, `format=firebird_quant`,
`policy_family=act`, `weights=model.fbq`, `compute_dtype=float32`, runtime versions
and the default recipe. A model identity is SHA256 initialized with
`firebird-native-packed-policy-v1\0`; for every sorted inference filename it
hashes its UTF-8 name length (unsigned 64-bit big-endian), name bytes, file size
(unsigned 64-bit big-endian), and raw 32-byte file SHA256. Thus config, processors,
statistics, optional temporal/control contracts, encoding and packed weights all
affect identity. Verification and lineage are hashed by the outer manifest but
do not introduce a model-ID cycle.
A consumer must validate the format and recompute identity before runtime load.

An optional simulator control contract is validated against the saved feature
names/shapes and temporal cadence. It is copied unchanged, included in the packed
identity, and reported as `control_contract` plus `control_contract_sha256` in
both worker probes, verification, manifest metadata and the operation report.
The owner rejects a dropped or changed sidecar, configuration or processor before
publication. Legacy policies omit both control fields; absent/null legacy probe
claims remain compatible. A canonical sidecar whose value is null is invalid.

This preservation does not enable CUDA serving, physical calibration or genuine
Isaac acceptance. The packed consumer remains CPU-only. Current cloud Run profiles
also need an explicit compatible policy runtime and its worker dependencies;
this change does not modify or activate those profiles. Structural tests can
establish byte/identity preservation, but native conversion/reload and HTTP tests
must be explicitly run in the pinned environment before claiming runtime evidence
for a simulator-bound packed policy.

## Evidence meaning

The converter uses the real saved pre/postprocessors on two explicitly generated
image/state observations (seeds 171 and 902). It retains the complete raw and
postprocessed prediction-horizon × 6 arrays, input hashes, and real queue refills
at the saved execution horizon plus reset checks.
The report computes raw/postprocessed RMSE and maximum absolute differences
against FP32. There is no invented acceptable-drift threshold. Postprocessed
units are the saved output coordinates; physical units/calibration are unverified.

Fresh packed reload must reproduce the candidate's exact arrays and queue/reset
behavior. **That exactness is packed-versus-itself, not parity with FP32.**
`fresh_reload_verified` and `cpu_reload_verified` describe only this CPU test.
`runtime_verified`, `isaac_runtime_verified`, `quality_verified`,
`calibration_verified` and `speedup_verified` remain false; `task_success`, GPU
memory and speedup remain null. Generated observations are not held-out robot
episodes, a simulator evaluation, or a successful cup pickup.

The report separates source weight bytes, packed file bytes and the complete
inference payload bytes (config/processors/stats/encoding/weights). The latter
excludes the outer evidence envelope. Eager operations expand accessed weights,
and conversion retains multiple models; these file counts are not peak memory
or latency measurements. No quantized GPU execution or native SmolVLA checkpoint
has been qualified by this operation.

## Tests

The ordinary worker suite runs contract, tampering, no-replace and real process
ownership tests without LeRobot. Two explicit runtime tests require the existing
pinned ACT environment and generated ACT weights; they do not download models:

```sh
FIREBIRD_TEST_NATIVE_ACT=1 \
PYTHONPATH=workers/firebird_quant/src:workers/act_optimizer/src \
  workers/act_optimizer/.venv/bin/python -m pytest workers/firebird_quant/tests
```

Native SmolVLA is a follow-on adapter: it needs a genuine complete checkpoint,
offline VLM config/processor/tokenizer assets, exact dtype/tie handling and the
same explicit flow-matching noise for FP32/candidate/reload measurements. A packed
file alone does not satisfy those gates. The existing GGUF lane remains separate.

## Local CPU serving consumer

The complete packed policy now has a strict optional consumer in
`firebird_quant.native_consumer`, used by the existing
[simulation policy HTTP server](../isaac_sim/PACKED_ACT.md).
It preserves the producer model identity and saved processors, rejects unsupported
devices and runtime versions, and loads a private verified byte snapshot.
CPU serving acceptance does not change the bundle's historical quality,
calibration, GPU or simulator verification flags.


Temporal portability is covered by the generated ACT producer/export/HTTP
[fixture workflow](../act_optimizer/README.md#changed-horizon-software-acceptance).
`temporal-contract.json`, when supplied, is validated against config, copied
byte-for-byte and included in the packed policy identity. Report drift uses
`prediction_horizon * 6` coordinates, while execution length remains separate.
Old 100/100 packages and reports without additive temporal metadata remain readable.
