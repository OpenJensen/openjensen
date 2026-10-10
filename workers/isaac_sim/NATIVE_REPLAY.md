# Local packed ACT observation replay

This bounded `policy.run` worker executes an exact complete packed ACT policy on
explicit immutable inputs using the existing policy HTTP protocol. It records
full100×6 postprocessed action chunks. **Actions are never applied to a robot or
simulator.** This is a CPU runtime check and observation replay, not scored task
evaluation. It does not enable packed CUDA/Isaac execution or SmolVLA support.

The source must be the complete inner policy produced by the native quantization
worker: config, encoding, INT8/INT4 packed weights, processors and statistics.
The admitted modelID and every source file must match. No floating master,
remote model, arbitrary code or download fallback is accepted. Original FP32,
SmolVLA and cloud routes are unchanged.

## Two isolated runtimes

Model execution uses the existing pinned ACT environment: Python3.12,
LeRobot0.6.1, Torch2.11.0, torchvision0.26.0 and safetensors0.8.0. Expose these
source roots through `PYTHONPATH`:
`workers/isaac_sim:workers/firebird_quant/src:workers/act_optimizer/src`.
No installation or environment mutation occurs at execution time.

Dataset preparation uses the separate existing LeRobot0.6.2 dataset environment.
Add `workers/smolvla_qlora/src` to its source path so the established local snapshot
verifier and offline dataset reader can be reused. Preparation verifies the whole
snapshot before/after reading and decodes only the explicit selected frames. It
preserves native RGB8 pixels, float32 state, episode/frame/time, task text, origin
and declared lineage. Camera dimensions must match exactly; no implicit resize,
unit conversion, augmentation or random selection occurs.

Neither equal shapes nor a model's successful load establish coordinate
compatibility. The operator must supply six matching state/action names and six
units, and explicitly attest `operator_attested_policy_recorded_coordinates` for
recorded sources. Generated snapshot origins require `generated_fixture` and
remain labeled generated; mixed synthetic/recorded selection is rejected. Unknown
ancestry is retained as null rather than invented. This is not a split-quality or
holdout claim.

## Prepare selected recorded inputs

Invoke `python -m sim_worker.rollout.native_replay_prepare REQUEST NEW_RESULT` in
the dataset environment. REQUEST has exactly:

- `schema_version: 1`
- `source: {path, files, model_id, artifact_id, artifact_manifest_sha256}` — complete
  flat packed-policy inventory entries `{sha256, bytes}` and registered outer ID.
- `dataset_snapshot: {path, id, manifest_sha256}` — existing verified local snapshot.
- `selection: [{episode_index, frame_index}, ...]` —1..32 unique explicit frames.
- `semantics: {state_names, action_names, units, compatibility}` — explicit values.
- `output_dir` — absolute operator-owned new observation directory.

Preparation has bounded data, not an independent elapsed-time supervisor. Its
caller must own a process/deadline when exposed through the application. The
128MiB aggregate RGB budget is checked before decode. The output contains
`manifest.json` plus canonical `frame-NNNNNN.rgb` files and returns their hashes.
Inputs are independently reset for prediction; sparse selected frames are not
presented as contiguous executed motion.

## Execute a bounded replay

Invoke `python -m sim_worker.rollout.native_replay REQUEST NEW_RESULT` in the model
environment. REQUEST has exactly:

```json
{
  "schema_version": 1,
  "job_id": "owned-job-id",
  "operation": "policy.run",
  "source": {
    "path": "/absolute/complete-packed-policy",
    "files": {"config.json": {"sha256": "<SHA256>", "bytes": 123}},
    "model_id": "sha256:<native packed identity>",
    "artifact_id": "registered-packed-policy",
    "artifact_manifest_sha256": "<registered outer manifest SHA256>"
  },
  "observations": {
    "path": "/absolute/prepared-observations",
    "manifest_sha256": "<SHA256>"
  },
  "output_dir": "/absolute/owned-job-output",
  "timeout_seconds": 600
}
```

The abbreviated `files` mapping must contain **all** policy files. There is no
implicit generated-observation mode or default. Source identity is verified
before and after work. Both inputs are copied to private verified snapshots;
the child cannot read the originals through Python file access. Its allowlisted
environment excludes provider/cloud keys and proxies. Python audit hooks permit
only its own ephemeral127.0.0.1 server connection, and reject floating-master
reads. These are application controls, not an operating-system sandbox.

A fresh process loads the policy once, starts the existing serialized HTTP
server, and calls health/reset/predict. Every selected observation is reset and
predicted twice; complete action chunks must match exactly across those resets.
Original dataset timestamps remain in provenance, while each independent HTTP
replay begins at transport step/time zero. The server is closed/joined before the
child exits. The POSIX supervisor owns its process group, handles repeated signals
and the spawn window, enforces a1..600second supervised-work deadline, and reaps
descendants even after leader exit. Initial validation and blocking copy/hash IO
are outside hard elapsed-time guarantees; the caller still owns its outer job.
Cleanup uncertainty remains a failure. No unmanaged persistent server is left.

## Result and limits

The exclusive `native-run/` directory contains `predictions.json`, `lineage.json`,
`report.json` and a core-compatible `manifest.json` with payload hashes. Files and
directories are synchronized before successful publication. Failures after a
rename may leave an unregistered artifact but never a successful result receipt;
the caller must only register a fully accepted job result. Existing outputs are
never replaced.

The response is `{schema_version:1, job_id, operation:"policy.run", artifact,
report}`; artifact format is `native_run_record`. The report has stage
`native_replay`, mode `independent_observation_replay`, source identity, actual
runtime versions, CPU device, input count,100×6 output shape, explicit coordinate
semantics and exact reset-comparison evidence. Output checks independently bind
all records to their source image/state and both action hashes before publication.
Lineage includes every source/input hash and worker/transport implementation hash.

Per-request elapsed time includes local HTTP/serialization overhead. It is not a
comparative latency claim, GPU-memory benchmark or real-time certification.
`task_success` stays null; quality, calibration, speedup and Isaac runtime flags
stay false. CPU computation uses eager float32 with packed weight storage.

Tests cover strict types, dimensions, coordinate attestations, origin integrity,
source mutation, link/path bounds, aggregate admission before RGB IO, output
coverage/hashes/claims, signals/deadlines, no overwrite and persistence failure.
Pinned-runtime tests exercise actual generated ACT INT8/INT4 models through the
full supervisor and HTTP path. Actual user weights and native-writer fixtures
are separate evidence: generated observations never become robot task proof.

Application registration and UI dispatch require the configured replay adapter.
Recorded observations must match the policy's camera, state, action coordinates
and cadence, with explicit operator attestation. Generated fixture checks do not
establish physical calibration, task quality or simulator execution.
