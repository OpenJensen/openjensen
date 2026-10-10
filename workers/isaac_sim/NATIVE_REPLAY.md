# Replay packed ACT observations locally

## Two isolated runtimes

Prepare the model runtime with Python **3.12**, LeRobot **0.6.1**, Torch **2.11.0**,
torchvision **0.26.0** and safetensors **0.8.0**. Set its `PYTHONPATH` to
`workers/isaac_sim:workers/firebird_quant/src:workers/act_optimizer/src` from the
repository root.

Prepare a separate LeRobot **0.6.2** dataset runtime; add
`workers/smolvla_qlora/src` to its source path. Supply a complete INT8/INT4 packed
ACT policy, a verified local snapshot and exact camera dimensions. Supply six
matching state/action names and units. For recorded inputs set compatibility to
`operator_attested_policy_recorded_coordinates`; for generated origins use
`generated_fixture`. Select one origin type per request.

## Prepare selected recorded inputs

Save a private request JSON with exactly:

- `schema_version: 1`.
- `source: {path, files, model_id, artifact_id, artifact_manifest_sha256}`;
  include every packed-policy file as `{sha256, bytes}`.
- `dataset_snapshot: {path, id, manifest_sha256}`.
- `selection: [{episode_index, frame_index}, ...]`; choose 1–32 unique frames.
- `semantics: {state_names, action_names, units, compatibility}`.
- `output_dir`: an absolute new observation directory; fit within the aggregate
  128 MiB RGB budget.

Run in the dataset runtime, using new result/output paths:

```sh
python -m sim_worker.rollout.native_replay_prepare REQUEST_JSON NEW_RESULT_JSON
```

Read the returned `manifest.json` identity and canonical `frame-NNNNNN.rgb` paths.

## Execute a bounded replay

Save this request in the model runtime. Replace the abbreviated `files` mapping
with the complete policy inventory and fill all registered source identities:

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

Use an integer `timeout_seconds` from 1 through 600 and new output/result paths:

```sh
python -m sim_worker.rollout.native_replay REQUEST_JSON NEW_RESULT_JSON
```

Keep the process supervised through completion. To cancel, signal that exact
worker process and wait for its owned process group to exit.

## Read outputs

Read `native-run/predictions.json`, `lineage.json`, `report.json` and
`manifest.json`. Match the job/source identities, camera/state input hashes,
full postprocessed action chunks and reset-comparison hashes before registering
an output. Keep output directories unique; preserve failed jobs for inspection.
