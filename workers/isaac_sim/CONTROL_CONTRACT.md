# Supply simulator policy coordinates

Keep `control-contract.json` beside a simulator-trained policy's weights and
processors. Preserve its exact bytes and SHA-256 through checkpoint copies,
imports, resumes and export.

## Version 1 record

Supply a JSON file of at most 64 KiB using the exact fields in
`sim_worker/rollout/control_schema.py`:

- `schema_version: 1`, `kind: simulator_joint_position`,
  `controller: joint_position_targets`.
- `state_key: observation.state`, `action_key: action`, both units `radians`,
  and `timebase: simulation_seconds`.
- Unique ordered `joint_order`; `camera: {key, width, height, prim}` with the
  saved RGB camera key, even dimensions from 2 through 1920, and USD camera prim.
- Integer `action_fps` from 1 through 60.
- `source` with `dataset_snapshot_id`, `dataset_manifest_sha256`,
  `demonstrations_sha256`, sorted unique `scene_sha256` and `origins` lists,
  and `scene_hash_scope: "root USD bytes; referenced assets not inventoried"`.
  Use `sha256:<dataset_manifest_sha256>` as the snapshot ID and select origins
  from `recorded` and `synthetic`.
- `physical_calibration_verified: false`, `task_success_verified: false`.

Use the admitted dataset snapshot's joint lists, applied action targets and
source identities when producing the file. Match the saved policy feature keys
and temporal-contract FPS. Serialize it with `control_schema.canonical(record)`
for the required sorted, two-space JSON and final newline.

## Explicit Run admission

Embed `control_contract: {record, sha256}` in the rollout manifest and set
`calibration: null`. Match its joint order, FPS, camera prim/dimensions and root
USD digest to the selected policy. Keep the scene's referenced assets together.

From `workers/isaac_sim`, validate the prepared manifest before submission:

```sh
python -m sim_worker.rollout --manifest /absolute/rollout.local.yaml --validate-only
```

Then use [the policy-server and rollout commands](ROLLOUT.md) or
[the SkyPilot launcher](../skypilot/ROLLOUT.md).
