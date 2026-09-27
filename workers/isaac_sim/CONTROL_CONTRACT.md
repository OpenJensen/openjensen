# Simulator-native policy coordinates

A policy trained from an admitted simulator recording can carry the canonical
`control-contract.json` sidecar. The record is optional for legacy policies. If a
known record is present, training resume, ACT inference export, checkpoint import
and Run must preserve it and its exact SHA256; absence is not a request to fall
back to physical calibration. Packed ACT structural admission preserves the exact
record and temporal file through the CPU quantizer, independent reload reports
and importer. Distillation and application transform admission remain gated until
their own reviewed implementations preserve this provenance; structural packed
admission alone does not activate an application or cloud Run route.

## Version 1 record

The bounded 64 KiB JSON record has exact fields and types. `schema_version` is 1;
`kind` is `simulator_joint_position`; `controller` is `joint_position_targets`;
`state_key` and `action_key` are `observation.state` and `action`. State and action
units are `radians`, and the timebase is `simulation_seconds`.

`joint_order` lists the unique ordered joint names. `camera` contains the native
observation key, RGB width and height, and USD camera prim. `action_fps` is an
integer from 1 through 60. `source` binds the snapshot ID and manifest SHA256,
recording-provenance file SHA256, sorted root scene hashes and original source
origins (`recorded` or `synthetic`). `physical_calibration_verified` and
`task_success_verified` are always false. Paths, credentials and capture contents
are not embedded. The exact schema is in `sim_worker/rollout/control_schema.py`;
the isolated producer and exporter ship byte-identical copies.

The snapshot verifier and training adapter validate all source file identities
before deriving the record. Joint names must match both native feature lists;
the action column denotes applied simulator targets, not requested corrections.
The contract then binds the resolved ACT/SmolVLA feature keys and dimensions. The
producer and exporter validate any saved temporal record's FPS independently.

## Explicit Run admission

An admitted rollout manifest embeds `control_contract: {record, sha256}` and sets
`calibration: null`. Before allocating a runtime, admission requires matching
ordered joints, control FPS, camera prim and image dimensions, and a root USD
digest listed in the policy provenance. The launcher binds this envelope to the
selected checkpoint; neither a missing contract nor a different envelope is
accepted for a known simulator policy. The contract file also contributes to the
native model identity used by the HTTP protocol.

The admitted mapping is a finite radian identity. It always retains the existing
experimental motion guard, using live joint limits and bounded target speed.
Reports explicitly record `coordinate_mapping: simulator_native_radians`, the
contract SHA256, `calibration_status: not_applicable_simulator`, and
`physical_calibration_verified: false`. The legacy `calibration_sha256` receipt
field contains the contract digest on this route. `experimental` remains true.

## Acceptance limits

This is declared data provenance, not proof that an operator's origin label is
true. Root USD bytes do not inventory referenced assets, articulation topology,
robot geometry or controller behavior. The recorder currently does not bind an
articulation prim or complete asset closure; a genuine Isaac acceptance run must
check those against both recording and execution. No physical-to-simulator map is
validated by this route, and the rejected SO101 calibration remains rejected.

The source regression fixtures are synthetic and run without Torch or Isaac.
Their passing results establish schema, identity, portable export metadata and
mismatch rejection only. Native training, fresh model export/reload, actual Isaac
execution and task quality require separately recorded acceptance evidence.
