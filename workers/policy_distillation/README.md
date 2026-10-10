# Run ACT distillation

## Install the model runtime

Use a POSIX host with Python 3.12 and the
[ACT CPU environment](../act_optimizer/README.md): LeRobot 0.6.1,
Torch 2.11.0, torchvision 0.26.0 and safetensors 0.8.0. Install local source
packages with `pip install --no-deps -e workers/act_optimizer -e workers/policy_distillation -e workers/smolvla_qlora`,
or expose the source roots in the command below.

Prepare a complete ACT teacher with ResNet18, one camera, six state/action
coordinates and `1 <= execution <= prediction <= 1024`. Include its saved
processors, statistics and optional timing/control records.

## Prepare the corpus

Use the separate LeRobot 0.6.2 dataset environment from the
[local CPU installer](../local_cpu/README.md). Add `workers/policy_distillation/src`,
`workers/act_optimizer/src` and `workers/smolvla_qlora/src` to `PYTHONPATH`, then
run `python -m firebird_distill.prepare PREPARE_REQUEST NEW_RESULT`.

Save this request with real paths, snapshot identity, matching feature names,
units and three nonempty disjoint episode/lineage partitions:

```json
{
  "schema_version": 1,
  "teacher": "/absolute/complete-native-act-policy",
  "dataset_snapshot": {
    "path": "/absolute/verified-snapshot",
    "id": "sha256:<manifest-sha256>", "manifest_sha256": "<manifest-sha256>"
  },
  "splits": {"train": [0, 1], "validation": [2, 3], "final": [4, 5]},
  "frame_stride": 30,
  "semantics": {
    "state_names": ["joint0", "joint1", "joint2", "joint3", "joint4", "gripper"],
    "action_names": ["joint0", "joint1", "joint2", "joint3", "joint4", "gripper"],
    "units": ["degrees", "degrees", "degrees", "degrees", "degrees", "recorded_gripper"],
    "compatibility": "operator_attested_teacher_recorded_coordinates"
  },
  "output_dir": "/absolute/new-corpus"
}
```

Set `compatibility` to `operator_attested_teacher_recorded_coordinates` for
recorded data and `generated_fixture` for generated data. Use the teacher's
camera resolution, feature order, coordinate system and cadence. For a
simulator-bound teacher, select the immutable snapshot named by its control record.
The new corpus is written to `output_dir`; use its manifest SHA256 below.

## Run the student job

Save a request with a complete flat teacher inventory and a new output directory:

```json
{
  "schema_version": 1,
  "job_id": "distillation-001",
  "operation": "policy.distill",
  "teacher": {
    "path": "/absolute/complete-native-act-policy",
    "files": {"config.json": {"sha256": "<sha256>", "bytes": 123}},
    "artifact_id": "<registered source artifact>",
    "artifact_manifest_sha256": "<registered outer manifest sha256>"
  },
  "dataset": {"path": "/absolute/prepared-corpus", "manifest_sha256": "<sha256>"},
  "recipe": {
    "adapter": "act-act-v1", "student": "act-256",
    "steps": 100, "learning_rate": 0.0001, "seed": 1729
  },
  "output_dir": "/absolute/job-output",
  "timeout_seconds": 600
}
```

Run from the repository root:

```sh
PYTHONPATH=workers/policy_distillation/src:workers/act_optimizer/src:workers/smolvla_qlora/src \
  workers/act_optimizer/.venv/bin/python -m firebird_distill.application \
  /absolute/operator-owned/request.json /absolute/new-result.json
```

Read `artifact.path` from the result. The package contains `policy/`,
`training.json`, `verification.json`, `lineage.json` and `manifest.json`.

## Run tests

```sh
PYTHONPATH=workers/policy_distillation/src:workers/act_optimizer/src:workers/smolvla_qlora/src \
  workers/act_optimizer/.venv/bin/python -m pytest workers/policy_distillation/tests -q
```

For the native 8/3 fixture, set `FIREBIRD_DISTILL_CONTROL_NATIVE=1` and select
`test_native_8_3_preserves_semantics_and_normalized_teacher_targets` in the same
pinned ACT environment.
