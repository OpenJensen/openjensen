# ACT action-policy distillation

This is a working, isolated **ACT teacher → smaller ACT student** training worker.
It performs native LeRobot gradient updates from detached teacher action chunks,
saves a complete smaller FP32 inference policy, and tests that saved policy in a
fresh process. It is neither weight quantization nor a relabelled scratch trainer.
Only this pair is implemented. SmolVLA and cross-family distillation are unavailable.

The first recipe retains the teacher's learned ResNet18 backbone (copied exactly
and frozen), six-coordinate ordering, single camera, saved pre/postprocessors and
100-action chunk/queue. The student has width256, FF1024, two encoder layers, one
decoder, four attention heads, no VAE, and dropout0. It must be smaller than the
teacher's inference tensors, excluding any teacher VAE. Native ACT's masked L1
loss learns **normalized teacher actions**; the saved normalizer processes the
observations once. Targets are never normalized twice. Demonstration actions are
used only for per-coordinate diagnostics, not secretly blended into training.

The underlying ACT architecture and zero-latent inference behavior are documented
by the [ACT authors](https://tonyzhaozh.github.io/aloha/) and the
[LeRobot ACT guide](https://huggingface.co/docs/lerobot/act). Exact execution uses
the pinned installed implementation, whose file hashes are recorded with results.

## Runtime and invocation

The model process requires Python3.12 and the existing isolated ACT tuple:
LeRobot0.6.1, Torch2.11.0, torchvision0.26.0, safetensors0.8.0. Linux CPU wheel
suffixes are accepted and the actual installed versions are retained. Reuse the
validated `workers/act_optimizer` environment; never install model dependencies
into the application/core environment. The local packages can be installed with
`pip install --no-deps -e workers/act_optimizer -e workers/policy_distillation`,
or exposed through these fixed source roots:

```sh
PYTHONPATH=workers/policy_distillation/src:workers/act_optimizer/src \
  workers/act_optimizer/.venv/bin/python -m firebird_distill.application \
  /absolute/operator-owned/request.json /absolute/new-result.json
```

This milestone runs only on CPU and the public supervisor currently requires
POSIX. It has a1–3600second supervised-work deadline after initial source admission and
owns/cleans its process groups,
including signal delivery during spawn and a child that exits before descendants.
Copying and hash verification check the deadline at execution/publication boundaries;
this is not a hard elapsed-time bound on filesystem I/O. Only local runtime/temp
environment settings pass to children; provider/cloud
credentials do not. HF offline flags and Python socket audit hooks prohibit
network fallback in the model process. These are application controls, **not an OS
sandbox**. The internal `runtime` module is not a public timeout boundary.

A request uses exact fields (paths resolved by the operator, never browser input):

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

The abbreviated `files` example must be replaced by the **complete** flat teacher
inventory: weights, config, processors, statistics and any existing provenance.
The caller supplies the registered outer identity; the worker independently checks
every selected source file and records that caller identity. Teacher/source files
are unchanged. A private verified copy is used; fresh reload is performed after
removing the teacher copy, with Python reads of original teacher/corpus denied.
The output must be new; publication is an atomic no-replace directory rename.

## Real robotics data, explicit splits

The trainer consumes a bounded immutable observation corpus: at most256 samples,
8MiB per sample,2GiB total. Each safetensors record has original uint8 RGB, raw
float32 state `[6]`, recorded float32 action chunk `[100,6]`, and bool padding
`[100]`. Padding must match the actual episode length and frame index. No pickle,
custom model code, remote loader or augmentation runs. The JSON manifest binds
all bytes, native source identity, camera shape, coordinate names/units, teacher
processor identity, episode/frame identity and declared ancestry.

Three explicit, nonempty **train/validation/final** partitions are required.
Episodes, their length, and lineage groups cannot cross partitions. Repeated
identical image/state pairs across partitions also fail. The selected last training
step is fixed in advance; there is no checkpoint search or final-based selection.
Validation is measured before/after training. Final imitation error is measured
only after saving the frozen student. Final data are schema-validated beforehand,
but never supplied to the optimizer or used for an untuned-final comparison.
Teacher training overlap is unknown; these are student-held-out partitions, not
proof that the teacher never saw those episodes.

Use the existing **LeRobot0.6.2 dataset environment**, separately, to prepare a
verified Firebird LeRobot-v3 snapshot. Expose these source packages in that reader:
`workers/policy_distillation/src`, `workers/act_optimizer/src`,
`workers/smolvla_qlora/src`. Invoke
`python -m firebird_distill.prepare PREPARE_REQUEST NEW_RESULT`.
The input has exact fields:

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

Names/order must match the native snapshot. The example units are **not a default**:
the operator must establish that data use the teacher's recorded coordinates.
Matching shapes alone does not do this. Simulator radians cannot be silently
substituted for recorded joint/gripper conventions. Camera resolution is exact;
no hidden resize or renormalization occurs. Preparation uses the existing complete
snapshot/media verifier before/after reading, denies Hub fallback and network,
and checks returned episode/frame identity. Only the explicitly sampled rows are
decoded. This local preparation CLI is a bounded-data reader; unlike the public
training supervisor it has no elapsed-time deadline of its own.

Synthetic snapshot lineage is automatically retained as `generated_fixture` and
requires `compatibility: generated_fixture`. Mixed generated/recorded sources fail.
A generated corpus can verify the algorithm and software; it is not recorded robot
training evidence. The first full native writer→snapshot→reader→training proof uses
three declared generated scene groups,6episodes,24frames and12sampled observations.

## Output and honest acceptance

`RESULT.artifact` is `{path,format:"native_checkpoint",label}`. The directory has
`policy/` with exact smaller native ACT weights/config/processors/statistics,
`training.json`, `verification.json`, `lineage.json` and a core-compatible
`manifest.json` with exact payload hashes. Metadata identifies the parent artifact,
model/recipe and the source dataset kind. Teacher target hashes, masks/partitions,
implementation/runtime source hashes, random seed and optimizer losses are kept.
The fresh process must reproduce all full action-chunk and postprocessed hashes,
including100-action queue/reset behavior, using only the saved student package and
its observation corpus. This establishes a reloadable inference artifact, not an
optimizer/RNG checkpoint: `training_resume_supported=false`.

The student is already `use_vae=false`; do **not** send it through ACT VAE-removal
export. The API snapshot-to-distillation job and checkpoint download are verified
in [application evidence](evidence/app-integration.json). The [browser integration
proof](evidence/browser-integration.json) also verifies a manually submitted 12-step
local job through the production web UI, successful registration and the exact
browser-downloaded package. It used a generated ACT teacher and three generated
scene groups, not recorded robot skill. All original source hashes were unchanged;
all 10 archive files matched the registered package. The student weights were
55,946,840 bytes; total job elapsed time was 170.426 seconds on macOS CPU. This is
software verification, not a speed benchmark. CUDA, Isaac rollout, timing speedup,
calibration and task quality remain unverified.
`quality_verified`, `calibration_verified`, `speedup_verified` and
`isaac_runtime_verified` remain false; `task_success` is null. Offline teacher
imitation error can improve while robot success worsens. Teacher failure may be
copied. Paired closed-loop teacher/student trials on frozen unseen states and exact
exported packages are required before recommending deployment.

Teacher/data licenses, redistribution permission, unknown teacher training overlap,
recorded-coordinate compatibility and robot calibration remain separate gates.
The local fixture proof does not establish any of them and redistributes no user
weights or recordings. Smaller weights are measured against the already-inference
teacher tensors so VAE removal is not counted as a distillation gain.

## Verification

```sh
PYTHONPATH=workers/policy_distillation/src:workers/act_optimizer/src \
  workers/act_optimizer/.venv/bin/python -m pytest workers/policy_distillation/tests -q
```

Tests exercise actual native teacher/student updates, smaller serialized weights,
strict fresh-process reload, group/episode leakage, padding, finite data, strict JSON,
input mutation, atomic no-overwrite, source preservation and real process cleanup.
The genuine native-v3 preparation proof runs separately in the dataset environment;
its generated captures and artifact evidence are retained outside Git.
