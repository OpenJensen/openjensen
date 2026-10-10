# codywang/so101_pickup_test evaluation intake

The dataset is registered in `configs/evaluation.so101.yaml`, pinned to
`ecef85bc07005f771ad86deeff1427f9d72953ed`. Cache metadata and tabular files
under `artifacts/datasets/so101_pickup_test`, recording SHA-256 hashes in
`download-manifest.json`. Video decoding is separate from metadata preflight.

This is a LeRobot v3 recording dataset (30 episodes, 4,500 frames at 30 FPS),
with six joint state/action channels and one front camera. Its Hub repository
contains data, metadata and videos; it does not ship a simulator environment.

The current quantization benchmark uses `HuggingFaceVLA/smolvla_libero`, with
an eight-channel observation state, seven-channel action and two cameras. The
preflight records these mismatches rather than truncating actions or inventing
an observation mapping. Feature compatibility alone is insufficient: units,
normalization, controller, camera views and task training also need to match.

```bash
uv run python -m policykit.dataset_preflight \
  --profile configs/evaluation.so101.yaml \
  --model-config artifacts/docker/sources/smolvla/config.json \
  --out artifacts/docker/runs/so101-preflight
```

## What can be measured without simulation

With a compatible policy, replay held-out observations through its floating and
quantized variants, fixing prompts, preprocessing and noise. Report model latency,
quantization wall time, per-channel output agreement and error against recorded
actions. Use the checkpoint's saved episode split and training-only statistics;
this dataset declares only a train split, so do not assume any frame is held out.
The separate QLoRA worker has a recipe for this dataset; a recipe does not
establish a trained checkpoint or compatible evaluation policy.

The `policykit.output_fidelity --pairs` interface accepts real recorded targets.
Recorded-action error measures imitation of the demonstrator, not the probability
that an autonomous robot completes the task.

## What is needed for task-success drop

A closed-loop environment must respond to the policy's actions, reset to paired
initial states, and evaluate a success condition. Obtain a compatible SO-101
simulator scene/controller/camera setup or run on the physical robot. Merely
replaying demonstration videos or known successful recorded actions does not
measure the evaluated policy succeeding. A robot URDF alone does not supply the
pickup scene, object dynamics, reset distribution or success predicate.

`task_success_drop_pp` remains null until both floating and quantized policies
have completed comparable trials. The user-supplied dataset is retained as the
evaluation candidate; no unrelated LIBERO score is attributed to SO-101 pickup.

Sources: [dataset](https://huggingface.co/datasets/codywang/so101_pickup_test),
[pinned schema](https://huggingface.co/datasets/codywang/so101_pickup_test/blob/ecef85bc07005f771ad86deeff1427f9d72953ed/meta/info.json),
[LeRobot real robot evaluation](https://huggingface.co/docs/lerobot/il_robots).
