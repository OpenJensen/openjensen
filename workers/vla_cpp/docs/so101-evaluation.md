# codywang/so101_pickup_test evaluation intake

The dataset is registered in `configs/evaluation.so101.yaml`, pinned to
`ecef85bc07005f771ad86deeff1427f9d72953ed`. Metadata and tabular files are cached
under `artifacts/datasets/so101_pickup_test`, with SHA-256 hashes in
`download-manifest.json`. Videos are not yet downloaded or decoded.

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
The existing QLoRA training recipe in the separate training checkout targets this
dataset, but the presence of a recipe does not establish a trained checkpoint.

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

## Public model discovery

The author's public Hub profile lists five model repositories and no public
Spaces as inspected. The stored config/train_config snapshots are under
`artifacts/datasets/so101_pickup_test/model-discovery`.

- `act_makermods_pick_dishes_and_stack_2` uses an ACT policy trained on
  `ArjunPrasaath/makermods_pick_dishes_and_stack_2`.
- `makermods_pour_drinks` uses ACT trained on `dhyuti-n/makermods_pour_drinks`.
- `cuponcup-act`, `cuponcup-pi05`, and `cuponcup-smolvla` were trained on
  `codywang/cuponcup_20260625_163207`.

None of the five published training configurations names `so101_pickup_test`.
All five have `env: null`; the listings do not establish a matching public
simulation environment.

The [cup-on-cup SmolVLA config](https://huggingface.co/codywang/cuponcup-smolvla/blob/eca17ca01f54abb0f53ed5122ca61ed5c10fc2d3/config.json)
uses six-dimensional state/actions and one `observation.images.front` view, so it
is a closer structural candidate than the current LIBERO model. Its [training
configuration](https://huggingface.co/codywang/cuponcup-smolvla/blob/eca17ca01f54abb0f53ed5122ca61ed5c10fc2d3/train_config.json)
identifies a different dataset/task. Matching tensor dimensions do not prove
camera/controller/action-unit compatibility or pickup ability. It is recorded
as a discovered candidate, not silently used to claim pickup task accuracy.
