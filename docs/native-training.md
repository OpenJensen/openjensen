# Native policy training

SmolVLA keeps its dedicated pinned LeRobot 0.4.4 LoRA/QLoRA worker. The native
LeRobot bridge uses a separate Python 3.12 environment and upstream revision
`e595b7902714ba51f91e47523f66f89c5181b649` (package version 0.6.2). It does not
replace or upgrade the SmolVLA environment.

The OPEN JENSEN cloud catalog offers these policy families:
ACT, Diffusion Policy, EO-1, EVO-1, GR00T N1.7, Multi-Task DiT, π₀, π₀-FAST,
π₀.₅, SmolVLA, VLA-JEPA, VQ-BeT, WALL-X (WALL-OSS), XVLA and Psi-Zero.
OpenVLA choices are available for operator-installed adapters; they are not
advertised as bundled trainers.

ACT, Diffusion, Multi-Task DiT and VQ-BeT initialize a new policy using the pinned
upstream implementation. Their `code://` model source identifies that fact.
Other models have immutable Hugging Face weight revisions. Native training uses
each policy's own preprocessing, temporal observations, loss, optimizer groups
and checkpoint save/load APIs. “Native training” includes the selected upstream
freezing recipe: EVO-1 trains stage 1, π₀/π₀.₅ train their action experts, and
XVLA trains soft prompts. It does not mean every base-model weight is unfrozen.

Model weights and datasets are fetched on the GCP worker. Checkpoints use
[cloud artifact storage](cloud-artifact-storage.md). The native bridge publishes
structured step/loss telemetry, held-out imitation loss and complete checkpoints
with optimizer state. It saves the resolved upstream configuration and episode
split, then requires a fresh-process deterministic action reload check before
reporting training success. Numerical reload and imitation loss do not establish
robot task success.

Model constraints are checked before allocation where metadata permits: GPU
memory floors, registered training method and camera count. VQ-BeT expects one
camera; the published VLA-JEPA world model expects two. Psi-Zero uses its separate
upstream worker and currently accepts LeRobot v2.1 data with one camera. The
native LeRobot bridge uses v3 datasets. ACT and SmolVLA support gradient
accumulation on compatible workers; other adapters require
`gradient_accumulation_steps=1`. Progress counts completed optimizer updates.

## Robot dimensions

The pinned upstream policy factory replaces output features with the selected
dataset's action dimensions before loading weights. Its training entrypoint also
replaces normalization features and statistics on a new fine-tune. Fixed-width
foundation-model heads pad and crop to their checkpoint limits: π and EO-1 use
32 dimensions, EVO-1 uses 24, WALL-X and the selected XVLA checkpoint use 20,
and GR00T uses 132. The API rejects inputs beyond these limits before allocating
compute.

VLA-JEPA derives action/state dimensions from the dataset and explicitly permits
reinitializing only its action encoder, action decoder and state encoder when
their shapes differ. Its source checkpoint's gripper snapping/binarization is
disabled in the general training profile so a LIBERO gripper index is not imposed
on a different robot. World-model camera-count constraints remain explicit.

## GPU memory labels

Model cards show a **GPU budget**, not a checkpoint download size or a promise of
peak usage. Native/Psi budgets come from the configured training profiles. SmolVLA
suggests 16 GB, consistent with the upstream [LeRobot hardware guidance](https://github.com/huggingface/lerobot/blob/main/AGENT_GUIDE.md#6-which-policy-should-i-train).
This suggestion does not raise the adapter's admission floor; smaller recipes
still need to meet the configured worker's requirements. Batch size, cameras,
precision and trainable layers change memory use. Models without an installed
adapter show that their GPU budget is unverified.

Completed native ACT checkpoints have an explicit [local CPU inference export](cloud-act-export.md) path from the Fine-tune checkpoint panel.
