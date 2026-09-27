# Training temporal contract

New training requests can place the following controls inside `training`:

```json
{"prediction_horizon": 8, "execution_horizon": 3, "observation_history": 1, "frame_stride": 1}
```

For ACT and SmolVLA, prediction maps to native `chunk_size` and execution maps to
`n_action_steps`. Explicit values must be integers from 1 to 1024, with execution
no greater than prediction. Prediction defaults to 100 for ACT and 50 for
SmolVLA; omitted execution defaults to the resolved prediction horizon. Only one
current observation and stride 1 are admitted. This does not enable history or
resampling. API admission and worker admission use identical dependency-free rules.

| Family | New independent controls | Existing `chunk_size` |
| --- | --- | --- |
| ACT | Supported mapping, actual generated CPU test | Preserved, coupled prediction/execution |
| SmolVLA | Supported mapping from pinned 0.4.4; GPU execution not revalidated here | Preserved, coupled prediction/execution |
| Diffusion, MultiTaskDiT, VQBET | Rejected: different native horizons/token semantics | Rejected for new jobs; previously silently ignored |
| Psi-Zero | Rejected; fixed native 30/30 and one observation | Only 30 accepted |
| Other registered native policies | Rejected pending family-specific proof | Existing coupled behavior preserved |

Do not combine legacy `chunk_size` with the new prediction/execution fields.
Legacy recipes load without rewriting their files. Exact resume still uses the
saved recipe and policy configuration; no architecture override is applied.
Historical native checkpoints with previously ignored `chunk_size` can resume
using their original `train_config.json`. Smol reload refuses a policy config
whose temporal dimensions disagree with its saved recipe. ACT's learned decoder
position embedding is sized at construction; a different prediction horizon is
not a harmless checkpoint reload override.

Workers write `temporal-contract.json` from the actual resolved policy config,
dataset FPS, native delta indices and timestamps. Both train and validation
loader timestamps are checked. The record is included in each new hashed
checkpoint, compared on resume when present, and verified against reloaded
configuration before inclusion in the artifact metadata and result report.
Older checkpoints without the record remain readable; resumed output gains a
record without changing the source checkpoint. FPS is read from pinned dataset
metadata; it is not a user override or a claim about wall-clock control frequency.

## Verification boundary

The opt-in `tests/test_temporal_native.py` runs the existing pinned LeRobot
0.6.2 CPU stack offline with generated 32px observations and two five-frame
episodes. It exercises real native dataset padding without crossing episodes,
ACT loss and padding gradients, 8-step predictions / 3-step execution queue,
strict shape mismatch refusal and a fresh interpreter checkpoint reload.
It does not run the CUDA trainer, SmolVLA weights, cloud jobs, a robot or a task
quality evaluation. Pure/API tests also prove rejected requests persist no job
and start no worker, and cover Smol loader mapping plus exact resume guards.

Upstream contracts: native LeRobot `e595b7902714ba51f91e47523f66f89c5181b649`
ACT config/model; [SmolVLA 0.4.4 configuration](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/smolvla/configuration_smolvla.py)
and [queue/loss implementation](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/smolvla/modeling_smolvla.py).
ACT changed-horizon native optimizer/resume, processor-bound inference export,
packing, fresh HTTP and observation replay now have a reproducible generated
[CPU fixture workflow](../act_optimizer/README.md#changed-horizon-software-acceptance).
Full TRAIN-006 remains open for genuine SmolVLA training/resume/export/serving,
GPU/simulator acceptance and any future history/stride controls. No recorded-data
quality acceptance follows from generated CPU fixtures.
