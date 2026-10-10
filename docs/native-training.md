# Set up native policy training

## Prepare the worker

Install the [training worker](../workers/smolvla_qlora/README.md) in its own environment. Use the pinned LeRobot 0.4.4/Python 3.11 environment for SmolVLA, Python 3.12 with upstream revision `e595b7902714ba51f91e47523f66f89c5181b649` (LeRobot 0.6.2) for native adapters, and the separate [Psi-Zero worker](../workers/psi0/README.md) for Psi-Zero.

Register the installed local worker or [connect a cloud runtime](compute-settings.md). Match the selected model's GPU-memory and camera requirements in the catalog. Save Hugging Face access and accept any gated model terms before starting.

## Prepare the dataset and recipe

Use LeRobot v3 for native adapters and LeRobot v2.1 with one camera for Psi-Zero. Match camera names to the selected policy; use one camera for VQ-BeT and two for VLA-JEPA.

Match action/state dimensions to the selected policy profile: π and EO-1 use up to 32, EVO-1 up to 24, WALL-X/XVLA up to 20, and GR00T up to 132. For VLA-JEPA, select the dataset's action/state dimensions when initializing its encoders/decoder.

Use `gradient_accumulation_steps=1` for adapters other than ACT/SmolVLA. For ACT/SmolVLA, select a runtime listed for accumulation support. Choose batch size and camera resolution to fit the GPU budget shown on the model card.

## Start and follow

1. Inspect the dataset and open **Fine-tune**.
2. Select data, cameras, model, method and runtime.
3. Review the recipe and start.
4. Open the run for optimizer progress, loss, events and checkpoints.
5. Download a checkpoint, resume from a committed checkpoint, or [export ACT for CPU inference](cloud-act-export.md).

For commands and runtime configuration, follow [policy setup](policy-workflow.md). For cloud output locations, see [checkpoint storage](cloud-artifact-storage.md).
