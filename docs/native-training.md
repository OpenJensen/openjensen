# Train robot policies

Fine-tune SmolVLA, or train ACT, another native LeRobot policy or Psi-Zero, on a dataset in your project. Each run saves checkpoints you can download, resume, export or compress.

## What you need

Each model family trains in its own worker environment, on a CUDA GPU on this machine or on Google Cloud. Pick a model and compute from the [support guide](support.md).

| Models | Worker environment |
| --- | --- |
| SmolVLA, LoRA or QLoRA | The [SmolVLA worker](../workers/smolvla_qlora/README.md): Python 3.11 with LeRobot 0.4.4. `uv run --frozen jensen install training` installs it in `workers/smolvla_qlora/.venv`. |
| SmolVLA full, ACT and the other native LeRobot models | Python 3.12 with LeRobot 0.6.2 at upstream revision `e595b7902714ba51f91e47523f66f89c5181b649`. |
| Psi-Zero | The separate [Psi-Zero worker](../workers/psi0/README.md). |

You also need:

- Enough GPU memory, and a dataset with the format, cameras and dimensions the model accepts. See [model requirements](#model-requirements).
- A Hugging Face token in **Settings → Compute** if the model needs one. Accept any gated model's terms before you start.

## Register the compute

- **This machine:** in **Settings → Compute → Local runs**, select **Check this machine → Add worker**, then **Enable local runs** and **Save local settings**. This registers the SmolVLA worker. For native LeRobot models, register a runtime with `"training_module": "jensen_vla.lerobot_application"` and their IDs in `training_model_ids` (see [custom training adapters](policy-workflow.md#custom-training-adapters)).
- **Google Cloud:** [connect it](compute-settings.md), then choose a cloud GPU in **Fine-tune**. Cloud runtimes accept every catalog model. See [cloud training](skypilot-training.md).

## Train

1. Inspect a dataset and select **Train on this dataset**, or open **Fine-tune**.
2. **Dataset:** choose the dataset and its training cameras. SmolVLA and native LeRobot models can combine up to eight datasets; Psi-Zero trains on one.
3. **Model:** choose the model and training method. Each card shows its GPU budget.
4. **Compute:** choose the worker or cloud GPU, review **Your training recipe** and select **Start fine-tuning**.
5. Open the run to follow optimizer progress, training and validation loss, activity and checkpoints.
6. From a saved checkpoint, download it, select **Resume from checkpoint**, or [export ACT for CPU inference](cloud-act-export.md).

Each checkpoint keeps its recipe, so a resume uses the same model, dataset, cameras and settings. Use **Download run configuration** to keep your own copy.

## Match the dataset to the model

- **Format:** native LeRobot models need LeRobot v3. Psi-Zero needs LeRobot v2.1.
- **Local datasets:** train only native LeRobot models (including SmolVLA full), from a verified [immutable training copy](local-training.md).
- **Cameras and dimensions:** stay within the [model requirements](#model-requirements). VLA-JEPA reinitializes its action encoder, action decoder and state encoder for the dataset's action and state dimensions.
- **Gradient accumulation:** only ACT and SmolVLA accept values above 1, and only on a runtime the app lists for accumulation support. Every other model needs `gradient_accumulation_steps=1`; **Advanced settings → Gradient accumulation** stays at 1 when unsupported.
- **Batch size:** choose a batch size and camera resolution that fit the GPU budget on the model card.
- **Action timing:** only ACT and SmolVLA take separate prediction and execution horizons (1–1024, execution no larger than prediction). Other models use their native timing; Psi-Zero uses a 30-step chunk.

## Model requirements

| Model | ID | Minimum GPU memory | Cameras | Maximum state/action dimensions |
| --- | --- | --- | --- | --- |
| SmolVLA, LoRA or QLoRA | `smolvla` | 16 GB suggested | — | 32 |
| SmolVLA, full | `smolvla` | 24 GB | — | 32 |
| ACT | `act` | 16 GB | — | — |
| Diffusion Policy | `diffusion` | 16 GB | — | — |
| VQ-BeT | `vqbet` | 16 GB | 1 | — |
| Multi-Task DiT | `multi_task_dit` | 24 GB | — | — |
| EVO-1 | `evo1` | 24 GB | Up to 3 | 24 |
| XVLA | `xvla` | 24 GB | — | 20 |
| WALL-X | `wall_x` | 40 GB | — | 20 |
| EO-1 | `eo1` | 40 GB | — | 32 |
| π₀, π₀-FAST, π₀.₅ | `pi0`, `pi0_fast`, `pi05` | 40 GB | — | 32 |
| GR00T N1.7 | `gr00t_n17` | 40 GB | — | 132 |
| VLA-JEPA | `vla_jepa` | 40 GB | 2 | — |
| Psi-Zero | `psi0` | 40 GB | 1 | 1–36 |

A dash means the model sets no fixed camera count or dimension limit. ACT, Diffusion Policy, VQ-BeT and Multi-Task DiT train from scratch; the others start from pinned pretrained weights. OpenVLA and OpenVLA-OFT appear in the catalog but need a [custom training adapter](policy-workflow.md#custom-training-adapters).

For commands and runtime configuration, see [policy setup](policy-workflow.md). For where cloud checkpoints are stored, see [checkpoint storage](cloud-artifact-storage.md).
