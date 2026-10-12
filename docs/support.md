# Requirements and platform support

OPEN JENSEN is a self-hosted, single-user robotics research workspace. Start with
[source installation and dataset preview](getting-started.md), which needs no GPU
or provider account. Training, inference and simulation use separate worker
environments.

| Workflow | Requirements | Current limits |
| --- | --- | --- |
| Web workspace, API and CLI | Python 3.14.7; Node 24.21.0 and pnpm 12.6.0 to build the frontend. | Install from source and build the web export. The Python wheel alone does not include the complete workspace interface. |
| Public dataset inspection and preview | Network access and the [CPU reader](../workers/_cpu_readers/README.md). No GPU or provider key. | Supports LeRobot v2/v3. Source permissions and available camera recordings depend on the dataset repository. |
| Local imports and training copies | Complete CPU reader requirements; FFmpeg/ffprobe for video conversion. Linux or macOS for immutable training copies. | Use the layouts listed in [dataset import](dataset-import.md). |
| SmolVLA LoRA/QLoRA training | Separate Python 3.11/CUDA worker, NVIDIA GPU, model weights and compatible dataset. | Automatic local setup targets Linux x86_64. GPU memory requirements depend on the model, method and batch size. See [compute settings](compute-settings.md). |
| ACT and other registered training policies | Separate Python 3.12/LeRobot 0.6.2 worker, sufficient CUDA memory and matching camera/state/action dimensions. | Native adapter training is experimental. Available methods depend on the selected policy and worker. See [native training](native-training.md). |
| ACT export, distillation, INT8/INT4 packing and observation replay | Separate CPU model and dataset environments; [CPU setup](../workers/local_cpu/README.md) covers Apple Silicon macOS and Linux x86_64. | Distillation and packing are experimental. Packed weights reduce storage; inference uses standard Torch operations. Evaluate the resulting policy before using it for a task. |
| SmolVLA GGUF quantization and native inference | Patched native runtime and complete source model; [worker setup](../workers/vla_cpp/README.md). | Q4 is experimental. Quantization can change policy behavior, and a smaller file does not guarantee faster inference. |
| LIBERO Spatial evaluation | Compatible Linux runtime, complete policy, processors, normalization and simulator assets. | Prepare the task and initial-state settings described in [LIBERO setup](spatial-workflow.md). Results depend on that setup. |
| Isaac simulation and Teaching | Linux NVIDIA/Isaac 6.1 runtime, scene and policy/control configuration; optional voice services installed separately. | Experimental. Physical robot deployment is outside this setup. See [simulation setup](native-simulation.md). |
| Desktop client | Rust/Tauri platform prerequisites and a separately running application; [desktop setup](../apps/desktop/README.md). | The source client connects to the local server. Bundled Python is an opt-in macOS ARM64 experiment; self-contained release installers and automatic updates are not provided. |
| Windows | Source instructions include Windows CPU reader and CLI paths. | Source setup is experimental. Use Linux for NVIDIA training and Isaac workers. |

A model shown as **ready** still needs the selected worker's dependencies, model
access, hardware and compatible inputs. Review the proposed training or evaluation
settings and selected compute target before starting a job.

Use a persistent workspace directory and follow [backup and upgrade](upgrading.md)
before changing an installation. For errors, start with [troubleshooting](getting-started.md#troubleshooting)
and report reproducible problems through [GitHub issues](https://github.com/OpenJensen/openjensen/issues).
Include the application version, OS, steps and redacted error message.

Project code uses [Apache License 2.0](../LICENSE). Third-party code, model weights,
datasets and assets keep their own licenses and access terms.
