# Runtime and support matrix

OPEN JENSEN is a self-hosted, single-user research workspace. Start with the [source installation and dataset preview](getting-started.md). Training, inference and simulation use separate workers with their own Python versions, dependencies, models and hardware.

Project code uses [Apache License 2.0](../LICENSE). Upstream code, model weights,
datasets and assets retain their own licenses/access terms; a listed workflow does
not grant redistribution rights for those inputs.

## What availability means

**Configured** means the application has the runtime fields needed to admit that operation. A model marked **ready** still needs a compatible installed worker, available hardware, correct inputs and any required model access. It does not mean every model/method/device combination has completed training or evaluation.

**Execution evidence** applies to the exact source, runtime, input and model identities in its receipt. Configuration parsing, dependency resolution, generated fixtures, real optimizer updates, fresh reload and closed-loop task quality are separate checks. **Experimental** workflows retain incomplete quality, calibration, performance or installation acceptance.

| Workflow | Prerequisites and platform | Evidence and limits |
| --- | --- | --- |
| Web workspace, API and CLI | Python 3.14.7; Node 24.21.0/pnpm 12.6.0 for the static build. Linux is the default CI platform. | Software, API and browser checks run in CI. This is a source installation; a Python wheel alone does not contain the complete web workspace. |
| Public dataset inspection/preview | Network and the isolated CPU reader. No GPU or provider key. | LeRobot v2/v3 metadata and sampled-data/video workflows; public access and media availability depend on the source repository. |
| Local imports and training copies | Complete CPU reader requirements; FFmpeg/ffprobe for conversion/validation. Immutable-copy preparation uses Linux/macOS. | Supported upload layouts are listed in [dataset import](dataset-import.md). Generated examples demonstrate mechanics, not robot skill or dataset quality. |
| SmolVLA LoRA/QLoRA | Isolated Python 3.11/CUDA environment; a compatible NVIDIA GPU and pinned model/dataset. Automatic local setup targets Linux x86_64. | Historical L4 training/checkpoint/reload evidence exists for SmolVLA LoRA. Other methods, GPUs, inputs and revisions require their own execution checks. |
| Native ACT and other registered training families | Pinned Python 3.12/LeRobot 0.6.2 worker with sufficient CUDA memory and matching cameras/dimensions. | ACT has historical real GPU training/reload evidence. Configuration/dependency checks for the other families do not establish training acceptance. See [native training](native-training.md). |
| ACT export, distillation, packed INT8/INT4 and observation replay | Separate pinned CPU model/dataset environments; [CPU setup](../workers/local_cpu/README.md) covers Apple Silicon macOS and Linux x86_64. | Real CPU export/reload and generated-input student/packing/replay evidence exists. Task quality, physical calibration, GPU memory savings and inference speedups remain unverified. |
| SmolVLA GGUF quantization and native inference | Pinned patched native runtime and complete source model; [worker setup](../workers/vla_cpp/README.md). | Recorded Linux CPU/GPU execution and paired LIBERO results apply to their exact packages. Q4 is experimental; storage/latency results do not guarantee retained task quality. |
| LIBERO Spatial evaluation | Complete pinned policy, fixtures, processors, normalization and simulator assets; compatible configured Linux runtime. | Recorded paired episodes are limited samples. Reusing those episodes does not create new independent quality evidence. |
| Isaac simulation and Teaching | Operator-configured Linux NVIDIA/Isaac 6.1 runtime, scene and policy/control contracts; optional services configured separately. | Experimental. A completed video or recorded replay does not prove task success, accepted SO101 calibration or physical robot deployment. |
| Desktop client | Tauri/Rust platform prerequisites plus a separately running application. | Connector mode is available from source. Bundled Python is an opt-in macOS ARM64 experiment; current-product installation, signing and upgrade acceptance remain open. |
| Windows | Source instructions include Windows reader/CLI paths; CI Windows lanes are a manual opt-in. | Native Windows worker, self-contained desktop and fresh-machine installation support have not been established. |

## Read the evidence at its scope

The repository retains [native configuration checks](native-config-validation.json), [dependency resolution](native-dependency-resolution.json), [ACT CPU export proof](../workers/act_optimizer/evidence/native-application-parity-linux.json), [generated student/INT8/replay proof](../workers/policy_distillation/evidence/student-pipeline.json), and the [dedicated L4 comparison](../workers/benchmark_gpu/evidence/2026-09-26/dedicated-l4/REPORT.md). These are historical receipts. Their recorded source, input and environment identities remain authoritative; renaming current commands does not extend their acceptance to a new release.

For a deployment, record the actual version, recipe, model/data revisions, runtime and output hashes. Review saved job events and terminal output before treating a result as accepted. Held-out imitation loss, generated-input numerical drift and complete robot-task success answer different questions.

## Choose the next setup

- [CPU reader](../workers/_cpu_readers/README.md) for inspection/import.
- [Compute settings](compute-settings.md) for local NVIDIA or Google Cloud training.
- [Policy workflow](policy-workflow.md) for explicit native runtime configuration.
- [Desktop connector](../apps/desktop/README.md) for the optional client.
- [Backup and alpha upgrades](upgrading.md) before changing an existing installation.
