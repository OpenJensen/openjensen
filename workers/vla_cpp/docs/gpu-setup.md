# Experimental RTX 3070 setup

Run these commands from `workers/vla_cpp` on Linux/WSL2 with Docker and NVIDIA
container support. Keep each experiment in its own working directory. The CUDA
images are optional research environments; installing the conversion worker does
not install ModelOpt, PyTorch, a simulator or a scheduler.

The existing `policykit-worker` contract remains conversion-only. Adding these
Python modules changes its code identity: resolve a new runtime/recipe identity
before submitting a conversion job. Do not reuse an identity from a different source revision.

## Required inputs

The scripts intentionally use the experiment's fixed relative artifact layout.
Weights, vendor repositories, calibration images, logs and videos are ignored by
Git. `configs/cuda-inputs.json` contains the small pinned source/candidate manifest;
it is an expected-byte contract, not evidence of a successful run on a new host.

| Path under this worker directory | Preparation |
|---|---|
| `artifacts/docker/vendor/vla.cpp` | Clone `https://github.com/VinRobotics/vla.cpp`, check out `52439f7c6c362d7bee218b400b9080cc32d75cc3`, apply `policykit/patches/vla-cpp-smolvla-packed.patch`. |
| `artifacts/cuda/vendor/llama-download` | Extract `https://codeload.github.com/ggml-org/llama.cpp/tar.gz/7ba604f1cb61cd14898138e9abc0b4ff2601f180` with its leading directory removed. Retain the archive. |
| `artifacts/docker/sources/smolvla` | Download `HuggingFaceVLA/smolvla_libero` revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`, including `config.json`, policy weights and both processor JSON/safetensors sets. |
| `artifacts/docker/runs/smolvla-screen-v1/results.json` | Copy `configs/cuda-inputs.json` into this path only if it does not already exist; preserve an existing screen manifest. |
| `artifacts/docker/runs/smolvla-screen-v1/models` | Restore the five recorded GGUFs or regenerate with the command below; all hashes must match the manifest. |
| `artifacts/docker/LIBERO` | Clone `https://github.com/Lifelong-Robot-Learning/LIBERO` at `8f1084e3132a39270c3a13ebe37270a43ece2a01`, including assets and fixed initial states. No training dataset download is needed for this simulator pilot. |
| `artifacts/cuda/modelopt-metadata` | Run `scripts/prepare_modelopt_metadata.py` in the ModelOpt image to download the pinned tokenizer/config files without base-model weights. |

Create `artifacts/cuda/setup` and the source/model parent directories first. Native
build outputs must be writable by the UID used for the build. For an archive-only
LIBERO checkout, `artifacts/docker/LIBERO-source.json` must contain `revision`,
`url`, `archive` (container-visible path) and `archive_sha256`; the rollout verifies
that archive hash. A Git checkout records its actual commit instead.

Build `Dockerfile.modelopt`, then regenerate the GGUFs in a disposable preparation
container. Its GGUF version matches the recorded experiment; the ordinary worker
lockfile keeps its own conversion dependency pins.

```bash
docker build -f Dockerfile.modelopt -t firebird-modelopt:20260926 .
docker run --rm -v "$PWD:/workspace" --entrypoint bash firebird-modelopt:20260926 -lc \
  'python -m pip install --no-cache-dir gguf==0.19.0 && python -m scripts.prepare_cuda_models'
docker run --rm -v "$PWD:/workspace" --entrypoint python firebird-modelopt:20260926 \
  scripts/prepare_modelopt_metadata.py
```

The converter hashes the pinned source and each completed candidate before
publishing it. It rejects conflicting existing artifacts. Follow the native build,
engine and rollout commands in the [RTX 3070 report](quantization-rtx3070.md).
Mount the worker directory, not the repository root, at `/workspace`.

## NVIDIA calibration inputs

The recorded AWQ/SmoothQuant pilot needs the seven frozen observations and
`result.json` from the completed development rollout at
`artifacts/docker/runs/smolvla-experiments-v4/float_reference/task0-init0-seed42-steps500/`.
The result lists each capture's ID, path, noise seed and hash. Restore those exact
files from the original evidence workspace to reproduce the reported diagnostic.
They are not included in this repository. The pilot refuses a different checkpoint,
capture count, capture hash, scope or nonfinite output.

New calibration data must be captured and assigned a separate experiment identity;
do not relabel new trajectories as this historical run. The three calibration and
four comparison frames share one trajectory. Broader work must split by episode
and use an independent final evaluation set as specified in the
[NVIDIA plan](nvidia-quantization-plan.md).

## Evidence and import boundary

`GPU_ORIGIN.json` maps the precursor scripts into this worker. The
[evidence manifest](quantization-rtx3070-evidence.json) retains original source,
model, binary and image hashes; those source paths refer to the historical
experiment snapshot. They do not establish GPU execution with the current source
revision. The [source import validation](gpu-branch-validation.md) records the
checks and their exact scope.

The optional ModelOpt CUDA extension fell back in the recorded runtime image,
and the full-policy exporter failed at the LeRobot configuration boundary. Neither
AWQ nor SmoothQuant has a verified packed deployment engine here. AutoQuantize is
still gated on a fixed-recipe export/reload; native NVFP4 requires a Blackwell lane.
