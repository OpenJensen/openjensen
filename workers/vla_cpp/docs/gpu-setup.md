# CUDA worker setup

Run from `workers/vla_cpp` on Linux/WSL2 with Docker, NVIDIA container support
and a CUDA-capable GPU. Mount this worker directory at `/workspace`.

## Prepare inputs and a CPU screen

Prepare these source directories before generating your own input manifest:

| Path under this worker directory | Preparation |
|---|---|
| `artifacts/docker/vendor/vla.cpp` | Clone `https://github.com/VinRobotics/vla.cpp`, check out `52439f7c6c362d7bee218b400b9080cc32d75cc3`, apply `policykit/patches/vla-cpp-smolvla-packed.patch`. |
| `artifacts/cuda/vendor/llama-download` | Extract `https://codeload.github.com/ggml-org/llama.cpp/tar.gz/7ba604f1cb61cd14898138e9abc0b4ff2601f180` with its leading directory removed. Retain the archive. |
| `artifacts/docker/sources/smolvla` | Download `HuggingFaceVLA/smolvla_libero` revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`, including `config.json`, policy weights and both processor JSON/safetensors sets. |
| `artifacts/docker/LIBERO` | Clone `https://github.com/Lifelong-Robot-Learning/LIBERO` at `8f1084e3132a39270c3a13ebe37270a43ece2a01`, including assets and fixed initial states. |
| `artifacts/cuda/modelopt-metadata` | For optional ModelOpt checks, run `scripts/prepare_modelopt_metadata.py` in the ModelOpt image to download pinned tokenizer/config files without base-model weights. |

Follow [CPU preparation](cpu-benchmark.md) and the
[CPU engine screen](quantization-benchmark.md) to generate GGUFs and a new
`results.json` from your own source files. The screen records source/model hashes
and runtime identity; retain the generated manifest with its models and logs.

The CUDA scripts currently read
`artifacts/docker/runs/smolvla-screen-v1/results.json` and its `models/` directory.
Use that name for a fresh CPU screen in a new work area. If it already contains
an active or retained run, preserve it and use a separate work area. Do not copy
another run's measured results or substitute historical hashes for newly
produced files.

Create `artifacts/cuda/setup` and the source/model parent directories. Make
native build outputs writable by the build UID. For archive-only LIBERO source,
provide `artifacts/docker/LIBERO-source.json` with `revision`, `url`,
`archive` (container path) and `archive_sha256`.

## Prepare models

Use GGUF 0.19.0 in the ModelOpt container. The preparation script verifies or
recreates the GGUF files against your CPU screen's manifest:

```bash
docker build -f Dockerfile.modelopt -t jensen-modelopt:local .
docker run --rm -v "$PWD:/workspace" --entrypoint bash jensen-modelopt:local -lc \
  'python -m pip install --no-cache-dir gguf==0.19.0 && python -m scripts.prepare_cuda_models'
docker run --rm -v "$PWD:/workspace" --entrypoint python jensen-modelopt:local \
  scripts/prepare_modelopt_metadata.py
```

## Build and run

Choose a new run name:

```bash
docker build -f Dockerfile.cuda -t jensen-quant-cuda:local .
docker run --gpus device=0 --user "$(id -u):$(id -g)" --cpus=4 --memory=8g \
  -v "$PWD:/workspace" --entrypoint bash jensen-quant-cuda:local scripts/build_cuda.sh
docker run --gpus device=0 --cpus=4 --memory=8g -v "$PWD:/workspace" \
  --entrypoint python3 jensen-quant-cuda:local \
  -m policykit.cuda_bench --run smolvla-cuda-new --reps 20 --rounds 3
```

Expose the GPU during WSL linking so `libcuda.so.1` resolves. To run rollouts,
build `Dockerfile.cuda-sim` and set the explicit input manifest in
`scripts/run_cuda_rollouts.sh` to the selected engine run.

## Prepare optional ModelOpt inputs

Capture observations from your own completed floating-policy rollout. Keep the
capture IDs, NPZ paths, noise seeds and SHA-256 hashes in its `captures` manifest,
paired with the matching checkpoint and processor configuration. Supply that
manifest with `--rollout`, the prepared metadata directory with `--metadata`,
and a new output directory with `--out` when invoking
`python -m policykit.modelopt_pilot`.

The current numerical protocol requires exactly seven captures and
`status: episode_complete`: the first three are calibration observations and
the remaining four are evaluation observations from the same trajectory. Run
`--method reference` before `--method awq` or `--method smoothquant`; compare
matched outputs with identical observations and noise. These numerical checks do
not establish packed-engine performance or closed-loop task quality. Keep
captures, measurements and reports in private local output directories.
