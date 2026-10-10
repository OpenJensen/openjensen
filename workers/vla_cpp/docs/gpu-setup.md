# CUDA worker setup

Run from `workers/vla_cpp` on Linux/WSL2 with Docker, NVIDIA container support
and a CUDA-capable GPU. Mount this worker directory at `/workspace`.

## Prepare input paths

Use the pins in `configs/cuda-inputs.json`:

| Path under this worker directory | Preparation |
|---|---|
| `artifacts/docker/vendor/vla.cpp` | Clone `https://github.com/VinRobotics/vla.cpp`, check out `52439f7c6c362d7bee218b400b9080cc32d75cc3`, apply `policykit/patches/vla-cpp-smolvla-packed.patch`. |
| `artifacts/cuda/vendor/llama-download` | Extract `https://codeload.github.com/ggml-org/llama.cpp/tar.gz/7ba604f1cb61cd14898138e9abc0b4ff2601f180` with its leading directory removed. Retain the archive. |
| `artifacts/docker/sources/smolvla` | Download `HuggingFaceVLA/smolvla_libero` revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`, including `config.json`, policy weights and both processor JSON/safetensors sets. |
| `artifacts/docker/runs/smolvla-screen-v1/results.json` | Copy `configs/cuda-inputs.json` into this path only if it does not already exist; preserve an existing screen manifest. |
| `artifacts/docker/runs/smolvla-screen-v1/models` | Supply GGUFs matching the pinned manifest, or regenerate with the command below; all hashes must match. |
| `artifacts/docker/LIBERO` | Clone `https://github.com/Lifelong-Robot-Learning/LIBERO` at `8f1084e3132a39270c3a13ebe37270a43ece2a01`, including assets and fixed initial states. No training dataset download is needed for this simulator pilot. |
| `artifacts/cuda/modelopt-metadata` | Run `scripts/prepare_modelopt_metadata.py` in the ModelOpt image to download the pinned tokenizer/config files without base-model weights. |

Create `artifacts/cuda/setup` and the source/model parent directories. Make
native build outputs writable by the build UID. For archive-only LIBERO source,
provide `artifacts/docker/LIBERO-source.json` with `revision`, `url`,
`archive` (container path) and `archive_sha256`.

## Prepare models

Use GGUF 0.19.0 in the ModelOpt container:

```bash
docker build -f Dockerfile.modelopt -t firebird-modelopt:20260926 .
docker run --rm -v "$PWD:/workspace" --entrypoint bash firebird-modelopt:20260926 -lc \
  'python -m pip install --no-cache-dir gguf==0.19.0 && python -m scripts.prepare_cuda_models'
docker run --rm -v "$PWD:/workspace" --entrypoint python firebird-modelopt:20260926 \
  scripts/prepare_modelopt_metadata.py
```

## Build and run

Choose a new run name:

```bash
docker build -f Dockerfile.cuda -t firebird-quant-cuda:20260926 .
docker run --gpus device=0 --user "$(id -u):$(id -g)" --cpus=4 --memory=8g \
  -v "$PWD:/workspace" --entrypoint bash firebird-quant-cuda:20260926 scripts/build_cuda.sh
docker run --gpus device=0 --cpus=4 --memory=8g -v "$PWD:/workspace" \
  --entrypoint python3 firebird-quant-cuda:20260926 \
  -m policykit.cuda_bench --run smolvla-cuda-new --reps 20 --rounds 3
```

Expose the GPU during WSL linking so `libcuda.so.1` resolves. To run rollouts,
build `Dockerfile.cuda-sim` and set the explicit input manifest in
`scripts/run_cuda_rollouts.sh` to the selected engine run.

## Prepare ModelOpt captures

Supply the seven frozen observations and `result.json` expected by the pilot at
`artifacts/docker/runs/smolvla-experiments-v4/float_reference/task0-init0-seed42-steps500/`.
Keep each capture's ID, path, noise seed and hash with these files. Use the
pinned checkpoint and capture identities when running the numerical pilot.
