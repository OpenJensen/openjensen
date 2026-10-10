# Run SmolVLA GPU comparisons

Run on Linux with an NVIDIA GPU. Use Python 3.11 for native/vLLM and a separate
Python 3.12 environment for TensorRT-LLM.

## Install native dependencies

On Ubuntu, install build tools, EGL and OpenMPI:

```sh
sudo apt-get update
sudo apt-get install -y build-essential libegl1 libgl1 libgl1-mesa-dev libosmesa6-dev libopenmpi-dev
```

Install the NVIDIA graphics/EGL package matching the installed driver, then
verify headless rendering. From `workers/benchmark_gpu`:

```sh
export GPU_WORKER_ROOT="$PWD"
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python cmake==3.31.6
PATH="$PWD/.venv/bin:$PATH" uv pip install --python .venv/bin/python \
  --index-strategy unsafe-best-match -r requirements-native.txt
```

## Prepare snapshots and runtime files

Cache these snapshots:

| Resource | Revision |
|---|---|
| `lerobot/smolvla_libero` | `31d453f7edd78c839a8bbc39744a292686daf0de` |
| `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |
| dataset `lerobot/libero-assets` | `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6` |

Use weights with SHA256
`9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`.
Set `LIBERO_CONFIG_PATH` to a directory containing `config.yaml` with absolute
`assets`, `bddl_files`, `benchmark_root`, `datasets` and `init_states` paths.
Create it beneath the chosen run root using the native interpreter:

```sh
.venv/bin/python scripts/prepare_libero_assets.py --config-dir /absolute/run-root/libero-config
```

For C++ candidates, prepare vla.cpp commit
`52439f7c6c362d7bee218b400b9080cc32d75cc3` with the
[packed-loader patch](../vla_cpp/patches/README.md), CUDA and serving enabled.
The run root needs `vla.cpp/build-cuda/vla-server`,
`vla.cpp/src/serving/vla.proto` and
`gguf/smolvla-{bf16,Q8_0,Q4_0,Q8_0-vision}.gguf`. Set the derived converter config's
state dimension to eight, keeping the original checkpoint and both config hashes.

## Capture and run

Switch to the run root and select the worker's scripts and native interpreter.
Use a new output directory for each command and retain the same fixture
directory for comparisons:

```sh
cd /absolute/run-root
export GPU_RUN_ROOT="$PWD"
export SCRIPTS="$GPU_WORKER_ROOT/scripts"
export PYTHON="$GPU_WORKER_ROOT/.venv/bin/python"
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH="$PWD/libero-config"
"$PYTHON" "$SCRIPTS/smolvla_cross_stack.py" \
  --backend native-bf16 --output outputs/capture --fixtures fixtures \
  --quality --capture --task-ids 0 1 2 3 4 --episodes 1
"$PYTHON" "$SCRIPTS/run_cross_stack_matrix.py" \
  --output outputs/latency --fixtures fixtures --samples 5
"$PYTHON" "$SCRIPTS/smolvla_cross_stack.py" \
  --backend cpp-Q8_0 --output outputs/quality-cpp-Q8_0 --fixtures fixtures \
  --quality --task-ids 0 1 2 3 4 5 6 7 8 9 --episodes 2
```

Run each GPU separately. Measure process startup with:

```sh
"$PYTHON" "$SCRIPTS/measure_startup.py" \
  --backend native-bf16 --root "$PWD" \
  --fixture fixtures/task0-init0-chunk0.npz --output outputs/startup-native-bf16
```

## Configure custom engines

Return to the worker directory to install each custom engine. For vLLM:

```sh
cd "$GPU_WORKER_ROOT"
uv venv --python 3.11 .venv-vllm
uv pip install --python .venv-vllm/bin/python cmake==3.31.6
PATH="$PWD/.venv-vllm/bin:$PATH" uv pip install --python .venv-vllm/bin/python \
  -r requirements-vllm.txt
.venv-vllm/bin/python scripts/patch_vllm_pooling.py
cd "$GPU_RUN_ROOT"
export VLLM_USE_V1=0 PYTHONPATH="$SCRIPTS"
"$GPU_WORKER_ROOT/.venv-vllm/bin/python" "$SCRIPTS/probe_vllm_smolvla.py" \
  --output outputs/probe-vllm --fixtures fixtures
```

For TensorRT-LLM, use the system OpenMPI installed above and a separate Python
3.12 environment:

```sh
cd "$GPU_WORKER_ROOT"
uv venv --python 3.12 .venv-trtllm
uv pip install --python .venv-trtllm/bin/python cmake==3.31.6
PATH="$PWD/.venv-trtllm/bin:$PATH" uv pip install --python .venv-trtllm/bin/python --index-strategy unsafe-best-match \
  --override requirements-trtllm-overrides.txt -r requirements-trtllm.txt
cd "$GPU_RUN_ROOT"
"$GPU_WORKER_ROOT/.venv-trtllm/bin/python" "$SCRIPTS/probe_trtllm_smolvla.py" \
  --output outputs/probe-trtllm --fixtures fixtures
```

For cross-stack runs, select the custom engine's interpreter and use
`--backend vllm-bf16` or `--backend trtllm-fp16`. When using portable Python,
add its `LIBDIR` to `LD_LIBRARY_PATH` before running the TensorRT-LLM probe.

## Run tests

```sh
cd "$GPU_WORKER_ROOT"
.venv/bin/python -m pytest -q tests
```

Store action arrays, input hashes, logs and dependency versions beneath the run
output directory. Use `compare_results.py` to compare saved matched outputs.
