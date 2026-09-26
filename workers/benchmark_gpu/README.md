# SmolVLA GPU comparison

A standalone benchmark worker for complete observation-to-action inference on
RTX 3070 and L4. It compares native LeRobot BF16/FP16, LM-only bitsandbytes INT8/NF4,
and patched vla.cpp BF16/Q8_0/Q4_0 with optional vision Q8_0. Experimental custom
vLLM and TensorRT-LLM adapters execute the full policy using native PyTorch
kernels inside those engines. They do not establish optimized engine-kernel or
quantized-engine support.

This worker is independent of the application workspace and training worker.
Use Python 3.11 for native/vLLM and a separate Python 3.12 environment for
TensorRT-LLM. Do not install these dependency sets together. No model weights,
private connection configuration, or credentials belong in this directory.

## Measurement contract

- Input: identical raw CPU RGB observations, state, task text, and explicit FP32
  diffusion noise. Output: a complete unnormalized CPU `1 x 50 x 7` action chunk.
- Timing includes checkpoint preprocessing, CPU/GPU copies, all ten denoising
  steps, postprocessing, and C++ RPC or custom engine dispatch where used. It
  excludes simulator stepping and camera acquisition.
- Startup is cached runtime initialization timed inside the harness process. It
  includes model/processor setup, quantization, C++ server launch and engine-worker
  initialization where applicable. Parent-process backend imports before the
  runtime timer, downloads and the harness's first-inference warmup are excluded;
  engine-internal profiling is included. This is not process-start-to-first-action
  time. `measure_startup.py` separately measures a fresh Python process through
  its first valid CPU action chunk, including interpreter/backend imports,
  initialization and first inference, with cached assets.
- Each candidate gets a fresh process. NVIDIA memory sampling tracks the process
  and its children at 100 ms intervals. This is a sampled peak, not an exact
  allocator maximum. PyTorch allocator counters are recorded separately.
- Quality uses LIBERO Spatial tasks 0–9, initial-state indices 0/1, environment
  seeds 42/43, noise seed `42 + 1000 * initial_state + chunk_index`, 50 replayed
  actions per chunk, and native episode limits. Success is the simulator flag.
- Fixture hashes and exact repeated actions are recorded. Synthetic fixtures
  measure execution only; they do not count as task success. Numerical parity
  and paired closed-loop outcomes are reported separately.

## Prepare native execution

On a minimal Ubuntu image, install the compiler/build tools and EGL libraries
before creating the Python environments. The TensorRT-LLM environment also needs
OpenMPI:

```sh
sudo apt-get update
sudo apt-get install -y build-essential libegl1 libgl1 libgl1-mesa-dev libosmesa6-dev libopenmpi-dev
```

CUDA-only VM images may omit NVIDIA's EGL library even when `nvidia-smi` works.
LIBERO needs a working headless rendering context. Install the graphics/EGL
package matching the existing NVIDIA driver and verify hardware rendering before
starting quality runs. On the recorded GCP CUDA M132 image, adding
`libnvidia-gl-580-server=580.178.04-0ubuntu0.22.04.1` supplied the missing EGL vendor
without changing the compute driver. This example is specific to that Ubuntu
image; WSL and other driver installations have different graphics packages.

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python cmake==3.31.6
PATH="$PWD/.venv/bin:$PATH" uv pip install --python .venv/bin/python \
  --index-strategy unsafe-best-match -r requirements-native.txt
```

Cache these exact Hugging Face snapshots before timing:

| Resource | Revision |
|---|---|
| `lerobot/smolvla_libero` | `31d453f7edd78c839a8bbc39744a292686daf0de` |
| `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |
| dataset `lerobot/libero-assets` | `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6` |

Checkpoint weights SHA256:
`9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`.
Set `LIBERO_CONFIG_PATH` to a directory containing `config.yaml` with absolute
`assets`, `bddl_files`, `benchmark_root`, `datasets`, and `init_states` paths.
Use `scripts/prepare_libero_assets.py --config-dir /absolute/path/libero-config`
with the selected environment to download the pinned assets and write this configuration.
Use `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl`; software rendering is not required.

The run root must contain `vla.cpp/build-cuda/vla-server`,
`vla.cpp/src/serving/vla.proto`, and `gguf/smolvla-{bf16,Q8_0,Q4_0,Q8_0-vision}.gguf`
for C++ candidates. Build vla.cpp commit
`52439f7c6c362d7bee218b400b9080cc32d75cc3` with the packed-weight patch from
[`../vla_cpp`](../vla_cpp/README.md), CUDA and serving enabled. Use that worker's
component allowlist quantizer; the upstream broad quantizer packs protected
weights and is unsuitable for this comparison. Verify protected tensors remain
byte-identical. The derived converter config needs state dimension 8: the
checkpoint metadata says 6, but its saved normalization and LIBERO observations
have 8 coordinates. Preserve source weights and record both config hashes.

## Run matched fixtures and paired quality

Run from the prepared run root. `SCRIPTS` must be an absolute path to this
worker's `scripts` directory and `PYTHON` to its environment's Python.

```sh
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

Repeat the final command for all eight candidates. Every output directory must
be new. Keep the captured fixture directory immutable for compared runs. The
matrix repeats native BF16 at the end to expose drift. Run each GPU separately.
Do not combine contention-affected timings with isolated measurements.

Measure full process startup separately in the appropriate backend environment:

```sh
"$PYTHON" "$SCRIPTS/measure_startup.py" \
  --backend native-bf16 --root "$PWD" \
  --fixture fixtures/task0-init0-chunk0.npz --output outputs/startup-native-bf16
```

This uses a fresh child process and a shared monotonic clock, stopping the timer
after the first complete, finite CPU action chunk. It excludes teardown and
artifact writes. Assets are already cached, and OS/file/driver caches are not
flushed. Keep these startup values separate from the harness's cached runtime
initialization field.

`scripts/with_paused_collection.py` is an optional administrative helper for an
explicitly authorized brief collection pause. It requires the exact collector
user/script, records PID/start-time identities, uses a detached resume watchdog,
and resumes on failure. Stopped CUDA processes retain VRAM; measure the benchmark
process's memory separately. Read its `--help` and obtain the job owner's
permission before use.

## Custom engine admission and execution

`llm_runtime_admission.py` records the real engines' checkpoint-admission errors.
SmolVLM support alone does not execute SmolVLA's action expert/denoising loop.
The custom adapters retain the original policy and transport continuous actions;
they are not stock supported-model integrations.

For vLLM, install CMake 3.31.6 in its own environment and put that environment's
`bin` directory on `PATH` while installing `requirements-vllm.txt` (EGL helper
builds need CMake older than 4). Then run
`scripts/patch_vllm_pooling.py` with that Python. vLLM 0.9.2's V0 pooling runner
otherwise drops `prompt_embeds`. The version-guarded patch retains a backup and
refuses unexpected source. Set `VLLM_USE_V1=0` and `PYTHONPATH="$SCRIPTS"`.
`probe_vllm_smolvla.py` compares worker actions with native BF16. The main harness
also accepts `--backend vllm-bf16` for shared fixtures and paired quality.

TensorRT-LLM remains the requested target. Its experimental probe is
`probe_trtllm_smolvla.py`; install `requirements-trtllm.txt` separately, plus system
OpenMPI. TensorRT-LLM 0.21 pins `datasets==3.1.0`, while LeRobot 0.4.4 requires
`datasets>=4`; its `numpy<2` constraint also conflicts with LeRobot's `rerun-sdk`.
The tested custom inference adapter uses `datasets==4.8.5` and `numpy==2.5.3` and
does not exercise TensorRT-LLM's dataset pipeline. Apply the explicit overrides with
`uv pip install --python .venv-trtllm/bin/python --index-strategy unsafe-best-match
--override requirements-trtllm-overrides.txt -r requirements-trtllm.txt` after
installing CMake 3.31.6 and putting that environment's `bin` on `PATH`.
This overrides an upstream declared constraint; full-action admission and quality
checks are required for the custom adapter, and do not validate unrelated engine
features. When using portable Python, expose its `LIBDIR` through `LD_LIBRARY_PATH`.
The validated FP16 adapter uses TensorRT-LLM's PyTorch executor, its context-logit-capable
sampler (`enable_trtllm_sampler=True`), and a custom context-output
buffer for continuous actions. Its first 350 entries are an action payload,
not language logits. It must pass complete-action comparison before any timing
or quality claim. INT8/INT4 TensorRT-LLM execution remains unvalidated. The shared harness accepts
`--backend trtllm-fp16` for the validated complete-action path.

## Tests

```sh
"$PYTHON" -m pytest -q tests
```

Checkpoint regression tests need the native environment; missing dependencies
are reported as skips, which must not be counted as GPU validation. The pooling
patch tests run without GPU dependencies. Keep raw result JSON, action arrays,
logs, dependency freezes, hashes and measurement commands alongside each report.

Latest committed measurements: [September 26 evidence](evidence/2026-09-26/REPORT.md).
Each host's `actions.tar.gz` contains the saved FP32 action arrays beside the
corresponding result paths. Extract it into that host's evidence directory to
rerun `compare_results.py` without loading a model. Input fixtures and complete
execution logs remain in the local run archive; their hashes identify the paired
inputs. The action archives contain no model weights.

RTX 3070 TensorRT-LLM setup is deferred by user decision: the host drive had
2.2 GiB free and WSL 6.6 GiB before cleaning this task's temporary transfer
archive, while the tested L4 environment occupies 17 GiB. No alternate disk was
available. This is a storage limitation, not a failed RTX model-execution test.
