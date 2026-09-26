# RTX 3070 quantization experiment — September 26, 2026

This report preserves the measured precursor experiment. See
[branch validation](gpu-branch-validation.md) for checks of the source imported
under `workers/vla_cpp`; those checks are separate from the hardware measurements.

The RTX 3070 is a working CUDA test target for the existing SmolVLA GGUF
quantization path. All five artifacts loaded, produced finite actions and
completed 60 timed predictions each. This is experimental evidence on a shared
WSL2 workstation; deployment selection still requires broader paired task
evaluation and a verified package.

## Target and isolation

- NVIDIA GeForce RTX 3070, 8192 MiB,
  compute capability 8.6, driver 610.88.
- WSL2 Linux `6.6.87.2-microsoft-standard-WSL2+`, Docker 29.6.1.
- A dedicated experiment workspace is mounted at `/workspace` inside the containers.
- CUDA build/runtime: `Dockerfile.cuda`; simulator: `Dockerfile.cuda-sim`;
  NVIDIA numerical pilot: `Dockerfile.modelopt`. Host Python and other services
  were not changed. Container jobs use four CPU cores and bounded host RAM.
- GPU jobs run sequentially within this experiment. Other workstation activity
  remains possible, so timing and whole-device memory are exploratory.

## Immutable inputs

| Input | Revision / SHA-256 |
|---|---|
| `HuggingFaceVLA/smolvla_libero` | `6721902bc4d61e50a3bfdb11dfb4cb626f05d102` |
| Source safetensors | `71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f` |
| vla.cpp | `52439f7c6c362d7bee218b400b9080cc32d75cc3` |
| llama.cpp / GGML | `7ba604f1cb61cd14898138e9abc0b4ff2601f180` |
| Packed SmolVLA patch | `b1ebe5bee36be84b843b5f140f97fe26ca66d36f4514331cf80b0fd4b8dbc36c` |
| CUDA benchmark binary | `f8e66d9c0efa3d4bba1c1edc75b2c53fa557c79d77d9e71f76603129d7ec29f5` |
| CUDA server binary | `0b8e2032a548bbc26b75220415e80ac66227c29969aa87bb331859644fafe311` |
| LIBERO | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| SmolVLM tokenizer/config | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |

All five GGUFs were regenerated on the RTX host from the pinned checkpoint and
matched the original CPU-screen file hashes byte for byte. The runtime preserves
224 packed LM matrices and, where requested, 72 packed vision matrices. Action
experts, projections, embeddings and other protected tensors retain their source
precision. The floating GGUF contains BF16/F32, rather than uniformly BF16.

The build enables CUDA for SM 8.6 and applies vla.cpp's GGML CUDA extension hook.
Benchmark instrumentation only records the unsorted per-call durations.
`runtime.patch`, `binary-source-sha256.txt`, CMake configuration and image IDs
are retained under `artifacts/cuda/setup/`.

## CUDA engine measurements

Each fresh process uses two synthetic 512px images, fixed state/language/noise,
four CPU threads, three warmups and 20 timed calls. Three rounds alternate
forward/reverse candidate order. CUDA execution, finite outputs, artifact hashes
and packed-matrix counts are required. All 300 timed predictions passed.

| Candidate | GGUF MiB | GPU weight buffer MiB | Pooled p50 ms | Pooled p95 ms | Synthetic action MAE |
|---|---:|---:|---:|---:|---:|
| Floating reference | 1074.18 | 1069.0 | 170.93 | 205.22 | 0 |
| LM Q8 | 792.93 | 787.8 | 161.42 | 167.57 | 0.00257 |
| LM Q4 | 642.93 | 637.8 | 161.03 | 167.37 | 0.01063 |
| LM Q8 + vision Q8 | 716.99 | 711.8 | 153.32 | 192.66 | 0.00508 |
| LM Q4 + vision Q8 | 566.99 | 561.8 | 152.91 | 161.45 | 0.00874 |

Latency covers the native prediction call, excluding camera capture, external
preprocessing, RPC and robot I/O. MAE compares 350 real action values with the
same-GPU floating control; all 1600 outputs are checked for finiteness. Synthetic
disagreement is not a task-success measure.

The smallest GGUF is 47.22% smaller than the floating reference. Its measured
median prediction time is about 10.5% lower in this screen. The LM Q4 batch
medians ranged from 142.22 to 162.88 ms, demonstrating host/run variability.
These data do not establish a stable production speedup or a quality winner.

GPU weight buffer size is the runtime-reported allocation, not total inference
VRAM. Sampled whole-device peaks were 5083, 4801, 4651, 4727 and 4577 MiB in table
order; they include desktop/other processes and use 100ms sampling. Host RSS is
reported separately. None of these quantities substitutes for audited package
size or a per-process peak-VRAM measurement.

## Paired LIBERO and NVIDIA pilots

The LIBERO pilot uses task 0, initial state 0, seed 42, a 500-step natural horizon and four replayed
actions per policy call. Noise seeds are paired across candidates. This is the
same development state already used in the CPU experiments, not a final test set.

| Candidate | Completed episode | Success | Environment steps |
|---|---|---|---:|
| Floating reference | Yes | Yes | 208 |
| LM Q8 | Yes | Yes | 128 |
| LM Q4 | Yes | Yes | 169 |
| LM Q8 + vision Q8 | Yes | Yes | 128 |
| LM Q4 + vision Q8 | Yes | **No** | 500 (natural horizon) |

All actions were finite, all five servers confirmed CUDA, and videos were saved.
The smallest candidate fails this paired pilot despite passing the engine screen.
Do not select it from file size or synthetic latency alone. LM Q4 succeeds here
but failed the same development state in the earlier CPU run; GPU and CPU quality
results are separate target evidence. One episode per candidate cannot establish
the planned success-rate floor or five-percentage-point non-inferiority gate.
No deployment winner is selected. Continue with additional episode-separated,
paired initial states/tasks for the surviving candidates and their float control.

The upstream simulator emits an observation-space warning for the task-description
field; image/state fields pass the recorded space check. This is retained in the
evidence rather than suppressing the warning.

ModelOpt uses version 0.41.0 with PyTorch 2.7.1 and LeRobot 0.4.3 in a separate
container. ModelOpt 0.47.0 requires PyTorch >=2.8, conflicting with this LeRobot
release's declared <2.8 requirement. Transformers is pinned to 4.57.1. These
experiments therefore do not claim validation of the latest ModelOpt API.

The native floating PyTorch policy loads strictly and produces finite action
chunks on four saved LIBERO observations. Its actual parameter inventory is
455,364,000 BF16 elements and 149,570,176 FP32 elements; all 314,634,240 target-LM
parameter elements are BF16. Predictions use BF16 CUDA autocast. The small NVIDIA feasibility pilot uses three earlier frames for
calibration and four disjoint later frames from that same trajectory for numerical
comparison. It does not satisfy the planned 128-example, episode-separated
calibration/evaluation protocol. Quantization scope is restricted to the 224 LM
linear layers; protected parameters are hash checked.

AWQ/SmoothQuant fake quantization is kept separate from the GGUF measurements.
It must not be entered as measured packed memory, packed latency or deployable
quality. AutoQuantize remains gated on a working fixed-recipe export/reload;
native NVFP4 remains a Blackwell experiment rather than an RTX 3070 capability.

| ModelOpt recipe | Enabled quantizers | Calibration seconds | Action MAE | Max absolute difference | Result |
|---|---:|---:|---:|---:|---|
| LM AWQ INT4, block 128 | 224 weight | 19.70 | 0.02157 | 1.98559 | Finite numerical pilot; export failed |
| LM SmoothQuant INT8 | 224 weight + 224 input | 7.23 | 0.04634 | 2.07386 | Finite numerical pilot; export failed |

Errors compare the four frames' unnormalized 50-by-7 action chunks against the
unchanged native policy in the same process. Channels have different physical
units; per-channel errors and the actual arrays are retained. Both protected
parameter hashes matched before/after calibration. The reported calibration
times are setup costs, not inference benchmarks.

The final records are `runs/modelopt-awq-v2/` and
`runs/modelopt-smoothquant-v2/`. These repeats add the measured dtype inventory
and reproduce the earlier action-error metrics exactly. Earlier logs are retained;
their blanket FP32 parameter label was an instrumentation assumption corrected
by the dtype audit, not a change to policy weights or inference.

Both full-policy `export_hf_checkpoint` attempts failed with
`AttributeError: 'SmolVLAConfig' object has no attribute 'torch_dtype'`.
The generic HF exporter assumes a Transformers configuration/save contract;
LeRobot's policy configuration and `save_pretrained` implementation differ.
A SmolVLA export adapter and packed engine remain unimplemented. Adding a
configuration attribute alone would not validate the saved tensor map, full
policy reload, native quantized kernels or action parity. This finding does not
mean SmolVLA export is impossible; it fails the current N1 compatibility gate.

ModelOpt also logged that its optional compiled CUDA quantization extension was
unavailable in the PyTorch runtime image (`CUDA_HOME`/compiler absent), and used
its fallback implementation. Policy tensors executed on CUDA, but no native
INT4/INT8 kernel claim is made. Peak PyTorch allocation was about 1665/1656 MiB
for these pilots, including model initialization/calibration; it is not packed
deployment VRAM. Stop H9 search until the export/runtime adapter works. Continue
paired evaluation through the working GGUF CUDA path.

## Reproduction and evidence

Run from `workers/vla_cpp` on the Linux CUDA host. Prepare the pinned
inputs described in [GPU setup](gpu-setup.md) first. The container mount maps
this worker directory to `/workspace`:

```bash
docker build -f Dockerfile.cuda -t firebird-quant-cuda:20260926 .
docker run --gpus device=0 --user "$(id -u):$(id -g)" --cpus=4 --memory=8g \
  -v "$PWD:/workspace" --entrypoint bash firebird-quant-cuda:20260926 scripts/build_cuda.sh
docker run --gpus device=0 --cpus=4 --memory=8g -v "$PWD:/workspace" \
  --entrypoint python3 firebird-quant-cuda:20260926 \
  -m policykit.cuda_bench --run smolvla-cuda-v2 --reps 20 --rounds 3
docker build -f Dockerfile.cuda-sim -t firebird-quant-sim:20260926 .
docker run --gpus device=0 --cpus=4 --memory=8g -v "$PWD:/workspace" \
  --entrypoint bash firebird-quant-sim:20260926 scripts/run_cuda_rollouts.sh smolvla-cuda-libero-v2
```

Use a fresh run name: the benchmark refuses to overwrite evidence. The rollout
script binds to the original `smolvla-cuda-v1` manifest; change that explicit path
when evaluating a different engine run. GPU exposure is also needed when linking
on this WSL target so `libcuda.so.1` resolves.

To repeat the NVIDIA numerical pilot against the frozen development captures:

```bash
docker build -f Dockerfile.modelopt -t firebird-modelopt:20260926 .
docker run -v "$PWD:/workspace" --entrypoint python firebird-modelopt:20260926 \
  scripts/prepare_modelopt_metadata.py
docker run --gpus device=0 --cpus=4 --memory=7g -v "$PWD:/workspace" \
  --entrypoint python firebird-modelopt:20260926 -m policykit.modelopt_pilot \
  --method awq --out artifacts/cuda/runs/modelopt-awq-repeat \
  --metadata artifacts/cuda/modelopt-metadata \
  --rollout artifacts/docker/runs/smolvla-experiments-v4/float_reference/task0-init0-seed42-steps500/result.json
```

Use `--method smoothquant` with a different output directory for H8. The script
checks the checkpoint/capture hashes, requires CUDA, checks enabled quantizer
scope and protected parameter hashes, and retains export errors. Export failure
does not invalidate the numerical diagnostic or qualify it for deployment.

Bulk files stay outside Git. Local evidence is copied to `artifacts/cuda/`, with
the original copy retained in the remote workspace. Engine evidence lives in
`runs/smolvla-cuda-v1/`; setup logs preserve failed attempts as well as successful
builds. The initial simulator failure was a missing Python development header;
the final Dockerfile adds it. ModelOpt setup failures exposed missing optional
processor dependencies, which are now explicit in its Dockerfile.

The compact [evidence manifest](quantization-rtx3070-evidence.json) records exact
candidate/image/source hashes and the measured results. Local archives
`artifacts/cuda/evidence-20260926.tar.gz` and `source-snapshot-20260926.tar.gz`
preserve the run outputs/videos and the uncommitted experiment source separately.

Validation: the remote Python suite passed **63 tests**, with two tests skipped
because their hardcoded CPU build path is absent. CUDA behavior is separately
exercised by the actual native engine matrix above. The six CUDA log-parser tests
reject CPU fallback and missing/nonfinite/nonpositive samples. Local focused
validation passed eight tests with the same two Linux-native skips.
