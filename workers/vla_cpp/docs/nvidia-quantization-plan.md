# NVIDIA quantization research and GPU plan

The [RTX 3070 report](quantization-rtx3070.md) records 300 GGUF CUDA predictions,
five paired development episodes and small AWQ/SmoothQuant numerical pilots.
The smallest GGUF failed its episode; both ModelOpt full-policy exports failed.
These experiments do not mark the conversion worker as deployment verified.

Current order: broaden paired CUDA evaluation for surviving GGUF candidates;
implement and validate a full SmolVLA ModelOpt export/runtime adapter before H9;
retain NVFP4 for a separately verified Blackwell target. The ModelOpt compatibility
spike did not complete N1 (fixed recipe export/reload), so N2 (AutoQuantize search)
remains gated. No deployment winner is selected.

## Sources and proposed experiments


Starting point: [upstream document 13 at architecture commit
130aa1c](https://github.com/sobhanb-eth/firebird-hackathon-prep/blob/130aa1c4a37b9910ed5b90b76834960ffa976461/docs/idea/13_optimizer-research-and-decision.md).
Its NVIDIA entries are an AutoQuantize technical announcement/API and a Model
Optimizer roadmap, rather than two VLA research papers. Following their references
also identifies useful PTQ papers. Sources below were checked September 26, 2026;
the original proposals are followed by the measured status below.

**Decision:** keep the existing packed GGUF path as the first deliverable. Add
Model Optimizer as an optional, isolated candidate-generation worker after that
path works. The shared core still evaluates and selects the smallest feasible
artifact; a quantizer's estimated accuracy or bit budget cannot certify it.

| Source and evidence | Useful technique | Addition to our plan |
|---|---|---|
| [NVIDIA AutoQuantize](https://nvidia.github.io/Model-Optimizer/announcements/autoquantize.html), August 24, 2026; technical announcement | Gradient-weighted operator output error estimates sensitivity; an integer-program search assigns formats under an effective-bit budget. Runtime-coupled operators can share one decision. | Reuse this search in H9 once a VLA adapter and export path work. Preserve QKV/gate-up grouping where the selected runtime requires it. Its cost excludes some weights and does not measure end-to-end latency or peak VRAM. |
| [AutoQuantize API](https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.torch.quantization.model_quant.html); documented interface | Custom forward/loss callbacks, disabled layers, calibration/scoring budgets and resumable search state. | Supply a model-native action loss and explicit protected modules; save search inputs/results with each recipe. Validate the pinned API before implementation. |
| [SmoothQuant](https://arxiv.org/abs/2211.10438), ICML 2023; MIT/NVIDIA paper | Offline channel rescaling balances activation outliers against weight quantization error, enabling W8A8 INT8. | H8 tests whether calibrated weight-and-activation quantization offers a useful target-device tradeoff. It requires compatible graph transformations and INT8 kernels; it is different from GGUF Q8_0 weight packing. |
| [AWQ](https://arxiv.org/abs/2306.00978), MLSys 2024; paper implemented in [ModelOpt's PTQ examples](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/hf_ptq) | Activation statistics guide channel scaling for low-bit weight-only quantization, without gradient-based calibration. | H7 is the first calibrated INT4 candidate: language backbone W4A16, preserving the other components initially. Check VLA-specific export and runtime support before allocating rollout time. |
| [Introducing NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/), June 24, 2025; NVIDIA technical article | Blackwell's microscaled FP4 uses FP8 scales for 16-value blocks and an additional tensor scale. | H10 is conditional on a compatible Blackwell deployment target. Count scale metadata and protected weights; nominal four-bit precision is not the complete artifact size. |
| [Model Optimizer roadmap #1699](https://github.com/NVIDIA/Model-Optimizer/issues/1699); evolving product plan | Lists VLA PTQ work for automotive Alpamayo and planned broader QAT/QAD work. | Track upstream adapters, but require evidence for our exact SmolVLA/OFT checkpoint. This roadmap is not proof of manipulation-policy or LIBERO support. |

AutoQuantize also cites [SqueezeLLM](https://arxiv.org/abs/2306.07629), ICML 2024,
for related sensitivity reasoning. Its nonuniform dense/sparse representation is
background reading, not a new GGUF preset. AWQ and SqueezeLLM should retain their
own paper attribution rather than being labeled NVIDIA-authored methods.

### Hardware and representation preflight

The existing RTX 3070 and upstream RTX 4070 profiles are different devices.
[NVIDIA's GPU table](https://developer.nvidia.com/cuda/gpus) lists them at compute
capabilities 8.6 and 8.9 respectively. Use the actual device, available memory,
driver and pinned runtime to resolve formats:

- **Intel Mac / Docker CPU:** continue H0–H4 with the current `vla.cpp` worker.
- **RTX 3070:** investigate native INT4 weight-only or INT8 execution; do not
  schedule the FP8/NVFP4 example recipe as a native accelerated candidate.
- **RTX 4070:** investigate INT4/INT8 and, when the exact runtime/model supports
  it, FP8. ModelOpt documents FP8 for Ada/Hopper; this is hardware eligibility,
  not evidence that the entire VLA can be exported or run.
- **Blackwell:** permit NVFP4 only after model/runtime compatibility is verified.
  Blackwell results remain a separate target row; they cannot establish a local
  RTX 3070/4070 speedup. Any emulated format must be labeled separately.

Format guidance comes from [ModelOpt's technical
resources](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/hf_ptq#technical-resources)
and NVIDIA's NVFP4 article above. Check the [TensorRT support
matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)
for the chosen version. TensorRT engine portability depends on platform, version
and hardware compatibility settings; include those in the package identity.

ModelOpt output is not interchangeable with GGUF Q4_0/Q8_0 or the trainer's NF4
weights. The current PolicyKit quantizer has a component allowlist but no arbitrary
per-layer format-map input. Do not translate an AutoQuantize assignment into GGUF
by changing dtype names. Any future bridge needs equivalent quantization math,
packing, graph transforms, kernels and independent numerical/rollout validation.

### Bounded experiments and stop conditions

These are project proposals, not paper results. Keep H0–H6 and add the following
experiment IDs. Within one GPU
backend, first establish a floating reference and one exported fixed recipe;
only then attempt mixed-precision search. This sequence is independent of the
numeric hypothesis IDs.

| ID | Candidate and control | Admission or falsification |
|---|---|---|
| H7 | ModelOpt AWQ W4A16 on eligible LM layers; native floating reference and, if available, same-backend ordinary INT4 control. Vision, action expert, projections and other protected modules stay at audited source precision. | Require packed export/reload, task-quality floor and measured memory/latency gates. Lower reconstruction error alone does not pass. Without a same-backend INT4 control, report a deployment comparison rather than an isolated AWQ benefit. |
| H8 | ModelOpt SmoothQuant W8A8 on supported LM operators; same-backend floating and ordinary INT8 controls where available. Freeze smoothing settings before final evaluation. | Verify the rescaling preserves the floating graph before quantization and the deployed graph executes intended INT8 kernels. Reject if quality or measured device constraints fail. |
| H9 | AutoQuantize with only preflight-approved formats and the same protected modules; compare fixed and seeded-random assignments under matched artifact-size and search/rollout budgets. | Start with two preregistered bit budgets, at most two exported search candidates. No benefit over controls means the search extension is not justified. Effective bits alone cannot match whole-policy size. |
| H10 | NVFP4 with selective FP8/source-float protection on a verified Blackwell backend; compare its own floating and fixed-format controls. | Deferred until hardware and complete export path are available. Exclude from the RTX 3070/4070 native candidate set. |

For H7–H10, proposed calibration starts with 128 representative observation/action
chunks at batch size 1, and H9 uses up to 32 of these for scoring. These are small
pilot limits to measure feasibility, not NVIDIA defaults or validated sample
requirements. Record the realized sample/batch counts and calibration coverage;
increase them only within a declared time/memory budget and using search data.
Use compatible camera images, instructions, state, masks, normalization and action
chunks; text-only calibration is not our VLA protocol. Split by episode to keep
calibration/search and final evaluation separate.

H9's gradient method needs a differentiable scalar loss: use the selected native
policy's supervised action or flow-matching training loss, with fixed noise/time
sampling where relevant. Check finite gradients reach the searched layers while
protected modules remain unquantized. Do not copy an LLM `output.loss` callback
without checking its meaning. The [API's KL option](https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.torch.quantization.model_quant.html)
expects logits; continuous SmolVLA actions are not automatically valid inputs.
If the loss adapter or scoring memory does not work, retain fixed PTQ and record
the unsupported search capability.

Timebox the first model/export compatibility spike to two engineer-hours after
GPU access. If it cannot export and reload one supported fixed recipe, defer
this backend and continue Q4–Q6. Reserve independent final evaluation and package
reload time before any further search. Quantizer simulation and paper-reported
scores stay diagnostic; neither enters the deployment leaderboard as a measured
policy result.

### Worker and evidence additions

Use a proposed `workers/modelopt/` environment alongside `workers/vla_cpp/`.
Keep NVIDIA/PyTorch/runtime imports out of `firebird_core`. Extend F1/Q2 contracts
with optional fields so existing calibration-free GGUF recipes stay valid:

- Quantizer/runtime commit or version, export format, architecture restrictions,
  weight/activation/KV precision separately, scale/block layout and exact module
  grouping. Leave KV quantization disabled in the initial experiment.
- Calibration and scoring dataset revisions, episode IDs, preprocessing hashes,
  sample counts, seeds, loss identity, elapsed time and peak calibration memory.
- Requested bit budget, search-eligible parameter count, excluded/protected
  modules, sensitivity/search-state artifacts and the resolved precision map.
- Exported tensor/graph audit, kernel or engine-inspection evidence, full policy
  weight bytes including float modules/scales, package bytes, target inference
  memory and observation-to-action timings. Calibration memory is a separate cost.

Use execution, paired-evaluation, constrained-selection and fresh-process export gates (the integration plan's Q3–Q6).
Record float parity between native PyTorch and the new deployment backend before
quantization comparisons. Cross-backend results may compare deployable choices on
one task/target contract, but must expose the backend change and its own floating
control; do not attribute the whole latency difference to quantization.
