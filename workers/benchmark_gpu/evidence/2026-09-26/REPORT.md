# SmolVLA GPU comparison — September 26, 2026

**No overall winner is selected.** Completed evidence includes all eight native/C++ L4 latency candidates, isolated custom-engine timings with native controls, seven RTX 20-episode quality runs, all ten L4 20-episode quality runs, and full-action custom vLLM/TensorRT-LLM admission probes. The RTX Tailscale gateway went offline at 14:14 UTC (18:14 Yerevan) and the host subsequently failed. Direct access to the existing L4 has been restored through the user's GCP account; all fourteen original fixtures were recovered. A dedicated replacement L4 is running within a user-authorized $6 budget. Results from unfinished runs are not scored.

RTX TensorRT-LLM setup is explicitly deferred by user decision: WSL had 6.6 GiB free and its Windows host drive 2.2 GiB, versus a 17 GiB tested L4 environment. Cleaning this task's temporary transfer archive recovered WSL space, but no sufficiently large alternate disk was available. This is a storage gate, not an RTX model-execution failure.

## Shared contract

Every timed request starts with raw CPU observations, state, task text and explicit FP32 diffusion noise, and ends with a complete unnormalized CPU 50 × 7 action chunk. Checkpoint preprocessing, all ten denoising steps, postprocessing, transfers and C++ RPC/custom engine dispatch are included. Simulator stepping and camera acquisition are excluded. Startup measures cached runtime initialization inside the harness process: model/processor setup, quantization, C++ server launch and engine-worker initialization where applicable. Top-level Python imports, downloads and harness first-inference warmup are excluded; engine-internal profiling is included. This is not process-start-to-first-action time.

Checkpoint: `lerobot/smolvla_libero@31d453f7edd78c839a8bbc39744a292686daf0de`; weights SHA256 `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`. The backbone/tokenizer and LIBERO asset revisions are pinned in the worker README. Fourteen identical fixtures (two synthetic, twelve captured observations) are paired by SHA256. Quality uses all ten LIBERO Spatial tasks, initial states 0/1, environment seeds 42/43, explicit noise seed `42 + 1000 * initial_state + chunk_index`, 50 replayed actions per chunk and native episode limits.

`passed` in a result file means execution and repeatability checks completed; it does not mean the candidate retained reference quality. Twenty episodes are a small behavior-retention sample. Checkpoint training overlap was not audited, so these are not generalization scores.

## L4: uncontended observation-to-action measurements

The authorized collection pause lasted about eight minutes. Ten pre-run GPU utilization samples were 0%; all eight collection workers were confirmed running again afterward. Stopped collectors retained 2,043 MiB VRAM. The table uses only the benchmark process and descendants, sampled every 100 ms, and excludes those collectors. Sampling can miss brief peaks. Each fresh process used one warmup and five timed calls per fixture.

| Candidate | Median of fixture p50 (ms/chunk) | Cached startup (s) | Sampled process peak (MiB) |
|---|---:|---:|---:|
| native-bf16 | 348.80 | 27.79 | 1186 |
| cpp-bf16 | 100.28 | 2.28 | 1162 |
| native-fp16 | 347.36 | 27.95 | 1186 |
| cpp-Q8_0 | 99.34 | 3.06 | 1018 |
| native-int8 | 418.20 | 29.16 | 1036 |
| cpp-Q4_0 | 101.62 | 2.84 | 946 |
| native-nf4 | 386.49 | 29.29 | 950 |
| cpp-Q8_0-vision | 101.02 | 2.84 | 944 |
| native-bf16-repeat | 350.64 | 27.92 | 1186 |

BF16 median drift from the first to the final fresh-process repeat was +0.53%. These measurements are specific to this existing `g2-standard-12` L4 window (12 vCPUs, 48 GiB RAM, driver 580.159.04).

A second authorized pause collected the custom engines in their own environments, each bracketed by native controls. Ten pre-run utilization samples were 0%; the helper resumed all nine recorded collection processes after 489 seconds. These are separate measurement windows from the core table, with one warmup and five timed calls per fixture.

| Environment / candidate | Median of fixture p50 (ms/chunk) | Cached startup (s) | Sampled process peak (MiB) |
|---|---:|---:|---:|
| vLLM environment: native-bf16 | 349.58 | 31.55 | 1186 |
| custom vLLM BF16 | 356.23 | 12.34 | 1588 |
| vLLM environment: native-bf16-repeat | 355.75 | 28.13 | 1186 |
| TensorRT-LLM environment: native-fp16 | 338.61 | 27.93 | 1186 |
| custom TensorRT-LLM FP16 | 409.42 | 42.93 | 1632 |
| TensorRT-LLM environment: native-fp16-repeat | 346.14 | 28.21 | 1186 |

Native-control drift was +1.76% in the vLLM environment and +2.23% in the TensorRT-LLM environment. Neither custom adapter demonstrates an observation-to-action speed advantage over its native controls in these measurements. Their contended admission-probe times are excluded.

## RTX 3070: observation-to-action diagnostics and paired quality

Each candidate used one warmup and three samples per fixture, followed by twenty paired episodes. This is a shared WSL desktop, not an isolated system. Process-scoped peak VRAM was not reliably collected; do not substitute whole-device usage or compare it with L4 process peaks. Earlier native allocator-only smoke measurements are a different memory scope and are excluded here.

| Candidate | Median of fixture p50 (ms/chunk) | Cached startup (s) | LIBERO success |
|---|---:|---:|---:|
| native-bf16 | 325.29 | 14.33 | 15/20 |
| native-fp16 | 312.30 | 14.27 | 15/20 |
| native-int8 | 372.32 | 14.85 | 15/20 |
| native-nf4 | 377.31 | 15.81 | 15/20 |
| cpp-bf16 | 112.72 | 1.41 | 14/20 |
| cpp-Q8_0 | 109.30 | 1.30 | 15/20 |
| cpp-Q4_0 | 110.92 | 1.31 | 8/20 |
| cpp-Q8_0-vision | Not collected | Not collected | Not collected |
| custom vLLM BF16 | Not collected | Not collected | Not collected |
| custom TensorRT-LLM FP16 | Deferred: disk space | — | — |

C++ Q4 scored 8/20 versus native BF16's 15/20. It lost ten episodes the reference solved and gained three the reference failed. Native NF4 also scored 15/20, but swapped three successes for three failures. Equal aggregate scores therefore do not establish equivalent behavior. Native FP16, native INT8 and C++ Q8 matched the reference's success/failure outcome on all twenty episodes; C++ BF16 lost one.

## L4: completed paired quality

These quality runs used the same episode identities with the collection job running. Their timings are excluded from the uncontended table.

| Candidate | Success | Reference-only successes | Candidate-only successes |
|---|---:|---:|---:|
| native-bf16 | 16/20 | — | — |
| native-fp16 | 15/20 | 2 | 1 |
| native-int8 | 18/20 | 0 | 2 |
| native-nf4 | 12/20 | 5 | 1 |
| cpp-bf16 | 17/20 | 0 | 1 |
| cpp-Q8_0 | 17/20 | 0 | 1 |
| cpp-Q4_0 | 4/20 | 13 | 1 |
| cpp-Q8_0-vision | 15/20 | 2 | 1 |
| custom vLLM BF16 | 16/20 | 1 | 1 |
| custom TensorRT-LLM FP16 | 16/20 | 1 | 1 |

All ten candidates completed against the same fixture hashes and episode identities. C++ Q4 lost thirteen reference successes and gained one; its 4/20 result does not retain reference quality. Native NF4 lost five and gained one. Vision Q8 lost two and gained one. GPU-specific quality outcomes are kept separate.

## Numerical fidelity and integration scope

| RTX candidate vs native BF16 | Action RMSE, 14 fixtures | Gripper-sign agreement |
|---|---:|---:|
| native-fp16 | 0.001969 | 100.00% |
| native-int8 | 0.049805 | 99.57% |
| native-nf4 | 0.098015 | 98.71% |
| cpp-bf16 | 0.005298 | 100.00% |
| cpp-Q8_0 | 0.003281 | 100.00% |
| cpp-Q4_0 | 0.101806 | 98.71% |

C++ BF16's maximum absolute error was 0.339498, driven by a synthetic gripper value whose sign agreed. On captured observations the maximum was 0.013786, versus native FP16/BF16's 0.012943. These are measured differences, not bit-exact parity or a post-hoc quality tolerance. Protected tensor audits confirm 112 LM matrices packed for Q8/Q4 and 184 LM+vision matrices for the optional vision recipe; 391/319 protected tensors respectively remain byte-identical. The action expert and projectors stay floating.

Both custom engines executed the complete policy, including the action expert and denoising loop, inside their workers. vLLM 0.9.2 V0 pooling matched its same-environment native BF16 reference exactly on three fixtures. TensorRT-LLM 0.21.0 PyTorch execution with its context-logit-capable sampler matched same-environment native FP16 exactly on three fixtures. A worker call counter verified that the full policy ran; dummy profiling outputs cannot pass admission.

These adapters use native PyTorch SmolVLA kernels, not vLLM paged-attention or TensorRT-compiled policy acceleration. The vLLM pooling runner needs the supplied version-guarded prompt-embedding forwarding patch. TensorRT-LLM carries continuous actions in a custom context-output buffer; those values are not language probabilities. INT8/INT4 engine-native execution remains unvalidated. The TensorRT-LLM environment explicitly overrides its declared `datasets==3.1.0` and `numpy<2` constraints with `datasets==4.8.5` and `numpy==2.5.3`, required by the LeRobot dependency set. Its package check reports those two incompatibilities. The supplied override file reproduces the tested custom inference path; it does not establish compatibility for unrelated TensorRT-LLM features.

Environment versions differ across engines: native Torch 2.7.1 / Transformers 4.57.1; vLLM Torch 2.7.0 / Transformers 4.53.3; TensorRT-LLM Torch 2.7.1 / Transformers 4.51.3. Three-fixture exact checks are against each engine's own native reference. Against the main L4 BF16 environment on all fourteen fixtures, custom vLLM RMSE was 0.002578 with full gripper-sign agreement; custom TensorRT-LLM FP16 RMSE was 0.026427 with 99.86% gripper-sign agreement. Each engine's twenty-episode quality swapped one success and one failure. The comparison does not conflate those two validation scopes.

## Validation and recovery

Six standalone worker regression tests passed on L4 with zero skips; four dependency-light tests passed locally. Ruff and diff checks passed. JSON evidence, paired comparisons, protected-tensor hashes and `SHA256SUMS` accompany this report. Each host's `actions.tar.gz` contains the saved FP32 action arrays for completed results, checked for finite `1 × 50 × 7` outputs. Extract it into that host's evidence directory to recompute numerical and paired comparisons with `compare_results.py`. Full logs and all fourteen input fixtures remain in the task's local evidence archive. Every recovered fixture was checked against its historical baseline hash; `l4/recovered-fixtures-sha256.json` records those identities. Package inventories, GPU/driver identity and harness hashes are recorded separately for the three L4 environments. Model weights are not committed.

The user authorized a $6 replacement rental. A dedicated `g2-standard-8` L4 (8 vCPUs, 32 GiB RAM, 200 GiB auto-delete disk, driver 580.178.04) started at 15:04 UTC. Its four-hour termination deadline is 19:04 UTC, with automatic VM deletion and boot-disk deletion. New measurements remain pending and will be recorded separately from this historical L4. Results are copied locally during execution; the VM will be explicitly deleted after collection.

Both collection pauses on the existing L4 were confirmed resumed. RTX TensorRT-LLM remains deferred, and remaining RTX runs cannot continue on the failed host.
