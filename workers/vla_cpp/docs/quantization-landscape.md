# Quantization implementation catalog

## Comparable in PolicyKit v1

| Implementation | Variants | What is quantized | Target | Why included |
|---|---|---|---|---|
| [vla.cpp](https://github.com/VinRobotics/vla.cpp) | BF16, Q8_0, Q4_0; optional vision packing | LM-backbone matrix weights by default; `--vision` also packs eligible vision-tower weights. Action expert, embeddings, output head, and norms remain float | x86 CPU now; CUDA, Metal, Jetson later | It supports SmolVLA and π0 under one GGUF runtime, includes the converter, quantizer, latency tool, and LIBERO client. |

## Tracked, not comparable in v1

| Implementation | Status | Reason |
|---|---|---|
| bitsandbytes LLM.int8 / 4-bit | Research candidate | Runtime loading rather than a portable inference artifact; requires model-specific PyTorch compatibility and supported accelerator validation. |
| ONNX Runtime quantization | Research candidate | Must inspect the exported graph to prove operators are actually quantized before reporting an INT8 row. |
| TensorRT | Future RTX lane | Requires NVIDIA CUDA/TensorRT hardware and model-specific export validation, unavailable on the Intel Mac target. |

The leaderboard only compares rows produced by the same vla.cpp version, with the same checkpoint revision, task IDs, seeds, and runtime flags. Cross-model success rates are displayed for exploration but are not normalized rankings.
