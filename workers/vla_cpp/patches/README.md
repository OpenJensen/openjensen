# Packed SmolVLA loader patch

The canonical patch is `../policykit/patches/vla-cpp-smolvla-packed.patch` and is
included in installed wheels. It targets vla.cpp commit
`52439f7c6c362d7bee218b400b9080cc32d75cc3` (v0.3.0).

It preserves Q8_0/Q4_0 LM and vision matrix storage when loading GGUF instead of
allocating permanent floating-point matrices. Action-expert, projector, embedding,
normalization and action/state tensor paths remain unchanged. Shape, type and byte
count checks reject incompatible packing. Runtime logs report packed allocations.

In the prepared checkout, use `git apply --check /absolute/path/to/this.patch`
then `git apply /absolute/path/to/this.patch`. Build its native targets using the
upstream CMake instructions. The worker's `--describe-runtime` operation requires
the exact commit and patch diff before resolving a job.

CPU benchmark/reproduction tooling and measured support are reviewed in
`feat/quantization_benchmark_cpu`. This source patch alone does not establish task
success, CUDA support or deployment compatibility.
