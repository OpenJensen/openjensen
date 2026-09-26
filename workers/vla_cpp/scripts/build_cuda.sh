#!/usr/bin/env bash
set -euo pipefail
cd /workspace
vendor=artifacts/docker/vendor/vla.cpp
llama=artifacts/cuda/vendor/llama-download
build=artifacts/cuda/build
test "$(git -C "$vendor" rev-parse HEAD)" = 52439f7c6c362d7bee218b400b9080cc32d75cc3
git -C "$vendor" apply --include=src/models/smolvla.cpp --reverse --check /workspace/policykit/patches/vla-cpp-smolvla-packed.patch
python3 "$vendor/scripts/patch_ggml_cuda_ext_hook.py" "$llama"
python3 scripts/instrument_cuda_bench.py "$vendor/src/serving/vla-bench.cpp"
cmake -S "$vendor" -B "$build" \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86-real \
    -DGGML_NATIVE=OFF -DGGML_CUDA_NCCL=OFF -DVLA_BUILD_TESTS=ON \
    -DFETCHCONTENT_SOURCE_DIR_LLAMA="/workspace/$llama"
cmake --build "$build" --target vla-bench vla-server vla_predict_check -j 3
git -C "$vendor" diff > artifacts/cuda/setup/runtime.patch
sha256sum "$build/vla-bench" "$build/vla-server" "$build/tests/vla_predict_check" \
    policykit/patches/vla-cpp-smolvla-packed.patch scripts/instrument_cuda_bench.py \
    "$llama/ggml/src/ggml-cuda/ggml-cuda.cu" > artifacts/cuda/setup/binary-source-sha256.txt
cp "$build/CMakeCache.txt" artifacts/cuda/setup/CMakeCache.txt
