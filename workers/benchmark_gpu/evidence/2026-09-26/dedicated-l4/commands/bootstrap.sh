#!/bin/bash
set -euo pipefail
cd "$HOME/smolvla-benchmark"
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update >setup/apt-update.log 2>&1
sudo apt-get install -y build-essential libopenmpi-dev libegl1 libgl1 libgl1-mesa-dev libosmesa6-dev patchelf >setup/apt-install.log 2>&1
test -x .venv-native/bin/python || ./uv venv --python 3.11 .venv-native >setup/native-install.log 2>&1
./uv pip install --python .venv-native/bin/python cmake==3.31.6 >>setup/native-install.log 2>&1
PATH="$PWD/.venv-native/bin:$PATH" ./uv pip install --python .venv-native/bin/python --index-strategy unsafe-best-match -r workers/benchmark_gpu/requirements-native.txt >>setup/native-install.log 2>&1
touch setup/native-installed
test -x .venv-vllm/bin/python || ./uv venv --python 3.11 .venv-vllm >setup/vllm-install.log 2>&1
./uv pip install --python .venv-vllm/bin/python cmake==3.31.6 >>setup/vllm-install.log 2>&1
PATH="$PWD/.venv-vllm/bin:$PATH" ./uv pip install --python .venv-vllm/bin/python --index-strategy unsafe-best-match -r workers/benchmark_gpu/requirements-vllm.txt >>setup/vllm-install.log 2>&1
.venv-vllm/bin/python workers/benchmark_gpu/scripts/patch_vllm_pooling.py >setup/vllm-patch.log 2>&1
touch setup/vllm-installed
test -x .venv-trtllm/bin/python || ./uv venv --python 3.12 .venv-trtllm >setup/trtllm-install.log 2>&1
./uv pip install --python .venv-trtllm/bin/python cmake==3.31.6 >>setup/trtllm-install.log 2>&1
PATH="$PWD/.venv-trtllm/bin:$PATH" ./uv pip install --python .venv-trtllm/bin/python --index-strategy unsafe-best-match --override workers/benchmark_gpu/requirements-trtllm-overrides.txt -r workers/benchmark_gpu/requirements-trtllm.txt >>setup/trtllm-install.log 2>&1
touch setup/trtllm-installed
