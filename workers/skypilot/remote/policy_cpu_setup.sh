#!/usr/bin/env bash
set -euo pipefail

readonly POLICY_UV_VERSION=0.8.22
readonly POLICY_PYTHON_VERSION=3.12
readonly POLICY_BOOTSTRAP="$HOME/vla-bootstrap"
readonly POLICY_ENV="$HOME/vla-env"
readonly CONTROL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

[[ "${POLICY_RUNTIME:-}" == packed-act-cpu && "${POLICY_DEVICE:-}" == cpu ]]
[[ "${POLICY_MODEL_FORMAT:-}" == firebird_quant ]]
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]]
test -f "$HOME/firebird-quant-src/firebird_quant/native_consumer.py"
test -f "$HOME/firebird-act-src/firebird_act/probe.py"

# New CPU worker only; the existing CUDA setup script remains unchanged.
sudo apt-get update
sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-venv
python3 -m venv "$POLICY_BOOTSTRAP"
"$POLICY_BOOTSTRAP/bin/python" -m pip install "uv==$POLICY_UV_VERSION"
"$POLICY_BOOTSTRAP/bin/uv" --no-config venv --python "$POLICY_PYTHON_VERSION" "$POLICY_ENV"
"$POLICY_BOOTSTRAP/bin/uv" --no-config pip install --python "$POLICY_ENV/bin/python" \
  --index-url https://pypi.org/simple -r "$CONTROL_DIR/policy-cpu.requirements.txt"

export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$HOME/sky_workdir:$HOME/firebird-quant-src:$HOME/firebird-act-src"
"$POLICY_ENV/bin/python" - <<'PY'
import sys
import torch
from firebird_act.probe import runtime_versions
from firebird_quant.native_consumer import load_packed_act

assert sys.version_info[:2] == (3, 12), "Packed ACT requires Python 3.12"
runtime_versions()
assert torch.version.cuda is None, "Packed ACT requires CPU-only Torch"
assert not torch.cuda.is_available(), "Packed ACT must not expose a CUDA device"
assert torch.empty(0).device.type == "cpu", "Packed ACT must allocate on CPU"
assert callable(load_packed_act)
PY
