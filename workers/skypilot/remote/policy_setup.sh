#!/usr/bin/env bash
set -euo pipefail

readonly POLICY_UV_VERSION=0.8.22
readonly POLICY_PYTHON_VERSION=3.12
readonly POLICY_BOOTSTRAP="$HOME/vla-bootstrap"
readonly POLICY_ENV="$HOME/vla-env"

# Keep model dependencies outside Isaac's Python and the SkyPilot runtime.
sudo apt-get update
sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-venv
python3 -m venv "$POLICY_BOOTSTRAP"
"$POLICY_BOOTSTRAP/bin/python" -m pip install "uv==$POLICY_UV_VERSION"
"$POLICY_BOOTSTRAP/bin/uv" venv --python "$POLICY_PYTHON_VERSION" "$POLICY_ENV"
"$POLICY_BOOTSTRAP/bin/uv" pip install --python "$POLICY_ENV/bin/python" -r rollout-server.requirements.txt
"$POLICY_ENV/bin/python" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"'
