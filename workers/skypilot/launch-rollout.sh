#!/usr/bin/env bash
set -euo pipefail

readonly SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SIM_PYTHON="$SIM_DIR/.venv/bin/python"

if [[ ! -x "$SIM_PYTHON" ]]; then
  echo 'Run bash install.sh first.' >&2
  exit 1
fi

exec "$SIM_PYTHON" "$SIM_DIR/rollout_launch.py" "$@"
