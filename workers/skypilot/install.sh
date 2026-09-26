#!/usr/bin/env bash
set -euo pipefail

readonly SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SIM_PYTHON="${SIM_PYTHON:-python3}"

"$SIM_PYTHON" - <<'PY'
import sys
if not (3, 9) <= sys.version_info[:2] <= (3, 13):
    raise SystemExit('Use Python 3.9–3.13; set SIM_PYTHON to its executable.')
PY

"$SIM_PYTHON" -m venv "$SIM_DIR/.venv"
"$SIM_DIR/.venv/bin/python" -m pip install -r "$SIM_DIR/requirements.txt"
printf 'SkyPilot installed. Authenticate GCP, then run bash sky.sh configure.\n'
