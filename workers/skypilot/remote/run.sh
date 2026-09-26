#!/usr/bin/env bash
set -euo pipefail

SIM_CONTROL_DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$SIM_CONTROL_DIR/runner.py"
