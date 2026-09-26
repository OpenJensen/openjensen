#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo 'Usage: bash watch-rollout.sh RUNNER_ROOT KEY_FILE JOB_ID GROUP_NAME OUTPUT_DIR [--once] [--experimental]' >&2
  exit 2
fi
readonly WATCH_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/monitor.py"
readonly WATCH_ROOT="$(cd "$1" && pwd)"
readonly WATCH_KEY="$2"
readonly WATCH_JOB="$3"
readonly WATCH_GROUP="$4"
readonly WATCH_OUTPUT="$5"
shift 5
readonly WATCH_SKY="$WATCH_ROOT/workers/skypilot"
export SIM_PYTHON="$WATCH_SKY/.venv/bin/python"
source "$WATCH_ROOT/auth.sh" "$WATCH_KEY" || exit 1
unset SKYPILOT_API_SERVER_ENDPOINT SKYPILOT_CONFIG
export SKYPILOT_GLOBAL_CONFIG="$WATCH_SKY/config.yaml"
export SKYPILOT_PROJECT_CONFIG="$WATCH_SKY/config.yaml"
export PATH="$WATCH_SKY/.venv/bin:$PATH"
exec "$SIM_PYTHON" "$WATCH_SCRIPT" --job-id "$WATCH_JOB" --run-id "$WATCH_GROUP" \
  --output-dir "$WATCH_OUTPUT" "$@"
