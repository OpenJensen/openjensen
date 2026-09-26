#!/usr/bin/env bash
set -euo pipefail

readonly SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SIM_PYTHON="$SIM_DIR/.venv/bin/python"
readonly SIM_SKY="$SIM_DIR/.venv/bin/sky"
readonly SIM_VERSION="0.13.0"
readonly SIM_PROJECT_ID="${SIM_PROJECT_ID:?Set SIM_PROJECT_ID to your GCP project ID}"
readonly SIM_GCLOUD_BIN="$SIM_DIR/.tools/google-cloud-sdk/bin"

if [[ ! -x "$SIM_SKY" ]]; then
  echo 'Run bash install.sh first.' >&2
  exit 1
fi

"$SIM_PYTHON" - "$SIM_VERSION" <<'PY'
from importlib.metadata import version
import sys
if version('skypilot') != sys.argv[1]:
    raise SystemExit('SkyPilot version differs; rerun install.sh.')
PY

# Isolate task configuration while keeping SkyPilot's cluster state reusable.
unset SKYPILOT_CONFIG
export SKYPILOT_GLOBAL_CONFIG="$SIM_DIR/config.yaml"
export SKYPILOT_PROJECT_CONFIG="$SIM_DIR/config.yaml"
export CLOUDSDK_CORE_PROJECT="$SIM_PROJECT_ID"
export PATH="$SIM_DIR/.venv/bin:$PATH"
if [[ -x "$SIM_GCLOUD_BIN/gcloud" ]]; then
  export PATH="$SIM_GCLOUD_BIN:$PATH"
  export CLOUDSDK_PYTHON="$SIM_PYTHON"
fi
cd "$SIM_DIR"

if [[ "${1:-}" == gcloud ]]; then
  shift
  exec gcloud "$@"
fi

if [[ "${1:-}" == configure ]]; then
  shift
  exec bash "$SIM_DIR/configure.sh" "$@"
fi

check_yaml() {
  if [[ ! -f "$1" ]]; then
    printf 'Missing %s. Copy and edit its .example.yaml template first.\n' "$1" >&2
    exit 1
  fi
  if grep -q 'CHANGE_ME' "$1"; then
    printf 'Replace CHANGE_ME placeholders in %s before running SkyPilot.\n' "$1" >&2
    exit 1
  fi
}

# Refuse incomplete local templates before provisioning or remote execution.
if [[ ! -f "$SIM_DIR/config.yaml" ]]; then
  echo 'Copy config.example.yaml to config.yaml and edit its project ID first.' >&2
  exit 1
fi
check_yaml "$SIM_DIR/config.yaml"
for sim_argument in "$@"; do
  case "$sim_argument" in
    *=*) continue ;;
    *.yaml|*.yml) check_yaml "$sim_argument" ;;
  esac
done

exec "$SIM_SKY" "$@"
