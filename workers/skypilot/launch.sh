#!/usr/bin/env bash
set -euo pipefail

readonly SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SIM_CLUSTER="${SIM_CLUSTER:-isaac-sim}"
readonly SIM_MANIFEST="${SIM_MANIFEST:?Set SIM_MANIFEST to a configured manifest inside ../isaac_sim}"

if [[ "${ACCEPT_EULA:-}" != Y ]]; then
  echo 'Set ACCEPT_EULA=Y after accepting the NVIDIA container license.' >&2
  exit 1
fi

if [[ "$SIM_MANIFEST" == /* || "/$SIM_MANIFEST/" == *"/../"* || ! -f "$SIM_DIR/../isaac_sim/$SIM_MANIFEST" ]]; then
  echo 'SIM_MANIFEST must name an existing relative file inside ../isaac_sim.' >&2
  exit 1
fi

# SkyPilot provisions or reuses this cluster and syncs current worker source.
exec bash "$SIM_DIR/sky.sh" launch "$SIM_DIR/task.yaml" \
  --cluster "$SIM_CLUSTER" --env ACCEPT_EULA=Y \
  --env "SIM_MANIFEST=$SIM_MANIFEST" "$@"
