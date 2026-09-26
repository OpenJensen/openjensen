#!/usr/bin/env bash
set -euo pipefail

readonly SIM_CONTROL_DIR="$(cd "$(dirname "$0")" && pwd)"
readonly SIM_CHECK_CONTAINER="isaac-ready-$$"
readonly SIM_SOURCE=/opt/sim-worker
readonly SIM_CONTROL=/opt/sim-control
readonly SIM_KILL_GRACE=30
readonly SIM_POLICY_PORT=8080

_cleanup() {
  docker rm -f "$SIM_CHECK_CONTAINER" >/dev/null 2>&1 || true
}
trap _cleanup EXIT

# Discover the matching policy VM's private IP; never load or move the robot.
SIM_POLICY_IP="$(python3 "$SIM_CONTROL_DIR/cloud.py" discover-policy "$ROLLOUT_ID" \
  --timeout "$POLICY_DISCOVERY_TIMEOUT")"
nvidia-smi
docker run --name "$SIM_CHECK_CONTAINER" --rm --init --gpus all --network host \
  --env ACCEPT_EULA=Y --env NVIDIA_DRIVER_CAPABILITIES=all \
  --env "PYTHONPATH=$SIM_SOURCE" \
  --mount "type=bind,src=$PWD,dst=$SIM_SOURCE,readonly" \
  --mount "type=bind,src=$SIM_CONTROL_DIR,dst=$SIM_CONTROL,readonly" \
  --workdir "$SIM_SOURCE" --entrypoint /usr/bin/timeout "$SIM_IMAGE" \
  --signal=TERM --kill-after="$SIM_KILL_GRACE" "$SIM_READY_TIMEOUT" \
  /isaac-sim/python.sh --no-ros-env "$SIM_CONTROL/check_policy.py" \
  --manifest "$SIM_SOURCE/$SIM_MANIFEST" \
  --endpoint "http://$SIM_POLICY_IP:$SIM_POLICY_PORT"
