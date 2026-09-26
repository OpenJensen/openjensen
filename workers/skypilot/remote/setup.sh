#!/usr/bin/env bash
set -euo pipefail

SIM_CONTROL_DIR="$(cd "$(dirname "$0")" && pwd)"
SIM_WATCHDOG_DIR=/opt/isaac-watchdog
SIM_PULL_TIMEOUT=1800
SIM_CHECK_TIMEOUT=120
SIM_KILL_GRACE=30
SIM_VULKAN_PACKAGE=libvulkan1
SIM_IMAGE="${SIM_IMAGE:?SIM_IMAGE must name an immutable worker image}"
SIM_CHECK_CONTAINER="isaac-preflight-$$"

_cleanup() {
  docker rm -f "$SIM_CHECK_CONTAINER" >/dev/null 2>&1 || true
}

# Count from VM creation, so reboots and repeated setup cannot extend its lifetime.
SIM_DEADLINE="$(python3 "$SIM_CONTROL_DIR/cloud.py" deadline)"
sudo install -d -m 0755 "$SIM_WATCHDOG_DIR"
sudo install -m 0644 "$SIM_CONTROL_DIR/cloud.py" "$SIM_WATCHDOG_DIR/cloud.py"
sudo tee /etc/systemd/system/isaac-lifetime.service >/dev/null <<'UNIT'
[Unit]
Description=Isaac host lifetime watchdog
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=0

[Service]
Type=oneshot
ExecStart=/usr/bin/python3 /opt/isaac-watchdog/cloud.py delete
Restart=on-failure
RestartSec=60
UNIT
sudo tee /etc/systemd/system/isaac-lifetime.timer >/dev/null <<UNIT
[Unit]
Description=Isaac maximum host lifetime

[Timer]
OnCalendar=$SIM_DEADLINE
Persistent=true
AccuracySec=1
Unit=isaac-lifetime.service

[Install]
WantedBy=timers.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now isaac-lifetime.timer
echo "Host watchdog deadline: $SIM_DEADLINE (requires the VM OS to remain operational)"

# The CUDA host includes NVIDIA graphics drivers but can omit the Vulkan loader.
if [[ "$(dpkg-query -W -f='${Status}' "$SIM_VULKAN_PACKAGE" 2>/dev/null || true)" != "install ok installed" ]]; then
  sudo apt-get update
  sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$SIM_VULKAN_PACKAGE"
fi

python3 "$SIM_CONTROL_DIR/preflight.py"
python3 "$SIM_CONTROL_DIR/cloud.py" login "$SIM_IMAGE"
timeout --kill-after="$SIM_KILL_GRACE" "$SIM_PULL_TIMEOUT" docker pull "$SIM_IMAGE"
trap _cleanup EXIT
timeout --kill-after="$SIM_KILL_GRACE" "$SIM_CHECK_TIMEOUT" \
  docker run --name "$SIM_CHECK_CONTAINER" --rm --gpus all --env NVIDIA_DRIVER_CAPABILITIES=all \
  --entrypoint nvidia-smi "$SIM_IMAGE"
