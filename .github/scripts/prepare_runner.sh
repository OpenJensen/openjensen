#!/usr/bin/env bash
set -euo pipefail

readonly CONFIG_ROOT="${RUNNER_CONFIG_ROOT:-/etc}"

# A package-triggered runner restart invokes its VM shutdown hook.
mkdir -p "$CONFIG_ROOT/needrestart/conf.d" "$CONFIG_ROOT/apt/apt.conf.d"
cat > "$CONFIG_ROOT/needrestart/conf.d/firebird-ci.conf" <<'CONF'
$nrconf{override_rc}{qr(^firebird-runner\.service$)} = 0;
CONF

# Disposable workers receive updates when their image is rebuilt.
# Foreground apt installs remain available; never interrupt an active transaction.
cat > "$CONFIG_ROOT/apt/apt.conf.d/99firebird-ci" <<'CONF'
APT::Periodic::Enable "0";
APT::Periodic::Update-Package-Lists "0";
APT::Periodic::Unattended-Upgrade "0";
CONF
systemctl mask --now apt-daily.timer apt-daily-upgrade.timer
