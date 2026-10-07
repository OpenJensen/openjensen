#!/usr/bin/env bash
# Hosted images use an Azure mirror that can stall package setup indefinitely.
set -euo pipefail

if [[ -f /etc/apt/apt-mirrors.txt ]]; then
  sed -i 's|http://azure.archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' /etc/apt/apt-mirrors.txt
fi
# Keep a mirror outage bounded instead of consuming the browser job's budget.
cat > /etc/apt/apt.conf.d/99-firebird-ci-timeouts <<'APT'
Acquire::Retries "1";
Acquire::http::Timeout "30";
Acquire::https::Timeout "30";
APT
