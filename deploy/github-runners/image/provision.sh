#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
readonly RUNNER_HOME=/opt/actions-runner
readonly UV_ROOT=/opt/uv
readonly BROWSER_ROOT=/opt/playwright
readonly VERSIONS=/tmp/runner-versions.json

# Install native libraries used by existing CPU, browser, and video checks.
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates curl git jq unzip zip xz-utils build-essential pkg-config \
  python3 python3-pip python3-venv ffmpeg libgl1 libglib2.0-0 sudo
useradd --create-home --shell /bin/bash runner
printf 'runner ALL=(ALL) NOPASSWD:ALL\n' > /etc/sudoers.d/runner
chmod 0440 /etc/sudoers.d/runner

# Read Node/pnpm/Playwright pins from the repo instead of maintaining a second set.
readonly NODE_VERSION="$(cat /tmp/node-version)"
readonly PNPM_VERSION="$(jq -r '.packageManager | sub("^pnpm@"; "")' /tmp/workspace-package.json)"
readonly PLAYWRIGHT_VERSION="$(jq -r '.devDependencies["@playwright/test"]' /tmp/workspace-package.json)"
readonly NODE_ARCHIVE="node-v${NODE_VERSION}-linux-x64.tar.xz"
cd /tmp
curl --fail --silent --show-error --location --output "$NODE_ARCHIVE" "https://nodejs.org/dist/v${NODE_VERSION}/${NODE_ARCHIVE}"
curl --fail --silent --show-error --location --output SHASUMS256.txt "https://nodejs.org/dist/v${NODE_VERSION}/SHASUMS256.txt"
grep " ${NODE_ARCHIVE}$" SHASUMS256.txt | sha256sum --check -
tar -xJf "$NODE_ARCHIVE" --strip-components=1 -C /usr/local
npm install --global "pnpm@${PNPM_VERSION}"

# Share interpreter downloads; workflows still create isolated, locked environments.
readonly UV_VERSION="$(jq -r '.uv' "$VERSIONS")"
python3 -m venv "$UV_ROOT/tool"
"$UV_ROOT/tool/bin/pip" install "uv==${UV_VERSION}"
ln -s "$UV_ROOT/tool/bin/uv" /usr/local/bin/uv
ln -s "$UV_ROOT/tool/bin/uvx" /usr/local/bin/uvx
export UV_PYTHON_INSTALL_DIR="$UV_ROOT/python"
uv python install "$(cat /tmp/python-version)"
while IFS= read -r version; do
  uv python install "$version"
done < <(jq -r '.worker_python[]' "$VERSIONS")
chown -R runner:runner "$UV_ROOT/python"

# Preinstall the exact browser revision used by the repository's Playwright package.
mkdir -p /opt/browser-bootstrap
cd /opt/browser-bootstrap
npm install --save-exact "@playwright/test@${PLAYWRIGHT_VERSION}"
PLAYWRIGHT_BROWSERS_PATH="$BROWSER_ROOT" npx playwright install --with-deps --only-shell chromium
chown -R runner:runner "$BROWSER_ROOT"

# Verify the official runner release before installing; register only at VM boot.
readonly RUNNER_VERSION="$(jq -r '.runner' "$VERSIONS")"
readonly RUNNER_SHA="$(jq -r '.runner_sha256' "$VERSIONS")"
mkdir -p "$RUNNER_HOME"
cd "$RUNNER_HOME"
curl --fail --silent --show-error --location --output /tmp/runner.tar.gz \
  "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
printf '%s  /tmp/runner.tar.gz\n' "$RUNNER_SHA" | sha256sum --check -
tar -xzf /tmp/runner.tar.gz
./bin/installdependencies.sh
chown -R runner:runner "$RUNNER_HOME"
install -m 0755 /tmp/runner.sh /usr/local/sbin/firebird-runner
install -m 0644 /tmp/runner.service /etc/systemd/system/firebird-runner.service
systemctl enable firebird-runner.service

# Store a package manifest for provenance; never bake registration or app credentials.
mkdir -p /opt/firebird-image
cp "$VERSIONS" /tmp/workspace-package.json /tmp/node-version /tmp/python-version /opt/firebird-image/
dpkg-query -W > /opt/firebird-image/packages.txt
uv python list --only-installed > /opt/firebird-image/python.txt
node --version > /opt/firebird-image/node.txt
pnpm --version > /opt/firebird-image/pnpm.txt
apt-get clean
rm -f /tmp/runner.tar.gz "/tmp/$NODE_ARCHIVE" /tmp/SHASUMS256.txt
rm -f /root/.bash_history /home/packer/.bash_history /home/packer/.ssh/authorized_keys
