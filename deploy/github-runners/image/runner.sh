#!/usr/bin/env bash
set -euo pipefail
readonly METADATA=http://metadata.google.internal/computeMetadata/v1/instance/attributes

# No registration during image creation. Never print the single-use JIT secret.
jit=$(curl --fail --silent --show-error --max-time 20 \
  --retry 5 --retry-delay 2 --header 'Metadata-Flavor: Google' "$METADATA/runner-jit")
cd /opt/actions-runner
exec ./run.sh --jitconfig "$jit"
