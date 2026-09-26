#!/usr/bin/env bash
# Source this file so authentication variables remain available to SkyPilot.
if [[ -z "${BASH_VERSION:-}" ]]; then
  printf 'Use Bash, then source auth.sh /absolute/path/to/key.json\n' >&2
  return 1 2>/dev/null || exit 1
fi
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf 'Source this helper: source auth.sh /absolute/path/to/key.json\n' >&2
  exit 1
fi

_sim_auth() {
  # Sky's wrapper must refuse commands if this authentication attempt fails.
  unset SIM_PROJECT_ID
  local sim_project='project-5693e83a-db3a-43e1-98c'
  local sim_account="sim-rollout-runner@$sim_project.iam.gserviceaccount.com"
  local sim_python="${SIM_PYTHON:-python3}"
  local sim_repo sim_key sim_state sim_active
  sim_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || return 1
  sim_state="${SIM_AUTH_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/isaac-act-runner}"

  if [[ $# != 1 ]]; then
    printf 'Usage: source auth.sh /absolute/path/to/key.json\n' >&2
    return 1
  fi
  if ! command -v gcloud >/dev/null; then
    printf 'Install Google Cloud CLI and put gcloud on PATH first.\n' >&2
    return 1
  fi
  sim_key="$("$sim_python" - "$1" "$sim_account" "$sim_project" "$sim_repo" <<'PY'
import json
import sys
from pathlib import Path

key = Path(sys.argv[1]).expanduser().resolve(strict=True)
repo = Path(sys.argv[4]).resolve()
if key.is_relative_to(repo):
    raise SystemExit('Keep the service-account key outside this repository.')
values = json.loads(key.read_text())
if (
    values.get('type') != 'service_account'
    or values.get('client_email') != sys.argv[2]
    or values.get('project_id') != sys.argv[3]
    or not values.get('private_key')
):
    raise SystemExit('Supply the dedicated sim-rollout-runner service-account key.')
print(key)
PY
  )" || return 1

  # Use a dedicated gcloud profile and the same service-account key for ADC.
  mkdir -p "$sim_state/gcloud" || return 1
  chmod 700 "$sim_state" "$sim_state/gcloud" || return 1
  export CLOUDSDK_CONFIG="$sim_state/gcloud"
  unset CLOUDSDK_ACTIVE_CONFIG_NAME
  export CLOUDSDK_CORE_ACCOUNT="$sim_account"
  export CLOUDSDK_CORE_PROJECT="$sim_project"
  export GOOGLE_APPLICATION_CREDENTIALS="$sim_key"
  export GOOGLE_CLOUD_PROJECT="$sim_project"
  unset CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT
  unset CLOUDSDK_AUTH_ACCESS_TOKEN CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE

  # A forced profile name prevents gcloud from lazily creating its first config.
  if [[ ! -f "$CLOUDSDK_CONFIG/configurations/config_default" ]]; then
    gcloud config configurations list --format='value(name)' >/dev/null || return 1
  fi
  if [[ ! -f "$CLOUDSDK_CONFIG/configurations/config_default" ]]; then
    printf 'Could not initialize the dedicated gcloud configuration.\n' >&2
    return 1
  fi
  export CLOUDSDK_ACTIVE_CONFIG_NAME=default
  gcloud auth activate-service-account "$sim_account" \
    --key-file="$sim_key" --project="$sim_project" --quiet || return 1
  gcloud config set project "$sim_project" --quiet || return 1
  sim_active="$(gcloud auth list --filter=status:ACTIVE --format='value(account)')" || return 1
  if [[ "$sim_active" != "$sim_account" ]]; then
    printf 'Dedicated service-account authentication was not confirmed.\n' >&2
    return 1
  fi
  export SIM_PROJECT_ID="$sim_project"
  printf 'Authenticated %s. Keep using this Bash session.\n' "$sim_account"
}

if _sim_auth "$@"; then
  unset -f _sim_auth
else
  unset -f _sim_auth
  return 1
fi
