#!/usr/bin/env bash
set -euo pipefail

readonly SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SIM_PROJECT_ID="${SIM_PROJECT_ID:?Set SIM_PROJECT_ID to your GCP project ID}"
readonly SIM_REGION="us-east4"
readonly SIM_NETWORK="sim-network"
readonly SIM_SUBNET="sim-us-east4"
readonly SIM_CIDR="10.42.0.0/24"
readonly SIM_SA="skypilot-v1@${SIM_PROJECT_ID}.iam.gserviceaccount.com"
readonly SIM_ROLE="simSkyCompute"
readonly SIM_ROLE_NAME="projects/${SIM_PROJECT_ID}/roles/${SIM_ROLE}"
readonly SIM_FIREWALL="sim-sky-iap-ssh"
readonly SIM_IAP_GROUP="sim-ssh"
readonly SIM_LAUNCHER="${SIM_LAUNCHER:?Set SIM_LAUNCHER to user:YOUR_EMAIL}"
readonly SIM_IAP_ROLE="roles/iap.tunnelResourceAccessor"
readonly SIM_IAP_RANGE="35.235.240.0/20"
readonly SIM_SSH_PORT="22"
readonly SIM_VERSION="0.13.0"

# Keep gcloud calls scoped without editing global configuration.
export CLOUDSDK_CORE_PROJECT="$SIM_PROJECT_ID"

for sim_command in gcloud jq python3; do
  command -v "$sim_command" >/dev/null
done

fail() {
  printf '%s\n' "$1" >&2
  exit 1
}

cloud() {
  gcloud "$@" --project="$SIM_PROJECT_ID" --quiet
}

# Fail before granting permissions if the installed release needs a broader role.
python3 - "$SIM_DIR/role.json" "$SIM_VERSION" <<'PY'
import importlib.metadata
import json
import sys

from sky.clouds.utils import gcp_utils

if importlib.metadata.version("skypilot") != sys.argv[2]:
    raise SystemExit(f"Install skypilot[gcp]=={sys.argv[2]} first.")

with open(sys.argv[1], encoding="utf-8") as stream:
    granted = set(json.load(stream)["includedPermissions"])

required = set(gcp_utils.get_minimal_compute_permissions())
if granted != required:
    raise SystemExit("SkyPilot permissions differ from role.json; review before setup.")
PY

# Check the existing network and same-name resources instead of replacing them.
sim_subnet="$(cloud compute networks subnets describe "$SIM_SUBNET" \
  --region="$SIM_REGION" --format=json)"
jq -e --arg network "$SIM_NETWORK" --arg cidr "$SIM_CIDR" \
  '(.network | endswith("/networks/" + $network)) and .ipCidrRange == $cidr' \
  <<<"$sim_subnet" >/dev/null || fail "Unexpected simulation subnet."

sim_roles="$(cloud iam roles list --show-deleted --format=json)"
sim_role="$(jq -c --arg name "$SIM_ROLE_NAME" \
  '.[] | select(.name == $name)' <<<"$sim_roles")"
if [[ -n "$sim_role" ]]; then
  sim_role="$(cloud iam roles describe "$SIM_ROLE" --format=json)"
  jq -e --slurpfile expected "$SIM_DIR/role.json" \
    '(.deleted // false | not) and .stage == "GA" and
     (.includedPermissions | sort) == ($expected[0].includedPermissions | sort)' \
    <<<"$sim_role" >/dev/null || fail "Existing $SIM_ROLE has different permissions."
fi

sim_policy="$(cloud projects get-iam-policy "$SIM_PROJECT_ID" --format=json)"
jq -e --arg member "serviceAccount:$SIM_SA" --arg role "$SIM_ROLE_NAME" \
  '[.bindings[]? | select(.members | index($member)) |
    select(.role != $role or has("condition"))] | length == 0' \
  <<<"$sim_policy" >/dev/null || fail "Existing SkyPilot account has other project grants; review them first."

sim_rules="$(cloud compute firewall-rules list \
  --filter="name=$SIM_FIREWALL" --format=json)"
if [[ "$(jq length <<<"$sim_rules")" != "0" ]]; then
  jq -e --arg network "$SIM_NETWORK" --arg source "$SIM_IAP_RANGE" \
    --arg account "$SIM_SA" --arg port "$SIM_SSH_PORT" \
    'length == 1 and (.[0] |
      (.network | endswith("/networks/" + $network)) and
      .direction == "INGRESS" and (.disabled // false | not) and
      .sourceRanges == [$source] and .targetServiceAccounts == [$account] and
      .allowed == [{"IPProtocol":"tcp", "ports":[$port]}] and
      ((.denied // []) | length == 0) and
      ((.sourceTags // []) | length == 0) and
      ((.sourceServiceAccounts // []) | length == 0))' \
    <<<"$sim_rules" >/dev/null || fail "Existing $SIM_FIREWALL differs; no firewall changed."
fi

cloud services enable iap.googleapis.com
sim_groups="$(cloud iap tcp dest-groups list --region="$SIM_REGION" --format=json)"
sim_group="$(jq -c --arg name "$SIM_IAP_GROUP" \
  '.[] | select((.name | split("/") | last) == $name)' <<<"$sim_groups")"
if [[ -n "$sim_group" ]]; then
  jq -e --arg cidr "$SIM_CIDR" \
    '.cidrs == [$cidr] and ((.fqdns // []) | length == 0)' \
    <<<"$sim_group" >/dev/null || fail "Existing $SIM_IAP_GROUP differs; no destination group changed."
fi

sim_accounts="$(cloud iam service-accounts list --format=json)"
if ! jq -e --arg email "$SIM_SA" \
  'any(.[]; .email == $email)' <<<"$sim_accounts" >/dev/null; then
  cloud iam service-accounts create skypilot-v1 --display-name="Simulation SkyPilot"
fi

if [[ -z "$sim_role" ]]; then
  cloud iam roles create "$SIM_ROLE" --file="$SIM_DIR/role.json"
fi
cloud projects add-iam-policy-binding "$SIM_PROJECT_ID" \
  --member="serviceAccount:$SIM_SA" --role="$SIM_ROLE_NAME" --condition=None >/dev/null

# The VM can attach only this service account; storage access stays bucket scoped.
cloud iam service-accounts add-iam-policy-binding "$SIM_SA" \
  --member="serviceAccount:$SIM_SA" --role=roles/iam.serviceAccountUser \
  --condition=None >/dev/null
cloud storage buckets add-iam-policy-binding "gs://${SIM_PROJECT_ID}-sim-assets" \
  --member="serviceAccount:$SIM_SA" --role=roles/storage.objectViewer >/dev/null
cloud storage buckets add-iam-policy-binding "gs://${SIM_PROJECT_ID}-sim-results" \
  --member="serviceAccount:$SIM_SA" --role=roles/storage.objectCreator >/dev/null
cloud artifacts repositories add-iam-policy-binding simulation \
  --location="$SIM_REGION" --member="serviceAccount:$SIM_SA" \
  --role=roles/artifactregistry.reader --condition=None >/dev/null

if [[ -z "$sim_group" ]]; then
  cloud iap tcp dest-groups create "$SIM_IAP_GROUP" \
    --region="$SIM_REGION" --ip-range-list="$SIM_CIDR"
fi
# IP-based tunnels authorize the operator against the destination group.
cloud iap tcp dest-groups add-iam-policy-binding \
  --dest-group="$SIM_IAP_GROUP" --region="$SIM_REGION" \
  --member="$SIM_LAUNCHER" --role="$SIM_IAP_ROLE" --condition=None >/dev/null
if [[ "$(jq length <<<"$sim_rules")" == "0" ]]; then
  cloud compute firewall-rules create "$SIM_FIREWALL" \
    --network="$SIM_NETWORK" --direction=INGRESS --action=ALLOW \
    --rules="tcp:$SIM_SSH_PORT" --source-ranges="$SIM_IAP_RANGE" \
    --target-service-accounts="$SIM_SA"
fi

printf 'SkyPilot network and identity configured. No VM created.\n'
