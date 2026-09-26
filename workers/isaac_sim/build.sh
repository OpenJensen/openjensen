#!/usr/bin/env bash
set -euo pipefail

SIM_PROJECT_ID="${SIM_PROJECT_ID:?Set SIM_PROJECT_ID to the target GCP project}"
SIM_REGION="${SIM_REGION:-us-east4}"
SIM_BUILD_SOURCE="${SIM_PROJECT_ID}-sim-build-source"
SIM_BUILD_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SIM_BUILD_DIR"

# Upload this worker only; Cloud Build resolves the Isaac base and tests encoding.
gcloud builds submit . \
  --project="$SIM_PROJECT_ID" \
  --region="$SIM_REGION" \
  --config=cloudbuild.yaml \
  --substitutions="_REGION=$SIM_REGION" \
  --gcs-source-staging-dir="gs://$SIM_BUILD_SOURCE/source" \
  --async \
  --format='value(id)' > build-id.txt

cat build-id.txt
