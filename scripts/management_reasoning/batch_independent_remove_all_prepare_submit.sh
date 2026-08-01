#!/usr/bin/env bash
# Independent MR (dx+a–d) × remove_all user text (+ Gemini raw).
# Claude raw independent is reused from prior run — not submitted here.
#
# Usage:
#   bash ./scripts/management_reasoning/batch_independent_remove_all_prepare_submit.sh
#   END_IDX=19 bash ./scripts/management_reasoning/batch_independent_remove_all_prepare_submit.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-bin-yu-green-shield}"
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"

export SUITE="${SUITE:-independent_remove_all}"
export INPUT="${INPUT:-./results/management_reasoning/data/hcm_legacy_diag_inputs.json}"
export BUCKET="${BUCKET:-bin-yu-green-shield-mgmt-reasoning}"

if [[ ! -f "$INPUT" ]]; then
  echo "Building legacy/remove_all inputs…"
  python3 -m management_reasoning.prepare_legacy_diag_data
fi

bash ./scripts/management_reasoning/batch_prepare_submit.sh
