#!/usr/bin/env bash
# Independent MR (dx+a–d) × new_neu content+tone (ct_old + ct_new).
# Claude/Gemini raw independent are reused from prior runs — not submitted here.
#
# Usage:
#   bash ./scripts/management_reasoning/batch_independent_new_neu_prepare_submit.sh
#   END_IDX=19 bash ./scripts/management_reasoning/batch_independent_new_neu_prepare_submit.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-bin-yu-green-shield}"
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"

export SUITE="${SUITE:-independent_new_neu}"
export BUCKET="${BUCKET:-bin-yu-green-shield-mgmt-reasoning}"

echo "Building new_neu ct_old/ct_new inputs…"
python3 -m management_reasoning.prepare_new_neu_inputs

bash ./scripts/management_reasoning/batch_prepare_submit.sh
