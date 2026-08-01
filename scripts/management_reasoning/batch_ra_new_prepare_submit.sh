#!/usr/bin/env bash
# Full ra_new runs: gpt-5.2 content+format+tone × Claude+Gemini.
# 4 Batch jobs: 2 legacy free-form dx + 2 independent MR.
#
# Usage:
#   bash ./scripts/management_reasoning/batch_ra_new_prepare_submit.sh
#   END_IDX=99 bash ./scripts/management_reasoning/batch_ra_new_prepare_submit.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-bin-yu-green-shield}"
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"

BUCKET="${BUCKET:-bin-yu-green-shield-mgmt-reasoning}"
DX_SUITE="${DX_SUITE:-legacy_dx_ra_new}"
MR_SUITE="${MR_SUITE:-independent_ra_new}"
END_IDX="${END_IDX:-}"

echo "Building ra_new inputs…"
python3 -m management_reasoning.prepare_new_neu_inputs --arm ra_new

mapfile -t JOBS < <(python3 - <<'PY'
from management_reasoning.batch.paths import RA_NEW_JOBS
for t, a in RA_NEW_JOBS:
    print(f"{t} {a}")
PY
)

submit_suite() {
  local suite="$1"
  echo "=== prepare+submit suite=$suite end_idx=${END_IDX:-FULL} ==="
  for ja in "${JOBS[@]}"; do
    read -r TARGET ARM <<<"$ja"
    if [[ -n "$END_IDX" ]]; then
      python3 -m management_reasoning.batch prepare \
        --suite "$suite" --provider "$TARGET" --arm "$ARM" \
        --bucket "$BUCKET" --end_idx "$END_IDX"
    else
      python3 -m management_reasoning.batch prepare \
        --suite "$suite" --provider "$TARGET" --arm "$ARM" \
        --bucket "$BUCKET"
    fi
  done
  for ja in "${JOBS[@]}"; do
    read -r TARGET ARM <<<"$ja"
    python3 -m management_reasoning.batch submit \
      --suite "$suite" --provider "$TARGET" --arm "$ARM" \
      --bucket "$BUCKET"
  done
}

submit_suite "$DX_SUITE"
submit_suite "$MR_SUITE"

echo "=== manifests ==="
python3 - <<PY
import json
from management_reasoning.batch.paths import RA_NEW_JOBS, local_manifest_path
for suite in ("$DX_SUITE", "$MR_SUITE"):
    for t, a in RA_NEW_JOBS:
        p = local_manifest_path(t, a, suite=suite)
        m = json.load(open(p))
        print(f"{suite} {t}/{a}: {m.get('job_name')}  {m.get('job_state')}")
PY

cat <<EOF

Next:
  SUITE=$DX_SUITE bash ./scripts/management_reasoning/batch_status_collect.sh collect
  SUITE=$MR_SUITE bash ./scripts/management_reasoning/batch_status_collect.sh collect
  SUITE=legacy_dx_ra_new bash ./scripts/management_reasoning/smoke_eval_ra_new_dx.sh
  SUITE=indep_dx_ra_new bash ./scripts/management_reasoning/smoke_eval_ra_new_indep_dx.sh

EOF
