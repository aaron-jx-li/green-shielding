#!/usr/bin/env bash
# Smoke n=10: ra_new (gpt-5.2 content+format+tone) × Claude+Gemini.
# Submits 4 Batch jobs: 2 legacy free-form dx + 2 independent MR.
# Reuses existing Claude/Gemini raw baselines (not submitted).
#
# Usage:
#   bash ./scripts/management_reasoning/smoke_ra_new_n10.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-bin-yu-green-shield}"
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"

END_IDX="${END_IDX:-9}"
BUCKET="${BUCKET:-bin-yu-green-shield-mgmt-reasoning}"
DX_SUITE="${DX_SUITE:-legacy_dx_ra_new_smoke_n10}"
MR_SUITE="${MR_SUITE:-independent_ra_new_smoke_n10}"

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
  echo "=== prepare+submit suite=$suite end_idx=$END_IDX ==="
  for ja in "${JOBS[@]}"; do
    read -r TARGET ARM <<<"$ja"
    python3 -m management_reasoning.batch prepare \
      --suite "$suite" --provider "$TARGET" --arm "$ARM" \
      --bucket "$BUCKET" --end_idx "$END_IDX"
  done
  for ja in "${JOBS[@]}"; do
    read -r TARGET ARM <<<"$ja"
    python3 -m management_reasoning.batch submit \
      --suite "$suite" --provider "$TARGET" --arm "$ARM" \
      --bucket "$BUCKET"
  done
  python3 - <<PY
import json, os
from management_reasoning.batch.paths import RA_NEW_JOBS, local_manifest_path
suite = os.environ["SUITE_PRINT"]
for t, a in RA_NEW_JOBS:
    p = local_manifest_path(t, a, suite=suite)
    m = json.load(open(p))
    print(f"  {suite} {t}/{a}: {m.get('job_name')}  {m.get('job_state')}")
PY
}

export SUITE_PRINT="$DX_SUITE"
submit_suite "$DX_SUITE"
export SUITE_PRINT="$MR_SUITE"
submit_suite "$MR_SUITE"

cat <<EOF

Submitted 4 smoke jobs (2 dx + 2 MR), end_idx=$END_IDX.

Next:
  SUITE=$DX_SUITE bash ./scripts/management_reasoning/batch_status_collect.sh status
  SUITE=$MR_SUITE bash ./scripts/management_reasoning/batch_status_collect.sh status
  SUITE=$DX_SUITE bash ./scripts/management_reasoning/batch_status_collect.sh collect
  SUITE=$MR_SUITE bash ./scripts/management_reasoning/batch_status_collect.sh collect

Full:
  bash ./scripts/management_reasoning/batch_ra_new_prepare_submit.sh

EOF
