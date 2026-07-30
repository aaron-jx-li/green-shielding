#!/usr/bin/env bash
# n=10 Vertex Claude frontier smoke (raw arm).
# Default = reference-triad Claude-4.5-Opus → claude-opus-4-5.
# Enable that Model Garden card (not a newer Opus) before running.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

INPUT="${INPUT:-./results/management_reasoning/data/hcm_full_inputs.json}"
MODEL="${MODEL:-claude-opus-4-5@20251101}"
# Sanitize path segment (@ is awkward in directories)
MODEL_DIR="${MODEL//@/_}"
OUT_DIR="${OUT_DIR:-./results/management_reasoning/responses/vertex/${MODEL_DIR}/raw}"

if [[ ! -f "$INPUT" ]]; then
  echo "Preparing inputs..."
  python3 management_reasoning/prepare_data.py --out_path "$INPUT"
fi

# Claude works on global and us-east5 via AnthropicVertex.
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"

python3 management_reasoning/run_inference.py \
  --input_path "$INPUT" \
  --out_dir "$OUT_DIR" \
  --provider claude \
  --model "$MODEL" \
  --arm raw \
  --start_idx 0 \
  --end_idx 9 \
  --location "${GOOGLE_CLOUD_LOCATION}" \
  --skip_existing
