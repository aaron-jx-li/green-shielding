#!/usr/bin/env bash
# n=10 Vertex Gemini frontier smoke (raw arm).
# Default = reference-triad Gemini-3-Pro → gemini-3-pro-preview @ global.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

INPUT="${INPUT:-./results/management_reasoning/data/hcm_full_inputs.json}"
MODEL="${MODEL:-gemini-3-pro-preview}"
OUT_DIR="${OUT_DIR:-./results/management_reasoning/responses/vertex/${MODEL}/raw}"

if [[ ! -f "$INPUT" ]]; then
  echo "Preparing inputs..."
  python3 management_reasoning/prepare_data.py --out_path "$INPUT"
fi

# Project resolves from env or gcloud config via run_inference / vertex.get_project.
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"

python3 management_reasoning/run_inference.py \
  --input_path "$INPUT" \
  --out_dir "$OUT_DIR" \
  --provider gemini \
  --model "$MODEL" \
  --arm raw \
  --start_idx 0 \
  --end_idx 9 \
  --location "${GOOGLE_CLOUD_LOCATION}" \
  --skip_existing
