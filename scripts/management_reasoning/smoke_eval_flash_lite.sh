#!/usr/bin/env bash
# Smoke: n=3 full-metric diagnosis eval with gemini-3.1-flash-lite + thinking.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ARM="${ARM:-raw}"
MAX="${MAX:-3}"
MODEL="${MODEL:-gemini-3.1-flash-lite}"
THINKING_LEVEL="${THINKING_LEVEL:-HIGH}"
RESPONSES="${RESPONSES:-./results/management_reasoning/responses/vertex/gemini-3.1-pro-preview/${ARM}_primary_batch/responses.jsonl}"
OUT_DIR="${OUT_DIR:-./results/management_reasoning/eval/gemini-3.1-flash-lite/${ARM}_smoke}"
OUT_PATH="${OUT_PATH:-${OUT_DIR}/eval.json}"
SEM_CACHE="${SEM_CACHE:-${OUT_DIR}/sem_cache.json}"

export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-bin-yu-green-shield}"
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"

mkdir -p "$OUT_DIR"

python3 -m management_reasoning.eval \
  --responses_jsonl "$RESPONSES" \
  --arm "$ARM" \
  --pxhx_path ./results/HCM-3k/truth/merged_truth_new.json \
  --output_path "$OUT_PATH" \
  --model "$MODEL" \
  --thinking_level "$THINKING_LEVEL" \
  --sem_cache_path "$SEM_CACHE" \
  --max "$MAX" \
  --save_every 1 \
  --location "${GOOGLE_CLOUD_LOCATION}"
