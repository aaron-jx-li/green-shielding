#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running this script.}"

DOCTOR_PATH=./results/HCM-3k/reference/eval_doctor.json
AGREEMENT_DIR=./results/HCM-3k/agreement
EXP_DIR=./results/HCM-3k/exp_4

for i in {1..5}; do
  python open_eval/llm_doctor.py \
    --doctor_path "${DOCTOR_PATH}" \
    --llm_path "${EXP_DIR}/eval_raw_${i}_gpt-4.1-mini.json" \
    --output_path "${AGREEMENT_DIR}/doctor_raw_${i}_gpt-4.1-mini.json"
done

for i in {1..5}; do
  python open_eval/llm_doctor.py \
    --doctor_path "${DOCTOR_PATH}" \
    --llm_path "${EXP_DIR}/eval_converted_${i}_gpt-4.1-mini.json" \
    --output_path "${AGREEMENT_DIR}/doctor_converted_${i}_gpt-4.1-mini.json"
done

for i in {1..5}; do
  python open_eval/llm_doctor.py \
    --doctor_path "${DOCTOR_PATH}" \
    --llm_path "${EXP_DIR}/eval_raw_${i}_gpt-5-mini.json" \
    --output_path "${AGREEMENT_DIR}/doctor_raw_${i}_gpt-5-mini.json"
done

for i in {1..5}; do
  python open_eval/llm_doctor.py \
    --doctor_path "${DOCTOR_PATH}" \
    --llm_path "${EXP_DIR}/eval_converted_${i}_gpt-5-mini.json" \
    --output_path "${AGREEMENT_DIR}/doctor_converted_${i}_gpt-5-mini.json"
done
