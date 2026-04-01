#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"
# shellcheck source=./common.sh
source "$SCRIPT_DIR/common.sh"

enter_repo_root
require_env OPENAI_API_KEY

DOCTOR_PATH=./results/HCM-3k/reference/eval_doctor.json
AGREEMENT_DIR=./results/HCM-3k/agreement
EXP_DIR=./results/HCM-3k/exp_4

for i in {1..5}; do
  run_doctor_agreement_once \
    "${DOCTOR_PATH}" \
    "${EXP_DIR}/eval_raw_${i}_gpt-4.1-mini.json" \
    "${AGREEMENT_DIR}/doctor_raw_${i}_gpt-4.1-mini.json"
done

for i in {1..5}; do
  run_doctor_agreement_once \
    "${DOCTOR_PATH}" \
    "${EXP_DIR}/eval_converted_${i}_gpt-4.1-mini.json" \
    "${AGREEMENT_DIR}/doctor_converted_${i}_gpt-4.1-mini.json"
done

for i in {1..5}; do
  run_doctor_agreement_once \
    "${DOCTOR_PATH}" \
    "${EXP_DIR}/eval_raw_${i}_gpt-5-mini.json" \
    "${AGREEMENT_DIR}/doctor_raw_${i}_gpt-5-mini.json"
done

for i in {1..5}; do
  run_doctor_agreement_once \
    "${DOCTOR_PATH}" \
    "${EXP_DIR}/eval_converted_${i}_gpt-5-mini.json" \
    "${AGREEMENT_DIR}/doctor_converted_${i}_gpt-5-mini.json"
done
