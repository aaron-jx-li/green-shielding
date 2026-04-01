#!/usr/bin/env bash

set -euo pipefail

evaluate_open_eval() {
  local input_path="$1"
  local pxhx_path="$2"
  local output_path="$3"
  local col_name="$4"

  python open_eval/evaluate.py \
    --input_path "${input_path}" \
    --pxhx_path "${pxhx_path}" \
    --output_path "${output_path}" \
    --col_name "${col_name}" \
    --resume_path "${output_path}"
}

run_doctor_agreement_once() {
  local doctor_path="$1"
  local llm_path="$2"
  local output_path="$3"

  python open_eval/llm_doctor.py \
    --doctor_path "${doctor_path}" \
    --llm_path "${llm_path}" \
    --output_path "${output_path}"
}
