#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

enter_repo_root
require_env OPENAI_API_KEY

INPUT_PATH=./data/HCM-3k.json
OUT_DIR=./results/HCM-3k/neutralized_prompts

python normalization/neutralize_factors.py --input_path "${INPUT_PATH}" --output_path "${OUT_DIR}/remove_content.json" --remove content
python normalization/neutralize_factors.py --input_path "${INPUT_PATH}" --output_path "${OUT_DIR}/remove_format.json" --remove format
python normalization/neutralize_factors.py --input_path "${INPUT_PATH}" --output_path "${OUT_DIR}/remove_tone.json" --remove tone
python normalization/neutralize_factors.py --input_path "${INPUT_PATH}" --output_path "${OUT_DIR}/remove_content_format.json" --remove content,format
python normalization/neutralize_factors.py --input_path "${INPUT_PATH}" --output_path "${OUT_DIR}/remove_content_tone.json" --remove content,tone
python normalization/neutralize_factors.py --input_path "${INPUT_PATH}" --output_path "${OUT_DIR}/remove_format_tone.json" --remove format,tone
python normalization/neutralize_factors.py --input_path "${INPUT_PATH}" --output_path "${OUT_DIR}/remove_all.json" --remove content,format,tone
