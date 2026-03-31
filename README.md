# Green Shielding

Code and experiment artifacts for the paper `Green Shielding: A User-Centric Approach Towards Trustworthy AI`.

This repository focuses on the paper's main empirical pipeline:
- static prompt-perturbation experiments on `MedQA` and `MedXpertQA`
- open-ended diagnosis experiments on `HCM-Dx`
- prompt neutralization ablations over `content`, `format`, and `tone`
- evaluation, aggregation, and plotting for the paper figures

## What Is In This Repo

- `data/`
  - `HCM-3k.json`: the main `HCM-Dx` benchmark used in the paper's open-ended experiments
  - `MedQA_ED_diagnosis.json`: static `MedQA` diagnostic benchmark
  - `medxpertqa_diag.json`: static `MedXpertQA` diagnostic benchmark
- `normalization/`
  - `neutralize_factors.py`: rewrites patient-authored prompts while removing selected factor categories
  - `inference.py`: runs an OpenAI model on raw or neutralized prompts
  - `inference_w_anthropic.py`: alternate inference entrypoint for Anthropic models
- `open_eval/`
  - `generate_truth.py`: produces per-model structured diagnosis reference sets
  - `merge_truth.py`: merges multiple reference runs into a majority-vote truth set
  - `evaluate.py`: scores model responses against the merged truth set
  - `llm_doctor.py`: compares model outputs against clinician-style references
  - `analysis.py`, `plot_factor.py`, `score.py`, `extract.py`: supporting analysis scripts
- `static_eval/`
  - static perturbation runner, prompt builders, judging, and CSV aggregation
- `plotting/`
  - scripts that generate the main ablation and factor-frequency figures
- `results/`
  - paper outputs included for reproducibility
- `scripts/`
  - shell wrappers for common experiment stages

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Export API keys before running model-backed experiments:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
```

Most commands below assume you are running from the repo root.

## Data Locations

- Main open-ended benchmark: `data/HCM-3k.json`
- Static benchmarks: `data/MedQA_ED_diagnosis.json` and `data/medxpertqa_diag.json`
- Included paper outputs: `results/HCM-3k/` and `results/static/`

## Example CLI To Replicate Experiments

### 1. Static benchmark perturbation experiments

Run one perturbation setting locally:

```bash
python static_eval/main.py \
  --task medxpertqa_diag \
  --model gpt-4.1-mini \
  --perturbation format_mc \
  --out_csv ./results/static/medxpertqa_mc_gpt-4.1-mini.csv \
  --include_raw \
  --judge_template with_Q \
  --num_runs 5 \
  --temperature 0.7
```

Analyze the resulting CSVs:

```bash
python static_eval/analyze.py --paths ./results/static/*.csv
python static_eval/analyze_bootstrap.py --paths ./results/static/*.csv
```

Helper script:

```bash
bash ./scripts/run_static_benchmarks.sh
```

### 2. Generate neutralized HCM-Dx prompts

Create one ablation condition:

```bash
python normalization/neutralize_factors.py \
  --input_path ./data/HCM-3k.json \
  --output_path ./results/HCM-3k/neutralized_prompts/remove_content.json \
  --remove content
```

Helper script for all main ablation conditions:

```bash
bash ./scripts/run_neutralize_hcm_prompts.sh
```

### 3. Run model inference on raw or neutralized prompts

Run one model on neutralized prompts:

```bash
python normalization/inference.py \
  --input_path ./results/HCM-3k/neutralized_prompts/remove_content.json \
  --output_path ./results/HCM-3k/exp_5/remove_content_4.1-mini.json \
  --model gpt-4.1-mini \
  --mode normalized \
  --num_runs 1
```

Run one model on both raw and neutralized prompts:

```bash
python normalization/inference.py \
  --input_path ./results/HCM-3k/neutralized_prompts/remove_all.json \
  --output_path ./results/HCM-3k/exp_5/remove_all_4.1-mini.json \
  --model gpt-4.1-mini \
  --mode both \
  --num_runs 1
```

### 4. Generate and merge truth sets for open-ended evaluation

Generate one truth file from extracted records:

```bash
python open_eval/generate_truth.py \
  --in ./results/HCM-3k/truth/extracted_gpt-4.1.json \
  --out ./results/HCM-3k/truth/truth_new_gpt-4.1.json \
  --gt_provider openai \
  --gt_model gpt-4.1 \
  --sem_provider openai \
  --sem_model gpt-4.1-mini
```

Merge multiple truth files into the majority-vote reference used by evaluation:

```bash
python open_eval/merge_truth.py \
  --inputs \
    ./results/HCM-3k/truth/truth_new_gpt-5.2.json \
    ./results/HCM-3k/truth/truth_new_gemini-3-pro.json \
    ./results/HCM-3k/truth/truth_new_claude-opus-4-5.json \
  --out ./results/HCM-3k/truth/merged_truth_new.json
```

### 5. Evaluate open-ended model outputs on HCM-Dx

Evaluate one ablation output:

```bash
python open_eval/evaluate.py \
  --input_path ./results/HCM-3k/exp_5/remove_content_4.1-mini.json \
  --pxhx_path ./results/HCM-3k/truth/merged_truth_new.json \
  --output_path ./results/HCM-3k/exp_5/eval_content_1_4.1-mini.json \
  --col_name model_response_neutralized_1 \
  --resume_path ./results/HCM-3k/exp_5/eval_content_1_4.1-mini.json
```

Evaluate a raw baseline response file:

```bash
python open_eval/evaluate.py \
  --input_path ./results/HCM-3k/exp_4/responses_gpt-5-nano.json \
  --pxhx_path ./results/HCM-3k/truth/merged_truth_new.json \
  --output_path ./results/HCM-3k/exp_4/eval_raw_1_gpt-5-nano.json \
  --col_name model_response_raw_1
```

Helper scripts:

```bash
bash ./scripts/run_open_hcm_baseline.sh
bash ./scripts/run_open_hcm_ablations.sh
```

### 6. Doctor agreement analysis

```bash
python open_eval/llm_doctor.py \
  --doctor_path ./results/HCM-3k/reference/eval_doctor.json \
  --llm_path ./results/HCM-3k/exp_4/eval_raw_1_gpt-4.1-mini.json \
  --output_path ./results/HCM-3k/agreement/doctor_raw_1_gpt-4.1-mini.json
```

Helper script:

```bash
bash ./scripts/run_doctor_agreement.sh
```

### 7. Aggregate runs and reproduce plots

Aggregate repeated open-ended evaluation runs:

```bash
python open_eval/analysis.py \
  --results-dir ./results/HCM-3k/exp_4 \
  --pattern "eval_converted_*_gpt-4.1-mini.json" \
  --run-regex "eval_converted_(\\d+)_"
```

Generate the main figures:

```bash
python plotting/ablation.py
python plotting/detect_factors.py \
  --data_path ./results/HCM-3k/neutralized_prompts/remove_all.json \
  --out_path ./factor_frequency.pdf
```

## Notes

- Running the full pipeline from scratch requires paid API access and can be expensive.
- Some final outputs are already checked into `results/` so that figure generation and spot checks can be reproduced without rerunning every upstream step.
- New outputs should be written under `results/HCM-3k/` and `results/static/`.
