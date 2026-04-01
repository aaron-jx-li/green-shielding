# Green Shielding

Research code for **Green Shielding: A User-Centric Approach Towards Trustworthy AI**. This repository reproduces the paper’s empirical pipeline: static prompt perturbations on MedQA and MedXpertQA, open-ended diagnosis on HCM-Dx, prompt neutralization ablations (content, format, tone), evaluation, aggregation, and figure generation.

## Requirements

- Python 3 with dependencies from `requirements.txt`
- API keys for whichever providers you use (OpenAI, Anthropic, Gemini)

```bash
pip install -r requirements.txt
```

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
```

Running the full pipeline uses paid APIs and can be costly. Checked-in artifacts under `results/` support partial reproduction (e.g. plots) without rerunning every step. Write new outputs under `results/HCM-3k/` and `results/static/`.

All example commands below assume the **repository root** as the current working directory.

## Repository layout

| Path | Purpose |
|------|---------|
| `data/` | Benchmarks: `HCM-3k.json` (open-ended), `MedQA_ED_diagnosis.json`, `medxpertqa_diag.json` (static) |
| `normalization/` | Neutralize prompts (`neutralize_factors.py`), run inference (`inference.py`, `inference_w_anthropic.py`); shared logic under `neutralize/`, `inference/`, `io.py` |
| `open_eval/` | Truth generation, merging, evaluation, doctor–LLM comparison, analysis helpers; library code under `core/`, `llm/`, `eval/`, `cli/` |
| `static_eval/` | Static benchmark runs, judging, CSV analysis (`main.py`, `analyze.py`, `analyze_bootstrap.py`); implementation split across `core/`, `llm/`, `pipeline/`, `analysis/`, `cli/` |
| `plotting/` | Scripts for ablation and factor-frequency figures |
| `results/` | Paper outputs for reproducibility |
| `scripts/` | Shell wrappers (`scripts/lib/`, `scripts/static/`, `scripts/normalization/`, `scripts/open_eval/`) |

## Data and outputs

- Open-ended benchmark: `data/HCM-3k.json`
- Static benchmarks: `data/MedQA_ED_diagnosis.json`, `data/medxpertqa_diag.json`
- Bundled results: `results/HCM-3k/`, `results/static/`

## Usage

### 1. Static benchmark (perturbation + judging)

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

```bash
python static_eval/analyze.py --paths ./results/static/*.csv
python static_eval/analyze_bootstrap.py --paths ./results/static/*.csv
```

Optional: `bash ./scripts/static/benchmarks.sh`

### 2. Neutralize HCM-Dx prompts

```bash
python normalization/neutralize_factors.py \
  --input_path ./data/HCM-3k.json \
  --output_path ./results/HCM-3k/neutralized_prompts/remove_content.json \
  --remove content
```

Optional: `bash ./scripts/normalization/neutralize_hcm.sh`

### 3. Inference on raw or neutralized prompts

Neutralized only:

```bash
python normalization/inference.py \
  --input_path ./results/HCM-3k/neutralized_prompts/remove_content.json \
  --output_path ./results/HCM-3k/exp_5/remove_content_4.1-mini.json \
  --model gpt-4.1-mini \
  --mode normalized \
  --num_runs 1
```

Raw and neutralized:

```bash
python normalization/inference.py \
  --input_path ./results/HCM-3k/neutralized_prompts/remove_all.json \
  --output_path ./results/HCM-3k/exp_5/remove_all_4.1-mini.json \
  --model gpt-4.1-mini \
  --mode both \
  --num_runs 1
```

### 4. Open-ended truth sets

Generate a truth file from extracted records:

```bash
python open_eval/generate_truth.py \
  --in ./results/HCM-3k/truth/extracted_gpt-4.1.json \
  --out ./results/HCM-3k/truth/truth_new_gpt-4.1.json \
  --gt_provider openai \
  --gt_model gpt-4.1 \
  --sem_provider openai \
  --sem_model gpt-4.1-mini
```

Merge models into a majority-vote reference:

```bash
python open_eval/merge_truth.py \
  --inputs \
    ./results/HCM-3k/truth/truth_new_gpt-5.2.json \
    ./results/HCM-3k/truth/truth_new_gemini-3-pro.json \
    ./results/HCM-3k/truth/truth_new_claude-opus-4-5.json \
  --out ./results/HCM-3k/truth/merged_truth_new.json
```

### 5. Evaluate open-ended outputs

Ablation column example:

```bash
python open_eval/evaluate.py \
  --input_path ./results/HCM-3k/exp_5/remove_content_4.1-mini.json \
  --pxhx_path ./results/HCM-3k/truth/merged_truth_new.json \
  --output_path ./results/HCM-3k/exp_5/eval_content_1_4.1-mini.json \
  --col_name model_response_neutralized_1 \
  --resume_path ./results/HCM-3k/exp_5/eval_content_1_4.1-mini.json
```

Raw baseline example:

```bash
python open_eval/evaluate.py \
  --input_path ./results/HCM-3k/exp_4/responses_gpt-5-nano.json \
  --pxhx_path ./results/HCM-3k/truth/merged_truth_new.json \
  --output_path ./results/HCM-3k/exp_4/eval_raw_1_gpt-5-nano.json \
  --col_name model_response_raw_1
```

Optional: `bash ./scripts/open_eval/baseline.sh`, `bash ./scripts/open_eval/ablations.sh`

### 6. Doctor vs LLM agreement

```bash
python open_eval/llm_doctor.py \
  --doctor_path ./results/HCM-3k/reference/eval_doctor.json \
  --llm_path ./results/HCM-3k/exp_4/eval_raw_1_gpt-4.1-mini.json \
  --output_path ./results/HCM-3k/agreement/doctor_raw_1_gpt-4.1-mini.json
```

Optional: `bash ./scripts/open_eval/doctor_agreement.sh`

### 7. Aggregate runs and plots

```bash
python open_eval/analysis.py \
  --results-dir ./results/HCM-3k/exp_4 \
  --pattern "eval_converted_*_gpt-4.1-mini.json" \
  --run-regex "eval_converted_(\\d+)_"
```

```bash
python plotting/ablation.py
python plotting/detect_factors.py \
  --data_path ./results/HCM-3k/neutralized_prompts/remove_all.json \
  --out_path ./factor_frequency.pdf
```

## Citation

If you use this repository, please cite the associated *Green Shielding* paper.
