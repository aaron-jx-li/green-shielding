# Management reasoning track

Independent extension of Green Shielding for **diagnosis + management reasoning** on full HCM-3k. Keeps `normalization/` / `open_eval/` untouched.

## Provider priority

**Org GCP Vertex** (`bin-yu-green-shield`): Gemini + Claude (AnthropicVertex). Async pilots → Batch full cohort. OpenAI later.

## Active targets

| | Value |
|--|--------|
| **Gemini** | `gemini-3.1-pro-preview` @ `global` (`google.genai`) |
| Preferred triad Gemini | `gemini-3-pro-preview` (404 previously; recheck) |
| **Claude** | `claude-opus-4-5@20251101` @ `global` (`AnthropicVertex`) |
| Dev smoke | `gemini-2.5-pro` |

See [`models.yaml`](models.yaml).

## Milestones 1–3 (done)

- Task card: [`prompts.py`](prompts.py)
- Inputs: `results/management_reasoning/data/hcm_full_inputs.json` (n=2697)
- Sync smokes + Vertex config (gcloud project fallback)

## Milestone 4 (done) — async Gemini

[`run_async.py`](run_async.py): bounded concurrency, retries, JSONL checkpoint, summary stats.

**Pilot A** (2026-07-30): n=50 raw, concurrency=8, `gemini-3.1-pro-preview` @ `global`

- **50/50 parse_ok**, 0 refusals
- Wall ~**55 s** for 40 new calls (10 already present from sync smoke; full 0–49 covered)
- Output: `results/management_reasoning/responses/vertex/gemini-3.1-pro-preview/raw/responses.jsonl`

**Pilot B reword v2 `(a)+(d)`** (2026-07-30): tighter diagnostic consensus (leading + can't-miss) + management next-steps `(d)`; n=50 both arms

- Artifacts: `raw_ad_reword_v2/`, `neutralized_ad_reword_v2/`
- `diagnostic_consensus`: raw 26H/24L; neut 23H/27L (similar to v1)
- `next_steps_consensus`: raw 32H/18L; neut 34H/16L
- Arm flips: care_seeking **7/50**, diagnostic_consensus **11/50**, next_steps_consensus **8/50**

**Pilot B reword v3 `(a)+(d)` shortened** (2026-07-30): shorter QUESTION_TEXT for diagnostic consensus + next-steps; n=50 both arms, Gemini + Claude

- Artifacts: `raw_ad_reword_v3/`, `neutralized_ad_reword_v3/` (per model)
- Gemini: diag raw 26H/24L → neut 19H/31L; next raw 33H/17L → neut 29H/21L; flips care **11**, diag **15**, next **12**
- Claude: diag raw 31H/19L → neut 24H/26L; next raw 32H/18L → neut 23H/27L; flips care **9**, diag **13**, next **15**
- All arms **50/50 parse_ok**, 0 refusals

```bash
export GOOGLE_CLOUD_LOCATION=global
bash ./scripts/management_reasoning/pilot_async_gemini.sh
# Pilot B (neutralized):
ARM=neutralized OUT_DIR=./results/management_reasoning/responses/vertex/gemini-3.1-pro-preview/neutralized \
  bash ./scripts/management_reasoning/pilot_async_gemini.sh
```

## GCP setup

```bash
gcloud auth application-default login
gcloud config set project bin-yu-green-shield
export GOOGLE_CLOUD_LOCATION=global
```

ADC needs **Vertex AI** + **Cloud Storage** on this project.

Batch I/O bucket (created): `gs://bin-yu-green-shield-mgmt-reasoning`

## Milestone 5 — Vertex Batch (primary suite)

Package: [`management_reasoning/batch/`](batch/) — `prepare` → upload → `submit` → (disconnect OK) → `status` → `collect`.

Primary = **4 jobs**: Gemini + Claude × raw + neutralized, n=2697, default multi-ask.  
Actual Batch cost ~**$61**; suite wall ~**16 min**.

**Important locations**

| | |
|--|--|
| Gemini Batch | `global` |
| Claude Batch | **`us-east5`** (global endpoint rejects Claude Batch) |
| Local manifests | `results/management_reasoning/batch/{suite}/{provider}/{arm}/` |
| Local responses | `results/management_reasoning/responses/vertex/{model}/{arm}_{suite}_batch/` |

```bash
# Full primary
bash ./scripts/management_reasoning/batch_prepare_submit.sh
bash ./scripts/management_reasoning/batch_status_collect.sh status
bash ./scripts/management_reasoning/batch_status_collect.sh collect
```

## Milestone 6 — Order + independent ablations (Batch)

Frozen in [`prompts.py`](prompts.py):

| Suite | Meaning |
|-------|---------|
| `order_ord1` | multi-ask order `b → dx → c → a → d` |
| `order_ord2` | multi-ask order `a → d → b → c → dx` |
| `order_ord3` | multi-ask order `dx → d → c → b → a` |
| `independent` | 5 separate calls per sample (dx, a, b, c, d) |

Default ablation grid: **Claude × {raw, neutralized}** (~$11/arm multi-ask; independent ≈5×).

```bash
# n=5 Claude smoke for one order + independent
SUITE=order_ord1 END_IDX=4 PROVIDER=claude ARM=raw \
  bash ./scripts/management_reasoning/batch_prepare_submit.sh
SUITE=independent END_IDX=4 PROVIDER=claude ARM=raw \
  bash ./scripts/management_reasoning/batch_prepare_submit.sh

# Full Claude both arms (example)
SUITE=order_ord1 bash ./scripts/management_reasoning/batch_prepare_submit.sh
SUITE=order_ord2 bash ./scripts/management_reasoning/batch_prepare_submit.sh
SUITE=order_ord3 bash ./scripts/management_reasoning/batch_prepare_submit.sh
SUITE=independent bash ./scripts/management_reasoning/batch_prepare_submit.sh
```

## Milestone 7 — Legacy diagnosis-only isolation (`legacy_diag`)

**Goal:** run frontier Claude on the **paper** diagnosis-only protocol (isolate model vs MR prompt/schema).

| Knob | Value |
|------|--------|
| Model | Claude Opus 4.5 Batch @ `us-east5` |
| Arms | **raw** + **remove_all** (not MR content+tone) |
| System | Old mini-model instruction (`LEGACY_DIAG_INSTRUCTION`) |
| User | Inquiry as-is (no `Patient inquiry:` / no JSON schema) |
| Temperature | **0.7** set on Claude Batch requests (accepted on smoke) |
| Target thinking | Off (no thinking_config) |
| Judge | Flash-Lite Batch on **full free-form** answer |
| Inputs | `results/management_reasoning/data/hcm_legacy_diag_inputs.json` |
| Responses | `results/HCM-3k/exp_frontier/claude-opus-4-5_20251101/{raw,remove_all}_legacy_batch/` |

```bash
# Build inputs (once)
python3 -m management_reasoning.prepare_legacy_diag_data

# Smoke n=20 then full
END_IDX=19 bash ./scripts/management_reasoning/batch_legacy_diag_prepare_submit.sh
bash ./scripts/management_reasoning/batch_legacy_diag_prepare_submit.sh

SUITE=legacy_diag PROVIDER=claude bash ./scripts/management_reasoning/batch_status_collect.sh status
SUITE=legacy_diag PROVIDER=claude bash ./scripts/management_reasoning/batch_status_collect.sh collect

# Flash-Lite judge (full free-form answer_mode)
python3 -m management_reasoning.eval.batch prepare --stage extract --suite legacy_diag
python3 -m management_reasoning.eval.batch prepare --stage unc --suite legacy_diag
python3 -m management_reasoning.eval.batch submit --stage extract --suite legacy_diag
python3 -m management_reasoning.eval.batch submit --stage unc --suite legacy_diag
# … status/collect, then sem+ground, then:
python3 -m management_reasoning.eval.batch aggregate --suite legacy_diag

# Compare table + radar (after aggregate)
python3 plotting/compare_legacy_diag_metrics.py
python3 plotting/management_reasoning_radar.py --panel legacy
```

**Do not confuse** `remove_all` with MR primary `neutralized` (content+tone only).

## Milestone 7b — Independent × remove_all (`independent_remove_all`)

MR single-question card (dx+a–d), `Patient inquiry:` wrapper, paper **remove_all** user text.

| Job | Notes |
|-----|--------|
| Claude × remove_all | New |
| Gemini × raw | New (no prior Gemini independent) |
| Gemini × remove_all | New |
| Claude × raw | **Reuse** `raw_independent_batch` |

```bash
END_IDX=19 bash ./scripts/management_reasoning/batch_independent_remove_all_prepare_submit.sh
bash ./scripts/management_reasoning/batch_independent_remove_all_prepare_submit.sh
SUITE=independent_remove_all bash ./scripts/management_reasoning/batch_status_collect.sh status
SUITE=independent_remove_all bash ./scripts/management_reasoning/batch_status_collect.sh collect

# Flip plots (after collect)
python3 -m management_reasoning.analysis.plot_independent_remove_all
```

## Diagnosis eval (management_reasoning track)

Separate from `open_eval/` OpenAI judges. Package: [`management_reasoning/eval/`](eval/).

- Pulls `parsed.diagnosis` from target JSONL (join by `sample_id`)
- Same radar metrics: Plausibility, H-coverage, S-coverage, Breadth, Evidence, Inference, Uncertainty
- Judge: **`gemini-3.1-flash-lite`** @ `global` with **thinking** (`HIGH` default)
- Does not edit `open_eval/`; only imports pure aggregators from `open_eval.eval.metrics`

**Sync smoke (online judges):**

```bash
bash ./scripts/management_reasoning/smoke_eval_flash_lite.sh
```

**Vertex Batch judges (multi-stage):** [`eval/batch/`](eval/batch/)

| Wave | Stages | Depends on |
|------|--------|------------|
| 1 | `extract`, `unc` | target `parsed.diagnosis` only |
| 2 | `sem`, `ground` | wave-1 extract collect |
| 3 | `aggregate` | all collects |

```bash
# n=3 smoke (prepare→submit→poll→collect→aggregate)
bash ./scripts/management_reasoning/smoke_eval_batch.sh
# Full primary grid (after smoke): repeat prepare/submit per stage for all 4 arms
python -m management_reasoning.eval.batch prepare --stage extract --suite primary --end_idx 2696
python -m management_reasoning.eval.batch submit --stage extract --suite primary
# ... status / collect, then sem+ground, then aggregate
```

## Not yet implemented

OpenAI Batch; management-specific gold labels.
