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

## Not yet implemented

Vertex Batch (full cohort), OpenAI Batch, evaluation / gold labels. Claude async pilots.
