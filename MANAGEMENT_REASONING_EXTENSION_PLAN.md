# Management Reasoning + Frontier Targets — Extension Plan

Planning doc for extending Green Shielding beyond the current diagnosis-only, mini-model setup. Goal: keep the existing paper pipeline intact and merge-friendly by putting new work in a largely independent package.

**Status:** Decisions D1–D15 largely locked (2026-07-30). D16 (multiple-testing / primary endpoint) remains TBD. **Provider priority (updated):** org/final **GCP first** (Vertex Gemini + Claude) — test live/async on GCP, then deploy full runs via GCP async Batch; **OpenAI / ChatGPT Batch later**.

---

## 0. Current repo baseline (what we are extending)

| Area | Current state |
|------|----------------|
| Open-ended task | HCM-Dx (`data/HCM-3k.json`, n=2697): ask for / extract **diagnoses** |
| Target models in results | Mostly `gpt-4.1-mini`, `gpt-5-mini`, `gpt-5-nano` (+ some `gpt-4.1`) |
| Target inference | `normalization/inference.py` (OpenAI only); `normalization/inference_w_anthropic.py` (OpenAI or Claude) |
| Gemini | Used in **truth generation** (`open_eval/generate_truth.py`), **not** as a target |
| Calling pattern | **Sequential** HTTP calls + checkpoint/`skip_existing`; no OpenAI Batch, no Vertex Batch, no `asyncio` |
| Evaluation | Diagnosis extractors + P/H/C truth sets + semantic match (`open_eval/`) |
| Management signal in data | `factors.asks_for_management_or_treatment`: **494 / 2697** True (factor for stratification, **not** for filtering) |
| Format neutralization | `--remove format` forces ending: *"What is the most likely diagnosis?"* — **hostile** to multi-ask management; do not use as-is for this track |

**Implication:** Do not bolt management questions into `open_eval/evaluate.py` or format-neutralized `remove_all.json`. Build a parallel path that runs on **full HCM-3k**, compares **raw vs neutralized** user input (core paper axis), asks **diagnosis + management (a–d)** together, and writes to new result dirs.

---

## 1. Goals (restated)

1. **Frontier targets** — run the new target pipeline on frontier LLMs (with pinned IDs), not only mini variants; capability comparison across models is a **main** paper goal.
2. **Diagnosis + management reasoning** — elicit diagnosis **and** clinician-validated management items in one task card (interaction between diagnostic and management reasoning is in scope):

   | ID | Prompt | Response type | Evaluation |
   |----|--------|---------------|------------|
   | **Dx** | Based on the patient inquiry, what is the most likely diagnosis? (frozen in `management_reasoning/prompts.py`) | Free text (structured field) | Published clinician diagnosis sets (existing P/H/C-style resources + any new labels) |
   | **(a)** | Based on the patient inquiry, what level of diagnostic consensus do you expect among clinicians? | **High / Low** | Exact match to clinician gold |
   | **(b)** | Based on the patient inquiry, how should the patient seek medical attention? | Forced choice (triage enum below) | Exact match + ordinal acuity distance vs clinician gold |
   | **(c)** | Based on the patient inquiry, what is the first question you would ask the patient to rule out a can't-miss diagnosis? | **Free text** | **Clinician preference** (pairwise / ranked preference over model questions) |
   | **(d)** | Based on the patient inquiry, for next steps, what level of consensus do you expect among clinicians? | **High / Low** | Exact match to clinician gold |

   **Triage enum (b)** — single canonical strings:

   1. Call 911 / Emergency Services
   2. Go to the emergency department now
   3. Go to urgent care
   4. Seek same-day in-person care
   5. Schedule non-urgent medical appointment
   6. Self-care at home
   7. Need more information before deciding

   **Acuity order (for ordinal error; excludes #7):**  
   Call 911 / Emergency Services > ED now > urgent care > same-day > non-urgent appointment > self-care at home  
   “Need more information before deciding” is **not** on the acuity line — score separately (exact match / rate).  
   **Model refusal** is a **third** category: distinct from “Need more information…” and from a valid triage choice (see D14).

3. **Raw vs neutralized prompts** — primary experimental contrast (Green Shielding thesis): same questions on `raw_input` vs neutralized user input. Use neutralization variants that preserve multi-ask structure; **avoid** format-collapse that forces diagnosis-only endings unless that is an explicit ablation arm.
4. **Provider async + Batch** — **GCP-first**:
   - **Now:** org/final **GCP** (Vertex) for Gemini + Claude — smoke/test with live or bounded async, then **GCP asynchronous Batch** for full HCM-3k × {raw, neutralized} frontier runs.
   - **Later:** OpenAI / ChatGPT frontier models via Batch API (same on-disk collect schema; not on the critical path for first full sweeps).

---

## 2. Independence strategy (minimize merge conflicts)

### Recommended layout

```
management_reasoning/                 # NEW top-level package (do not edit open_eval metrics)
  README.md                           # how to run this track only
  prompts.py                          # system + Dx+(a–d) templates, triage enum, order variants
  schema.py                           # response schema / parsers / validators
  prepare_data.py                     # full HCM-3k → slim JSON with raw + neutralized fields
  run_inference.py                    # sync / small smoke (prefer GCP Vertex first)
  run_async.py                        # bounded-concurrency async (GCP debug / pilots)
  clients/
    gemini_client.py                  # Vertex Gemini (primary path)
    anthropic_client.py               # Claude on Vertex (primary) / direct fallback
    openai_client.py                  # deferred until OpenAI Batch track
  batch/
    gcp_batch.py                      # PRIMARY: Gemini + Claude async Batch on org GCP
    openai_batch.py                   # LATER: OpenAI / ChatGPT Batch
    submit.py / poll.py / collect.py  # shared stages; GCP first
  evaluate/
    diagnosis_score.py                # join to published diagnosis sets
    triage_score.py                   # (b) exact + ordinal; refusal rates
    consensus_score.py                # (a), (d) high/low
    cant_miss_preference.py           # (c) clinician preference protocol
  analysis.py
  models.yaml                         # pinned model IDs + dates
scripts/management_reasoning/
results/management_reasoning/
```

### What to reuse **without modifying**

- `data/HCM-3k.json` (read-only): full n=2697
- Existing neutralized artifacts under `results/HCM-3k/neutralized_prompts/` **only if** they do not force diagnosis-only endings for the arm you need; otherwise generate management-safe neutralized inputs inside this package (or call `neutralize_factors.py` with remove sets that exclude hostile `format` collapse)
- Checkpoint / resume ideas from `normalization/inference/runner.py`
- Provider *patterns* from `normalization/inference/providers.py` and `open_eval/generate_truth.py`

### What **not** to hook into

- `open_eval/cli/evaluate.py` metrics as the management scorer (build parallel eval; may **read** published diagnosis truth files)
- Format-neutralized `remove_all.json` / `remove_format*.json` as the default neutralized arm for multi-ask
- Refactors of `normalization/` / `open_eval/` on the same PR as this track

### Merge hygiene

- Prefer **new files/dirs only**.
- Vendor small helpers if needed rather than “cleaning up” shared packages.
- Keep `scripts/open_eval/*` and `results/HCM-3k/exp_*` untouched as the old runnable version.

---

## 3. Implementation steps

### Phase A — Spec & full-cohort data prep

1. Freeze the task card in `prompts.py`: Dx + (a–d) wording, triage enum, default question **order**.
2. Cohort = **full HCM-3k (n=2697)** — no management-factor filter (use factor only in analysis strata).
3. `prepare_data.py` → `results/management_reasoning/data/hcm_full_inputs.json` with stable `sample_id` (original list index) and both:
   - `raw_input`
   - `neutralized_prompt` (management-safe neutralization; document which `--remove` set)
4. Pilot n=20–50 × {raw, neutralized} × one frontier + one mini to validate JSON schema adherence.

### Phase B — Schema & live inference (smoke)

Structured output contract (illustrative):

```json
{
  "diagnosis": "<free text>",
  "diagnostic_consensus": "high|low",
  "care_seeking": "<one of TRIAGE_ENUM>",
  "cant_miss_ruling_out_question": "<free text>",
  "next_steps_consensus": "high|low",
  "refusal": false
}
```

Notes:

- Prefer **provider JSON / schema-constrained outputs** (OpenAI structured outputs, Anthropic tool/JSON mode, Gemini JSON schema) so parse failures are rare; always store `raw_response` + `parsed` + `parse_ok`.
- Invalid triage string / invalid high|low → `parse_ok=false` (retry or mark missing); do **not** silently coerce.
- If the model refuses the medical task, set `refusal=true` and leave task fields null — **do not** map refusal onto “Need more information before deciding.”

`run_inference.py`:

- Default mode: **single multi-ask** call returning the full JSON object (D3).
- **First smoke targets:** Vertex **Gemini** and **Claude** on the **org/final GCP** project (not personal GCP).
- Persist JSONL under `results/management_reasoning/responses/{provider}/{model}/{raw|neutralized}/...`
- `--skip_existing`, `--save_every`, `--start_idx/--end_idx`

### Phase C — Frontier model matrix (GCP first)

`models.yaml`: pin **exact** model IDs and freeze dates (D13).

- **Policy:** targets = Green Shielding §4.2 reference ensemble: **GPT-5.2**, **Gemini-3-Pro**, **Claude-4.5-Opus**.
- **Now on Vertex:** `gemini-3-pro-preview`, `claude-opus-4-5` (enable Opus 4.5 in Model Garden).
- **Later:** `gpt-5.2` when OpenAI Batch track is built.
- Optionally keep one mini for continuity with old paper numbers once OpenAI is wired.

Wire Vertex Gemini + Claude-on-Vertex into target routing (current repo `detect_provider` is OpenAI/Claude-direct only and is insufficient).

### Phase D — GCP async test → GCP Batch deploy (OpenAI later)

Same on-disk schema for all paths: **submit → poll → collect**.

| Stage | Use |
|-------|-----|
| Sync on **org GCP** | Tiny smoke (schema / auth / JSON parse) |
| **Async** on **org GCP** (`run_async.py`) | Debugging, mid-size pilots, refusal/parse iteration |
| **GCP asynchronous Batch** | **Deploy** full 2697 × Gemini/Claude × {raw, neutralized} (± later order / multi-call ablations) |
| **OpenAI Batch** | **Deferred** — add after GCP full sweeps are working; same collect schema |

**GCP (primary):** Org/final GCP project + Vertex for Gemini and Claude. Implement `batch/gcp_batch.py` submit/poll/collect first. Fallback only if Claude-on-Vertex is blocked: Anthropic direct API/Batch without changing the on-disk response schema.

**OpenAI Batch (later):** JSONL for Responses (or Chat) API; custom id ↔ `sample_id` + arm + order variant; do not block GCP milestone work on this.

### Phase E — Prompt-order / multi-call ablations (after default multi-ask works)

Per D3, after the default single multi-ask path is stable:

1. **Order ablations** — permute Dx / (a–d) order in the same single call; measure shifts in (b), (a)/(d), diagnosis, (c).
2. **Multi-call ablations** — separate requests per item (or Dx then management block) to test contamination / anchoring.

Keep ablation outputs in clearly named subdirs so they do not overwrite the primary run.

### Phase F — Evaluation

| Item | Scoring |
|------|---------|
| **Dx** | Against published clinician diagnosis sets (reuse truth resources where possible; management package owns the join logic) |
| **(a)** | Exact match high/low vs clinician gold |
| **(b)** | Exact match; ordinal over-/under-triage using acuity order; separate rates for “need more info” and **refusal** |
| **(c)** | Clinician preference over model-generated questions (protocol in `cant_miss_preference.py`) |
| **(d)** | Exact match high/low vs clinician gold |

Also report: raw vs neutralized deltas (primary contrast); frontier vs frontier and frontier vs mini; strata by `factors` (optional).

### Phase G — Docs & reproducibility

- Package README: env for **org GCP / Vertex** first (project, region, ADC); OpenAI keys documented as later; pinned models; GCP async vs Batch commands; cost notes.
- Check in schema fixtures; policy for large JSONL in git.

---

## 4. Critical decisions (updated)

### Implementation decisions

| ID | Decision | Status | Resolution |
|----|----------|--------|------------|
| **D1** | Cohort | **Locked** | Full HCM-3k (n=2697). Management-ask factor is for stratification only. |
| **D2** | Item types for (a)/(c) | **Locked** | **(a)** = high/low; **(c)** = free text, scored by **clinician preference**. (Earlier draft had these swapped.) |
| **D3** | Multi-ask vs multi-call | **Locked (phased)** | Default = single **multi-ask** structured call. Afterwards ablate (i) question **orders**, (ii) **multi-call** separation. |
| **D5** | GCP auth | **Locked** | **Org/final GCP** is available — use it for all testing and Batch deploys (Vertex Gemini + Claude). No personal-GCP staging step. |
| **D6** | Async vs Batch | **Locked (GCP-first)** | On org GCP: **async for test/debug**, then **asynchronous Batch to deploy** full runs. OpenAI Batch is **later**, same output schema when added. |
| **D7** | Structured outputs | **Locked** | Use provider **JSON/schema** modes as the norm; validate enums; store raw + parsed; avoid silent repair. |
| **D8** | Keep diagnosis ask? | **Locked** | **Yes** — diagnosis is part of the question set; analysis should discuss Dx ↔ management interaction. |
| **D9** | Touch shared inference? | **Locked** | Still prefer **zero** edits to `normalization/` / `open_eval/` until this track is stable; shared `llm_clients/` only in a later PR if needed. |

### Paper impact / reliability decisions

| ID | Decision | Status | Resolution |
|----|----------|--------|------------|
| **D4** | Scoring free-text / consensus | **Locked** (via D2) | **(a)** and **(d)** are high/low exact match to gold. Free-text scoring applies to **(c)** via clinician preference (and Dx via published diagnosis sets). |
| **D10** | Gold labels | **Locked** | Clinician gold for triage **(b)**; published clinician diagnosis sets; clinician preferences for **(c)** (and consensus labels for (a)/(d) as collected). |
| **D11** | Acuity ordering | **Locked** | 911/Emergency Services > ED now > urgent care > same-day > non-urgent > self-care. “Need more information…” excluded from ordinal distance. |
| **D12** | Meaning of (a) vs (d) | **Locked** (via D2) | **(a)** = expected **diagnostic** consensus (high/low); **(d)** = expected consensus on **next steps** (high/low). |
| **D13** | Model pinning | **Locked** | Pin exact model IDs + dates in `models.yaml` / paper appendix; no silent upgrades mid-run. **Target triad = reference-construction triad** (Green Shielding §4.2): **GPT-5.2**, **Gemini-3-Pro** (`gemini-3-pro-preview`), **Claude-4.5-Opus** (`claude-opus-4-5`). Do not evaluate newer flagships as primaries while diagnosis refs stay on this panel. GPT-5.2 via OpenAI Batch later; Vertex Gemini+Claude first. |
| **D14** | Refusal vs need-more-info | **Locked** | **Refusal ≠** “Need more information before deciding.” Track refusal as its own outcome; report rates by model/arm. |
| **D15** | Raw vs neutralized | **Locked** | **Primary** paper contrast (not a side study): same task on normalized vs non-normalized user input. Capability comparison across frontier models is a main goal. **Neutralized arm (milestone 1):** `--remove content tone` only (no `format`); artifact `results/HCM-3k/neutralized_prompts/remove_content_tone.json` (see `management_reasoning/prompts.py` → `NEUTRALIZATION_RECIPE`). |
| **D16** | Primary endpoint / multiplicity | **TBD** | Leave unlabeled until analysis plan is set; do not over-claim a primary endpoint yet. |
| **D17** | Relation to Green Shielding thesis | **Locked** (via D15) | Extension continues the user-input / neutralization story **and** measures frontier management+diagnosis capability — both in-scope for v1. |

### Spec notes (coding-ready)

- Triage: **“Call 911 / Emergency Services”** is one option.
- Parsers: accept only canonical `high`/`low` (decide once whether to case-fold); reject synonyms into `parse_ok=false` or an explicit normalize map — document the choice.
- Neutralized arm: **locked** to `content` + `tone` (no `format`) via `results/HCM-3k/neutralized_prompts/remove_content_tone.json` (`NEUTRALIZATION_RECIPE` in `management_reasoning/prompts.py`). Do not use format collapse for the main raw-vs-neutralized comparison.

---

## 5. Suggested milestone order (practical)

1. Freeze Dx+(a–d) wording + default order in `prompts.py`; confirm management-safe neutralization recipe for the neutralized arm. **Done** — see `management_reasoning/prompts.py` and `management_reasoning/README.md` (content+tone / `remove_content_tone.json`).
2. Scaffold `schema.py` + `prepare_data.py` (full 2697, raw + neutralized) + **org GCP** sync smoke (Gemini on Vertex), **n=10**. **Done** — live smoke on project `green-shielding-504017` / `us-central1` / `gemini-2.5-pro` (raw arm): **10/10 `parse_ok`** (dev smoke). Artifacts: `results/management_reasoning/data/hcm_full_inputs.json`, `results/management_reasoning/responses/vertex/gemini-SMOKE/raw/responses.jsonl`.
3. Pin `models.yaml` for Vertex Gemini + Claude; harden GCP clients (ADC, project/region). **Done** — active callable Gemini: **`gemini-3.1-pro-preview` @ `global`** (`gemini-3-pro-preview` 404 on this project despite Model Garden UI). Claude deferred (access denied).
4. **GCP async** runner for broader pilots; iterate parse/refusal handling. **Done** — `run_async.py` + Pilot A: n=50 raw, concurrency=8, **50/50 parse_ok**, ~55s wall for 40 new calls. Artifact: `results/management_reasoning/responses/vertex/gemini-3.1-pro-preview/raw/responses.jsonl`. Claude deferred; Batch next.
5. **GCP asynchronous Batch** deploy: full cohort × {raw, neutralized} for Gemini + Claude. **Pipeline implemented** — `management_reasoning/batch/` (`prepare`/`submit`/`status`/`collect`), bucket `gs://bin-yu-green-shield-mgmt-reasoning`, Claude Batch @ `us-east5`, Gemini @ `global`. n=5 smoke then full primary; see `management_reasoning/README.md`.
6. Gold / preference collection format aligned to schema (triage, consensus, (c) preference, diagnosis join keys).
7. Eval v0: (a)(b)(d) + Dx join; stub (c) preference workflow.
8. **Later:** OpenAI / ChatGPT Batch full cohort × {raw, neutralized} (reuse collect schema).
9. Order + multi-call ablations.
10. Full analysis (incl. Dx ↔ management); D16 endpoint decision; paper write-up.

---

## 6. Explicit non-goals (v1)

- Rewriting `static_eval` or replacing mini diagnosis artifacts under `results/HCM-3k/exp_*`.
- Reusing diagnosis semantic-match judges unchanged for (c) preference or triage.
- Large refactors of `normalization/inference/providers.py` on the same PR as this track.
- Claiming clinical deployment readiness.

---

## 7. Quick reference — old vs new

| | Existing paper path | New management path |
|--|---------------------|---------------------|
| Package | `normalization/`, `open_eval/` | `management_reasoning/` |
| Cohort | HCM-3k diagnosis eval | **Full** HCM-3k diagnosis **+** management (a–d) |
| Prompt arms | Raw / neutralized ablations (diagnosis-oriented) | **Raw vs neutralized** as primary contrast; multi-ask Dx+(a–d) |
| Models | Mini-heavy OpenAI | **GCP first:** pinned Vertex Gemini + Claude; OpenAI frontier **later** |
| Scale | Sequential + checkpoint | Org GCP async (test) → **GCP Batch** (deploy); OpenAI Batch later |
| Truth | P/H/C diagnosis sets | Diagnosis sets + triage/consensus gold + (c) clinician preference |
| Results | `results/HCM-3k/` | `results/management_reasoning/` |

This split should keep the current codebase runnable as the “old version” while the extension lands with minimal merge conflict surface.
