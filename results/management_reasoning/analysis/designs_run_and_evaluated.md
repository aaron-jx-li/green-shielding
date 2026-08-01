# Management reasoning — designs run and evaluated

Summary of neutralization arms, protocols, and Flash-Lite diagnosis eval coverage on HCM-3k (**n=2697**).  
Targets: **Claude** `claude-opus-4-5@20251101`, **Gemini** `gemini-3.1-pro-preview`.  
Judge: **Gemini 3.1 Flash-Lite** (extract → unc → sem → ground → aggregate).

Date of this inventory: 2026-08-01.

---

## 1. Arms (user-text designs)

| Arm id | What it removes | Source artifact | Notes |
|--------|-----------------|-----------------|-------|
| `raw` | — (baseline inquiry) | HCM cohort inputs | Reused across protocols |
| `neutralized` | content + tone (MR primary) | MR neutralized primary inputs | Primary multi-ask only |
| `remove_all` | content + format + tone (**paper**) | `results/HCM-3k/neutralized_prompts/remove_all.json` | Paper-style full neutralize |
| `format_tone` | format + tone | `…/remove_format_tone.json` | Paper 2-factor pair |
| `content_format` | content + format | `…/remove_content_format.json` | Paper 2-factor pair |
| `ct_old` | content + tone | `results/new_neu/gpt-5.2_old_remove_content_tone.json` | gpt-5.2 “old” phrasing |
| `ct_new` | content + tone | `results/new_neu/gpt-5.2_new_remove_content_tone.json` | gpt-5.2 “new” phrasing |
| `ra_new` | content + format + tone (**gpt-5.2**) | `results/new_neu/gpt-5.2_old_remove_content_format_tone.json` | New-model full neutralize; distinct from paper `remove_all` |

---

## 2. Protocols

| Protocol | Suite name(s) | Prompt style | Output |
|----------|---------------|--------------|--------|
| **Primary multi-ask** | `primary` | MR task card; one call, multi-field JSON | dx + a–d in one response |
| **Order ablation** | `order_ord1`–`ord3` | Same multi-ask, permuted field order | Claude × raw only (full) |
| **Independent MR** | `independent*`, `independent_remove_all`, `independent_new_neu`, `independent_factor`, `independent_ra_new` | 5 separate calls (dx, a, b, c, d); `Patient inquiry:` wrapper | Per-question JSON |
| **Legacy free-form diagnosis** | `legacy_diag`, `legacy_dx`, `legacy_dx_factor`, `legacy_dx_ra_new` | Paper system (`LEGACY_DIAG_INSTRUCTION`), temp **0.7**, free-form answer (no JSON schema) | Full-text diagnosis |

Flash-Lite on **legacy** judges the full free-form answer; on **indep_dx** / **primary** it judges the diagnosis field / parsed diagnosis.

---

## 3. Coverage matrix

Legend: **Gen** = target Batch collected (full n≈2697); **FL** = Flash-Lite diagnosis metrics aggregated; **Flip** = independent MR flip analysis CSV exists (card fields vs raw).

### 3a. Legacy free-form diagnosis

| Arm | Claude Gen | Claude FL | Gemini Gen | Gemini FL |
|-----|------------|-----------|------------|-----------|
| `raw` | ✓ (`legacy_diag`) | ✓ `claude_raw_legacy_diag` | ✓ (`legacy_dx`) | ✓ `gemini_raw_legacy_dx` |
| `remove_all` | ✓ | ✓ `claude_remove_all_legacy_diag` | ✓ | ✓ `gemini_remove_all_legacy_dx` |
| `format_tone` | ✓ | ✓ | ✓ | ✓ |
| `content_format` | ✓ | ✓ | ✓ | ✓ |
| `ct_old` | ✓ | ✓ | ✓ | ✓ |
| `ct_new` | ✓ | ✓ | ✓ | ✓ |
| `ra_new` | ✓ | ✓ | ✓ | ✓* |

\*Gemini `ra_new` legacy generate: **2597/2697** parse_ok (100 errors); FL aggregate still reports n=2697 after join.

### 3b. Independent MR (dx card + flips)

| Arm | Claude Gen | Claude FL (dx) | Claude Flip | Gemini Gen | Gemini FL (dx) | Gemini Flip |
|-----|------------|----------------|-------------|------------|----------------|-------------|
| `raw` | ✓ | ✓ | baseline | ✓ | ✓ | baseline |
| `remove_all` | ✓ | ✓ | ✓ | ✓ | — | ✓ |
| `ct_old` / `ct_new` | ✓ | ✓ | ✓ | ✓ | — | ✓ |
| `format_tone` / `content_format` | ✓ | — | ✓ | ✓ | — | ✓ |
| `ra_new` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `neutralized` | — (not on indep grid) | — | — | — | — | — |

### 3c. Primary multi-ask

| Arm | Claude Gen | Claude FL | Gemini Gen | Gemini FL |
|-----|------------|-----------|------------|-----------|
| `raw` | ✓ | ✓ `claude_raw_primary` | ✓ | ✓ `gemini_raw_primary` |
| `neutralized` | ✓ | ✓ | ✓ | ✓ |

### 3d. Order ablations (generated only)

| Suite | Claude × raw Gen | Flash-Lite |
|-------|------------------|------------|
| `order_ord1` / `ord2` / `ord3` | ✓ (n=2697 each) | — |

---

## 4. Flash-Lite metrics (full aggregates)

Metrics from `results/management_reasoning/eval/gemini-3.1-flash-lite/*/eval.json` (non-smoke).  
**B** = mean normalized breadth, **P** = plausibility, **H** / **C** = H- / C-coverage.

### Legacy free-form

| Model | Arm | B | P | H | C |
|-------|-----|---|---|---|---|
| Claude | raw | 0.747 | 0.779 | 0.695 | 0.299 |
| Claude | remove_all | 0.847 | 0.774 | 0.672 | 0.404 |
| Claude | format_tone | 0.844 | 0.773 | 0.670 | 0.396 |
| Claude | content_format | 0.779 | 0.803 | 0.702 | 0.346 |
| Claude | ct_old | 0.844 | 0.774 | 0.697 | 0.372 |
| Claude | ct_new | 0.844 | 0.773 | 0.695 | 0.368 |
| Claude | ra_new | 0.839 | 0.771 | 0.663 | 0.405 |
| Gemini | raw | 0.794 | 0.764 | 0.712 | 0.357 |
| Gemini | remove_all | 0.639 | 0.783 | 0.583 | 0.306 |
| Gemini | format_tone | 0.656 | 0.775 | 0.594 | 0.319 |
| Gemini | content_format | 0.743 | 0.794 | 0.683 | 0.360 |
| Gemini | ct_old | 0.855 | 0.768 | 0.699 | 0.393 |
| Gemini | ct_new | 0.857 | 0.761 | 0.707 | 0.397 |
| Gemini | ra_new | 0.596 | 0.768 | 0.544 | 0.281 |

### Independent diagnosis field

| Model | Arm | B | P | H | C |
|-------|-----|---|---|---|---|
| Claude | raw | 0.404 | 0.883 | 0.554 | 0.181 |
| Claude | remove_all | 0.341 | 0.874 | 0.471 | 0.163 |
| Claude | ct_old | 0.400 | 0.868 | 0.530 | 0.177 |
| Claude | ct_new | 0.404 | 0.867 | 0.530 | 0.186 |
| Claude | ra_new | 0.334 | 0.873 | 0.456 | 0.157 |
| Gemini | raw | 0.176 | 0.941 | 0.405 | 0.120 |
| Gemini | ra_new | 0.154 | 0.877 | 0.327 | 0.121 |

*(Gemini `remove_all` / CT / factor-pair indep_dx: responses collected; Flash-Lite not run.)*

### Primary multi-ask (parsed diagnosis)

| Model | Arm | B | P | H | C |
|-------|-----|---|---|---|---|
| Claude | raw | 0.524 | 0.856 | 0.637 | 0.262 |
| Claude | neutralized | 0.540 | 0.855 | 0.625 | 0.271 |
| Gemini | raw | 0.377 | 0.894 | 0.573 | 0.246 |
| Gemini | neutralized | 0.377 | 0.891 | 0.558 | 0.252 |

---

## 5. Downstream analysis artifacts

Under `results/management_reasoning/analysis/`:

| Artifact family | Covers |
|-----------------|--------|
| `ablation_legacy_dx_metrics.csv` | Legacy FL metrics (through CT + factor; **not yet** `ra_new`) |
| `ablation_indep_dx_metrics.csv` | Claude raw/remove_all/CT + Gemini raw indep FL (**stale vs** newer `ra_new` / missing Gemini arms) |
| `indep_remove_all_*_flips.csv` | Independent flips: paper remove_all |
| `indep_new_neu_*_flips.csv` | Independent flips: ct_old / ct_new |
| `indep_factor_*_flips.csv` | Independent flips: format_tone / content_format |
| `indep_ra_new_*_flips.csv` | Independent flips: ra_new |
| `flip_overlap_*`, `core_flip_*`, `coherent_multi_flips_*` | Cross-arm flip overlap / cores (pre–`ra_new` fold-in) |
| Plots | `plotting/management_reasoning/indep_*_*.{png,pdf}` |

---

## 6. Notable gaps (generated ≠ fully FL-evaluated)

1. **Independent × Gemini** for `remove_all`, `ct_old`, `ct_new`, `format_tone`, `content_format`: Batch collected; **no** Flash-Lite `indep_dx` aggregates (flips only).
2. **Independent × factor pairs (both models)**: Batch + flips; **no** Flash-Lite indep_dx.
3. **Order suites**: Claude raw full generate; **no** Flash-Lite.
4. **`ra_new` not yet folded** into ablation comparison CSV / core flip scoreboard (plan left that optional).
5. Early pilots (n=50 reword v2/v3, cost calib, etc.) are exploratory only — not part of the full diagnosis eval grid above.

---

## 7. Quick map: suite → purpose

| Suite | Purpose |
|-------|---------|
| `primary` | Main MR multi-ask: raw vs content+tone `neutralized` |
| `order_ord*` | Prompt-order sensitivity (Claude raw) |
| `independent` / `independent_remove_all` | Indep MR × raw / paper remove_all |
| `independent_new_neu` | Indep MR × gpt-5.2 content+tone |
| `independent_factor` | Indep MR × paper factor pairs |
| `independent_ra_new` | Indep MR × gpt-5.2 full neutralize |
| `legacy_diag` | Paper free-form dx isolation (Claude raw / remove_all) |
| `legacy_dx` | Free-form dx grid: Gemini + CT arms (reuses Claude legacy_diag) |
| `legacy_dx_factor` | Free-form dx × factor pairs |
| `legacy_dx_ra_new` | Free-form dx × `ra_new` |
