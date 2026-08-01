#!/usr/bin/env python3
"""Core flip sets + arm scoreboard for clinician-annotation arm choice.

Builds stable cores of samples that move under related paper neutralizations,
then scores each ablation by core coverage and directional purity.

Cores (per model, per track):
  - ``ra_ft``: remove_all ∩ format_tone
  - ``paper_ge2``: flipped by ≥2 of {format_tone, content_format, remove_all}
  - ``paper_ge3``: flipped by all 3 paper arms
  - ``all_ge3``: flipped by ≥3 of all 5 ablations

Tracks:
  - diagnostic_consensus flips (with high→low / low→high)
  - care_seeking |Δ|>1 (with up/down and Same-day→ED flag)

Usage:
  python -m management_reasoning.analysis.core_flip_scoreboard
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

from management_reasoning.analysis.compare import (
    ACUITY_SHORT,
    answers_equal,
    acuity_index,
    field_valid,
    flip_sample_ids,
)
from management_reasoning.analysis.plot_changes import configure_matplotlib, save_fig
from management_reasoning.analysis.plot_flip_overlap import (
    ABLATION_DISPLAY,
    ABLATIONS,
    load_full_independent_grid,
)

PAPER_ARMS = ("format_tone", "content_format", "remove_all")
CORE_DEFS = ("ra_ft", "paper_ge2", "paper_ge3", "all_ge3")


def _consensus_direction(
    baseline: Dict[int, Dict[str, Any]],
    ablation: Dict[int, Dict[str, Any]],
    *,
    field: str,
) -> Dict[int, str]:
    """sample_id → high_to_low | low_to_high | other (only flipped samples)."""
    out: Dict[int, str] = {}
    for sid in set(baseline) & set(ablation):
        b_ent, a_ent = baseline[sid], ablation[sid]
        if not field_valid(b_ent, field) or not field_valid(a_ent, field):
            continue
        bv, av = b_ent["fields"][field], a_ent["fields"][field]
        if answers_equal(field, bv, av):
            continue
        bb, aa = str(bv).strip().lower(), str(av).strip().lower()
        if bb == "high" and aa == "low":
            out[sid] = "high_to_low"
        elif bb == "low" and aa == "high":
            out[sid] = "low_to_high"
        else:
            out[sid] = "other"
    return out


def _acuity_gt1_meta(
    baseline: Dict[int, Dict[str, Any]],
    ablation: Dict[int, Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    """sample_id → {delta, direction, from_short, to_short, same_day_to_ed}."""
    out: Dict[int, Dict[str, Any]] = {}
    for sid in set(baseline) & set(ablation):
        b_ent, a_ent = baseline[sid], ablation[sid]
        if not field_valid(b_ent, "care_seeking") or not field_valid(
            a_ent, "care_seeking"
        ):
            continue
        bv, av = b_ent["fields"]["care_seeking"], a_ent["fields"]["care_seeking"]
        bi, ai = acuity_index(bv), acuity_index(av)
        if bi is None or ai is None:
            continue
        delta = ai - bi
        if abs(delta) <= 1:
            continue
        from_s = ACUITY_SHORT.get(str(bv).strip(), str(bv))
        to_s = ACUITY_SHORT.get(str(av).strip(), str(av))
        out[sid] = {
            "delta": delta,
            "abs_delta": abs(delta),
            "direction": "down" if delta < 0 else "up",
            "from_short": from_s,
            "to_short": to_s,
            "same_day_to_ed": from_s == "Same-day" and to_s == "ED",
            "ed_to_same_day": from_s == "ED" and to_s == "Same-day",
        }
    return out


def _multi_hit(sets: Dict[str, Set[int]], arms: Sequence[str]) -> Counter:
    hits: Counter = Counter()
    for ab in arms:
        for sid in sets[ab]:
            hits[sid] += 1
    return hits


def build_cores(flip_sets: Dict[str, Set[int]]) -> Dict[str, Set[int]]:
    paper_hits = _multi_hit(flip_sets, PAPER_ARMS)
    all_hits = _multi_hit(flip_sets, ABLATIONS)
    return {
        "ra_ft": flip_sets["remove_all"] & flip_sets["format_tone"],
        "paper_ge2": {sid for sid, n in paper_hits.items() if n >= 2},
        "paper_ge3": {sid for sid, n in paper_hits.items() if n >= 3},
        "all_ge3": {sid for sid, n in all_hits.items() if n >= 3},
    }


def score_arm(
    arm_set: Set[int],
    core: Set[int],
    *,
    directions: Optional[Dict[int, str]] = None,
    acuity_meta: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    inter = arm_set & core
    n_arm = len(arm_set)
    n_core = len(core)
    n_inter = len(inter)
    row: Dict[str, Any] = {
        "n_arm": n_arm,
        "n_core": n_core,
        "n_in_core": n_inter,
        "n_unique_outside_core": len(arm_set - core),
        "pct_arm_in_core": round(100.0 * n_inter / n_arm, 2) if n_arm else "",
        "pct_core_covered": round(100.0 * n_inter / n_core, 2) if n_core else "",
        "jaccard_vs_core": round(
            len(inter) / len(arm_set | core), 4
        )
        if (arm_set or core)
        else "",
    }
    if directions is not None:
        # Direction among arm flips that sit in core (prefer); else all arm flips
        use = inter if inter else arm_set
        h2l = sum(1 for sid in use if directions.get(sid) == "high_to_low")
        l2h = sum(1 for sid in use if directions.get(sid) == "low_to_high")
        directed = h2l + l2h
        row["n_high_to_low"] = h2l
        row["n_low_to_high"] = l2h
        row["pct_high_to_low_among_directed"] = (
            round(100.0 * h2l / directed, 2) if directed else ""
        )
        # Purity: |h2l - l2h| / directed  (1 = all one way)
        row["direction_purity"] = (
            round(abs(h2l - l2h) / directed, 4) if directed else ""
        )
        row["net_more_worried"] = h2l - l2h  # + = more high→low
    if acuity_meta is not None:
        use = inter if inter else arm_set
        down = sum(1 for sid in use if acuity_meta.get(sid, {}).get("direction") == "down")
        up = sum(1 for sid in use if acuity_meta.get(sid, {}).get("direction") == "up")
        s2e = sum(1 for sid in use if acuity_meta.get(sid, {}).get("same_day_to_ed"))
        e2s = sum(1 for sid in use if acuity_meta.get(sid, {}).get("ed_to_same_day"))
        directed = down + up
        row["n_acuity_down"] = down
        row["n_acuity_up"] = up
        row["n_same_day_to_ed"] = s2e
        row["n_ed_to_same_day"] = e2s
        row["pct_down_among_directed"] = (
            round(100.0 * down / directed, 2) if directed else ""
        )
        row["acuity_purity"] = (
            round(abs(down - up) / directed, 4) if directed else ""
        )
        row["net_more_acute"] = down - up
    return row


def write_csv(path: str, rows: List[dict], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote  {path}")


def write_id_list(path: str, ids: Set[int], meta_rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    by_id = {int(r["sample_id"]): r for r in meta_rows}
    rows = []
    for sid in sorted(ids):
        m = by_id.get(sid, {"sample_id": sid})
        rows.append(m)
    if not rows:
        rows = [{"sample_id": ""}]
        fieldnames = ["sample_id"]
    else:
        # stable column order
        keys = ["sample_id"]
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote  {path}  n={len(ids)}")


def plot_scoreboard(
    score_rows: List[dict],
    *,
    model: str,
    track: str,
    core_name: str,
    stem: str,
) -> None:
    rows = [
        r
        for r in score_rows
        if r["model"] == model and r["track"] == track and r["core"] == core_name
    ]
    if not rows:
        return
    arms = [r["arm"] for r in rows]
    labels = [ABLATION_DISPLAY[a] for a in arms]
    cov = [float(r["pct_core_covered"] or 0) for r in rows]
    in_core = [float(r["pct_arm_in_core"] or 0) for r in rows]
    x = np.arange(len(arms))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.bar(x - width / 2, cov, width, label="% of core covered by arm", color="#1b9e77")
    ax.bar(
        x + width / 2,
        in_core,
        width,
        label="% of arm flips inside core",
        color="#7570b3",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.set_title(f"{model.capitalize()} — {track} vs core={core_name}")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    fig.tight_layout()
    save_fig(fig, stem)


def run_model(
    model: str,
    *,
    out_dir: str,
    fig_dir: str,
) -> Dict[str, Any]:
    print(f"\n=== {model} ===")
    grid = load_full_independent_grid(model)
    raw = grid["raw"]

    # --- diagnostic consensus track ---
    diag_sets = {
        ab: flip_sample_ids(raw, grid[ab], field_name="diagnostic_consensus")
        for ab in ABLATIONS
    }
    diag_dir = {
        ab: _consensus_direction(raw, grid[ab], field="diagnostic_consensus")
        for ab in ABLATIONS
    }
    diag_cores = build_cores(diag_sets)

    # --- acuity |Δ|>1 track ---
    acuity_meta = {ab: _acuity_gt1_meta(raw, grid[ab]) for ab in ABLATIONS}
    acuity_sets = {ab: set(acuity_meta[ab]) for ab in ABLATIONS}
    acuity_cores = build_cores(acuity_sets)

    score_rows: List[dict] = []
    core_size_rows: List[dict] = []
    id_meta_diag: List[dict] = []
    id_meta_acuity: List[dict] = []

    for core_name, core in diag_cores.items():
        core_size_rows.append(
            {
                "model": model,
                "track": "diagnostic_consensus",
                "core": core_name,
                "n": len(core),
            }
        )
        for ab in ABLATIONS:
            sc = score_arm(diag_sets[ab], core, directions=diag_dir[ab])
            score_rows.append(
                {
                    "model": model,
                    "track": "diagnostic_consensus",
                    "core": core_name,
                    "arm": ab,
                    **sc,
                }
            )

    for core_name, core in acuity_cores.items():
        core_size_rows.append(
            {
                "model": model,
                "track": "acuity_gt1",
                "core": core_name,
                "n": len(core),
            }
        )
        for ab in ABLATIONS:
            sc = score_arm(acuity_sets[ab], core, acuity_meta=acuity_meta[ab])
            score_rows.append(
                {
                    "model": model,
                    "track": "acuity_gt1",
                    "core": core_name,
                    "arm": ab,
                    **sc,
                }
            )

    # Per-sample metadata for ID lists (union over arms for richness)
    all_diag = set().union(*diag_sets.values())
    for sid in sorted(all_diag):
        arms_hit = [ab for ab in ABLATIONS if sid in diag_sets[ab]]
        # prefer remove_all direction if present
        direction = ""
        for ab in ("remove_all", "format_tone", "content_format", "ct_old", "ct_new"):
            if sid in diag_dir[ab]:
                direction = diag_dir[ab][sid]
                break
        id_meta_diag.append(
            {
                "sample_id": sid,
                "n_arms": len(arms_hit),
                "arms": "|".join(arms_hit),
                "in_ra_ft": int(sid in diag_cores["ra_ft"]),
                "in_paper_ge2": int(sid in diag_cores["paper_ge2"]),
                "in_paper_ge3": int(sid in diag_cores["paper_ge3"]),
                "in_all_ge3": int(sid in diag_cores["all_ge3"]),
                "direction": direction,
                "interesting": int(direction == "high_to_low"),
            }
        )

    all_acuity = set().union(*acuity_sets.values())
    for sid in sorted(all_acuity):
        arms_hit = [ab for ab in ABLATIONS if sid in acuity_sets[ab]]
        meta = {}
        for ab in ("remove_all", "format_tone", "content_format", "ct_old", "ct_new"):
            if sid in acuity_meta[ab]:
                meta = acuity_meta[ab][sid]
                break
        id_meta_acuity.append(
            {
                "sample_id": sid,
                "n_arms": len(arms_hit),
                "arms": "|".join(arms_hit),
                "in_ra_ft": int(sid in acuity_cores["ra_ft"]),
                "in_paper_ge2": int(sid in acuity_cores["paper_ge2"]),
                "in_paper_ge3": int(sid in acuity_cores["paper_ge3"]),
                "in_all_ge3": int(sid in acuity_cores["all_ge3"]),
                "direction": meta.get("direction", ""),
                "delta": meta.get("delta", ""),
                "from_short": meta.get("from_short", ""),
                "to_short": meta.get("to_short", ""),
                "same_day_to_ed": int(bool(meta.get("same_day_to_ed"))),
                "interesting": int(
                    bool(meta.get("same_day_to_ed"))
                    or meta.get("direction") == "down"
                ),
            }
        )

    # Shortlists for annotation
    # Primary: ra_ft core ∩ interesting direction (diag high→low OR acuity Same-day→ED)
    shortlists = {
        "diag_ra_ft": diag_cores["ra_ft"],
        "diag_ra_ft_high_to_low": {
            sid
            for sid in diag_cores["ra_ft"]
            if any(diag_dir[ab].get(sid) == "high_to_low" for ab in ABLATIONS)
        },
        "diag_paper_ge3": diag_cores["paper_ge3"],
        "diag_paper_ge3_high_to_low": {
            sid
            for sid in diag_cores["paper_ge3"]
            if any(diag_dir[ab].get(sid) == "high_to_low" for ab in ABLATIONS)
        },
        "acuity_ra_ft": acuity_cores["ra_ft"],
        "acuity_ra_ft_same_day_to_ed": {
            sid
            for sid in acuity_cores["ra_ft"]
            if any(
                acuity_meta[ab].get(sid, {}).get("same_day_to_ed") for ab in ABLATIONS
            )
        },
        "acuity_paper_ge3": acuity_cores["paper_ge3"],
        # Combined annotation pool: union of interesting cores
        "annotation_pool": set(),
    }
    shortlists["annotation_pool"] = (
        shortlists["diag_ra_ft_high_to_low"]
        | shortlists["acuity_ra_ft_same_day_to_ed"]
    )

    # CT survival of ra_ft core
    ct_survival_rows = []
    for track, cores, sets in (
        ("diagnostic_consensus", diag_cores, diag_sets),
        ("acuity_gt1", acuity_cores, acuity_sets),
    ):
        core = cores["ra_ft"]
        for ab in ("ct_old", "ct_new"):
            inter = core & sets[ab]
            ct_survival_rows.append(
                {
                    "model": model,
                    "track": track,
                    "core": "ra_ft",
                    "ct_arm": ab,
                    "n_core": len(core),
                    "n_also_flip_under_ct": len(inter),
                    "pct_core_survives_ct": round(100.0 * len(inter) / len(core), 2)
                    if core
                    else "",
                }
            )
        both = core & sets["ct_old"] & sets["ct_new"]
        ct_survival_rows.append(
            {
                "model": model,
                "track": track,
                "core": "ra_ft",
                "ct_arm": "ct_old_and_ct_new",
                "n_core": len(core),
                "n_also_flip_under_ct": len(both),
                "pct_core_survives_ct": round(100.0 * len(both) / len(core), 2)
                if core
                else "",
            }
        )

    # Write outputs
    sub = os.path.join(out_dir, f"core_flips_{model}")
    os.makedirs(sub, exist_ok=True)

    write_csv(
        os.path.join(out_dir, f"core_flip_scoreboard_{model}.csv"),
        score_rows,
        [
            "model",
            "track",
            "core",
            "arm",
            "n_arm",
            "n_core",
            "n_in_core",
            "n_unique_outside_core",
            "pct_arm_in_core",
            "pct_core_covered",
            "jaccard_vs_core",
            "n_high_to_low",
            "n_low_to_high",
            "pct_high_to_low_among_directed",
            "direction_purity",
            "net_more_worried",
            "n_acuity_down",
            "n_acuity_up",
            "n_same_day_to_ed",
            "n_ed_to_same_day",
            "pct_down_among_directed",
            "acuity_purity",
            "net_more_acute",
        ],
    )
    write_csv(
        os.path.join(out_dir, f"core_flip_sizes_{model}.csv"),
        core_size_rows,
        ["model", "track", "core", "n"],
    )
    write_csv(
        os.path.join(out_dir, f"core_ct_survival_{model}.csv"),
        ct_survival_rows,
        [
            "model",
            "track",
            "core",
            "ct_arm",
            "n_core",
            "n_also_flip_under_ct",
            "pct_core_survives_ct",
        ],
    )

    write_id_list(
        os.path.join(sub, "diag_all_flips_meta.csv"), all_diag, id_meta_diag
    )
    write_id_list(
        os.path.join(sub, "acuity_gt1_all_meta.csv"), all_acuity, id_meta_acuity
    )
    for name, ids in shortlists.items():
        meta = id_meta_diag if name.startswith("diag") or name == "annotation_pool" else id_meta_acuity
        if name.startswith("acuity"):
            meta = id_meta_acuity
        elif name == "annotation_pool":
            # merge meta
            by = {r["sample_id"]: dict(r) for r in id_meta_diag}
            for r in id_meta_acuity:
                sid = r["sample_id"]
                if sid in by:
                    by[sid]["acuity_direction"] = r.get("direction", "")
                    by[sid]["same_day_to_ed"] = r.get("same_day_to_ed", 0)
                    by[sid]["interesting_acuity"] = r.get("interesting", 0)
                else:
                    row = dict(r)
                    row["direction"] = ""
                    row["interesting_diag"] = 0
                    by[sid] = row
            meta = list(by.values())
        write_id_list(os.path.join(sub, f"{name}_ids.csv"), ids, meta)

    # Plots for primary core ra_ft
    plot_scoreboard(
        score_rows,
        model=model,
        track="diagnostic_consensus",
        core_name="ra_ft",
        stem=os.path.join(fig_dir, f"core_scoreboard_{model}_diag_ra_ft"),
    )
    plot_scoreboard(
        score_rows,
        model=model,
        track="acuity_gt1",
        core_name="ra_ft",
        stem=os.path.join(fig_dir, f"core_scoreboard_{model}_acuity_ra_ft"),
    )

    # Print summary
    print(f"\nCore sizes ({model}):")
    for r in core_size_rows:
        print(f"  {r['track']:22} {r['core']:12} n={r['n']}")
    print(f"\nScoreboard vs ra_ft ({model}, diagnostic):")
    for r in score_rows:
        if r["track"] != "diagnostic_consensus" or r["core"] != "ra_ft":
            continue
        print(
            f"  {r['arm']:16} cover={r['pct_core_covered']}%  "
            f"in_core={r['pct_arm_in_core']}%  "
            f"h2l/l2h={r.get('n_high_to_low')}/{r.get('n_low_to_high')}  "
            f"net_worried={r.get('net_more_worried')}"
        )
    print(f"\nCT survival of ra_ft ({model}):")
    for r in ct_survival_rows:
        if r["core"] != "ra_ft":
            continue
        print(
            f"  {r['track']:22} {r['ct_arm']:20} "
            f"{r['n_also_flip_under_ct']}/{r['n_core']} ({r['pct_core_survives_ct']}%)"
        )
    print(f"\nShortlists ({model}):")
    for name, ids in shortlists.items():
        print(f"  {name:32} n={len(ids)}")

    summary = {
        "model": model,
        "diag_cores": {k: len(v) for k, v in diag_cores.items()},
        "acuity_cores": {k: len(v) for k, v in acuity_cores.items()},
        "shortlists": {k: len(v) for k, v in shortlists.items()},
        "ct_survival": ct_survival_rows,
    }
    with open(os.path.join(sub, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    configure_matplotlib()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("claude", "gemini", "both"), default="both")
    p.add_argument("--out-dir", default="./results/management_reasoning/analysis")
    p.add_argument("--fig-dir", default="./plotting/management_reasoning")
    args = p.parse_args(argv)
    models = ("claude", "gemini") if args.model == "both" else (args.model,)
    all_sum = {}
    for m in models:
        all_sum[m] = run_model(m, out_dir=args.out_dir, fig_dir=args.fig_dir)
    with open(
        os.path.join(args.out_dir, "core_flip_summary.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(all_sum, f, indent=2)
        f.write("\n")
    print(f"\nWrote  {os.path.join(args.out_dir, 'core_flip_summary.json')}")
    print("Done core flip scoreboard.")


if __name__ == "__main__":
    main()
