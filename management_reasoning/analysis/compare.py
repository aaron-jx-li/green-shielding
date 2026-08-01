"""Flip / acuity-delta metrics between baseline and ablation answer sets."""

from __future__ import annotations

import csv
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from management_reasoning.prompts import (
    ACUITY_ORDER,
    NEED_MORE_INFO_OPTION,
    RESPONSE_FIELDS,
)

_WS = re.compile(r"\s+")

ACUITY_INDEX = {opt: i for i, opt in enumerate(ACUITY_ORDER)}

# Short labels for heatmaps
ACUITY_SHORT = {
    "Call 911 / Emergency Services": "911",
    "Go to the emergency department now": "ED",
    "Go to urgent care": "UC",
    "Seek same-day in-person care": "Same-day",
    "Schedule non-urgent medical appointment": "Non-urgent",
    "Self-care at home": "Home",
    NEED_MORE_INFO_OPTION: "NMI",
}

ABLATION_LABELS: Tuple[str, ...] = ("neut", "indep", "ord1", "ord2", "ord3")

FLIP_FIELDS: Tuple[str, ...] = (
    "diagnostic_consensus",
    "next_steps_consensus",
    "diagnosis",
    "cant_miss_ruling_out_question",
    "care_seeking",
)

FIELD_DISPLAY = {
    "diagnostic_consensus": "a: diag. consensus",
    "next_steps_consensus": "d: next-steps consensus",
    "diagnosis": "dx: diagnosis",
    "cant_miss_ruling_out_question": "c: can't-miss Q",
    "care_seeking": "b: care seeking",
}


def normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = _WS.sub(" ", str(value).strip().lower())
    return s or None


def field_valid(entry: Dict[str, Any], field_name: str) -> bool:
    fields = entry.get("fields") or {}
    if field_name not in fields or fields[field_name] is None:
        return False
    field_ok = entry.get("field_ok")
    if isinstance(field_ok, dict) and field_name in field_ok:
        return bool(field_ok[field_name])
    return bool(entry.get("parse_ok"))


def answers_equal(field_name: str, a: Any, b: Any) -> bool:
    if field_name in ("diagnosis", "cant_miss_ruling_out_question"):
        return normalize_text(a) == normalize_text(b)
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    return a == b


def acuity_index(value: Any) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip()
    if s == NEED_MORE_INFO_OPTION:
        return None
    return ACUITY_INDEX.get(s)


@dataclass
class PairStats:
    ablation: str
    field: str
    n_paired: int = 0
    n_flip: int = 0
    n_high_to_low: int = 0
    n_low_to_high: int = 0
    # acuity
    n_ordinal: int = 0
    n_changed: int = 0
    n_nmi: int = 0
    sum_delta: float = 0.0
    sum_abs_delta: float = 0.0
    deltas: List[int] = field(default_factory=list)
    transitions: Counter = field(default_factory=Counter)

    @property
    def pct_flip(self) -> float:
        return 100.0 * self.n_flip / self.n_paired if self.n_paired else float("nan")

    @property
    def pct_high_to_low(self) -> float:
        return 100.0 * self.n_high_to_low / self.n_paired if self.n_paired else float("nan")

    @property
    def pct_low_to_high(self) -> float:
        return 100.0 * self.n_low_to_high / self.n_paired if self.n_paired else float("nan")

    @property
    def pct_changed(self) -> float:
        return 100.0 * self.n_changed / self.n_ordinal if self.n_ordinal else float("nan")

    @property
    def pct_nmi(self) -> float:
        return 100.0 * self.n_nmi / self.n_paired if self.n_paired else float("nan")

    @property
    def mean_delta(self) -> float:
        return self.sum_delta / self.n_ordinal if self.n_ordinal else float("nan")

    @property
    def mean_abs_delta(self) -> float:
        return self.sum_abs_delta / self.n_ordinal if self.n_ordinal else float("nan")


def compare_field(
    baseline: Dict[int, Dict[str, Any]],
    ablation: Dict[int, Dict[str, Any]],
    *,
    ablation_name: str,
    field_name: str,
) -> PairStats:
    stats = PairStats(ablation=ablation_name, field=field_name)
    shared = set(baseline) & set(ablation)
    for sid in shared:
        b_ent = baseline[sid]
        a_ent = ablation[sid]
        if not field_valid(b_ent, field_name) or not field_valid(a_ent, field_name):
            continue
        b_val = b_ent["fields"][field_name]
        a_val = a_ent["fields"][field_name]
        stats.n_paired += 1

        equal = answers_equal(field_name, b_val, a_val)
        if not equal:
            stats.n_flip += 1

        if field_name in ("diagnostic_consensus", "next_steps_consensus"):
            bb = str(b_val).strip().lower()
            aa = str(a_val).strip().lower()
            if bb == "high" and aa == "low":
                stats.n_high_to_low += 1
            elif bb == "low" and aa == "high":
                stats.n_low_to_high += 1

        if field_name == "care_seeking":
            stats.transitions[(str(b_val), str(a_val))] += 1
            b_nmi = str(b_val) == NEED_MORE_INFO_OPTION
            a_nmi = str(a_val) == NEED_MORE_INFO_OPTION
            if b_nmi or a_nmi:
                stats.n_nmi += 1
            bi = acuity_index(b_val)
            ai = acuity_index(a_val)
            if bi is None or ai is None:
                continue
            stats.n_ordinal += 1
            delta = ai - bi
            stats.deltas.append(delta)
            stats.sum_delta += delta
            stats.sum_abs_delta += abs(delta)
            if delta != 0:
                stats.n_changed += 1

    return stats


def flip_sample_ids(
    baseline: Dict[int, Dict[str, Any]],
    ablation: Dict[int, Dict[str, Any]],
    *,
    field_name: str,
    care_seeking_abs_delta: bool = True,
) -> set[int]:
    """Sample IDs where ``field_name`` differs from baseline (paired + valid).

    For ``care_seeking``, by default only counts ordinal pairs with |Δ|≥1
    (matches the flip-rate plots' ``pct_changed`` definition).
    """
    out: set[int] = set()
    shared = set(baseline) & set(ablation)
    for sid in shared:
        b_ent = baseline[sid]
        a_ent = ablation[sid]
        if not field_valid(b_ent, field_name) or not field_valid(a_ent, field_name):
            continue
        b_val = b_ent["fields"][field_name]
        a_val = a_ent["fields"][field_name]
        if field_name == "care_seeking" and care_seeking_abs_delta:
            bi = acuity_index(b_val)
            ai = acuity_index(a_val)
            if bi is None or ai is None:
                continue
            if abs(ai - bi) >= 1:
                out.add(sid)
            continue
        if not answers_equal(field_name, b_val, a_val):
            out.add(sid)
    return out


def compare_all(
    grid: Dict[str, Dict[int, Dict[str, Any]]],
    *,
    baseline_key: str = "raw",
    ablation_keys: Sequence[str] = ABLATION_LABELS,
) -> Dict[str, Dict[str, PairStats]]:
    baseline = grid[baseline_key]
    out: Dict[str, Dict[str, PairStats]] = {}
    for ab in ablation_keys:
        out[ab] = {
            fname: compare_field(baseline, grid[ab], ablation_name=ab, field_name=fname)
            for fname in FLIP_FIELDS
        }
    return out


def mean_order_flip_rates(
    all_stats: Dict[str, Dict[str, PairStats]],
    fields: Sequence[str] = ("diagnostic_consensus", "next_steps_consensus", "diagnosis", "cant_miss_ruling_out_question", "care_seeking"),
) -> Dict[str, float]:
    """Mean pct_flip across ord1–ord3 for each field."""
    means: Dict[str, float] = {}
    for fname in fields:
        vals = [all_stats[o][fname].pct_flip for o in ("ord1", "ord2", "ord3") if o in all_stats]
        means[fname] = sum(vals) / len(vals) if vals else float("nan")
    return means


def write_summary_csv(path: str, all_stats: Dict[str, Dict[str, PairStats]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for ab, by_field in all_stats.items():
        for fname, st in by_field.items():
            row = {
                "ablation": ab,
                "field": fname,
                "n_paired": st.n_paired,
                "n_flip": st.n_flip,
                "pct_flip": round(st.pct_flip, 4) if st.n_paired else "",
                "n_high_to_low": st.n_high_to_low,
                "pct_high_to_low": round(st.pct_high_to_low, 4) if st.n_paired else "",
                "n_low_to_high": st.n_low_to_high,
                "pct_low_to_high": round(st.pct_low_to_high, 4) if st.n_paired else "",
                "n_ordinal": st.n_ordinal,
                "n_changed": st.n_changed,
                "pct_changed": round(st.pct_changed, 4) if st.n_ordinal else "",
                "mean_delta": round(st.mean_delta, 4) if st.n_ordinal else "",
                "mean_abs_delta": round(st.mean_abs_delta, 4) if st.n_ordinal else "",
                "n_nmi": st.n_nmi,
                "pct_nmi": round(st.pct_nmi, 4) if st.n_paired else "",
            }
            rows.append(row)
    # mean across order variants
    means = mean_order_flip_rates(all_stats)
    for fname, pct in means.items():
        rows.append(
            {
                "ablation": "ord_mean",
                "field": fname,
                "n_paired": "",
                "n_flip": "",
                "pct_flip": round(pct, 4),
                "n_high_to_low": "",
                "pct_high_to_low": "",
                "n_low_to_high": "",
                "pct_low_to_high": "",
                "n_ordinal": "",
                "n_changed": "",
                "pct_changed": "",
                "mean_delta": "",
                "mean_abs_delta": "",
                "n_nmi": "",
                "pct_nmi": "",
            }
        )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def transition_matrix(
    stats: PairStats,
    *,
    include_nmi: bool = False,
) -> Tuple[List[str], List[List[float]]]:
    """Row-normalized %: rows=raw, cols=ablation. NMI dropped by default."""
    labels = list(ACUITY_ORDER)
    if include_nmi:
        labels = labels + [NEED_MORE_INFO_OPTION]
    n = len(labels)
    counts = [[0.0] * n for _ in range(n)]
    idx = {lab: i for i, lab in enumerate(labels)}
    for (raw, abl), c in stats.transitions.items():
        if raw not in idx or abl not in idx:
            continue
        counts[idx[raw]][idx[abl]] += c
    mat: List[List[float]] = []
    for row in counts:
        s = sum(row)
        if s <= 0:
            mat.append([0.0] * n)
        else:
            mat.append([100.0 * v / s for v in row])
    return labels, mat
