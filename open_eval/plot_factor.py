 #!/usr/bin/env python3
"""
Factor-level analysis plots (raw vs converted) for:
- plausibility
- h_precision
- h_coverage
- breadth
- uncertainty_rate (mean of uncertainty_flag)

Generates one bar-chart per metric, grouped by prompt-level factor ("Yes" only),
with mean ± standard error, similar to your screenshot.

Assumptions:
- factors_json is a LIST of samples (same dataset), each with:
    {"factors": {...}}  and optionally "raw_input"/"normalized_prompt"/etc.
- eval_raw_json and eval_conv_json each have:
    {"summary": ..., "per_sample": [{"index": i, "metrics": {...}}, ...]}
- Indices align across the two eval files (same ordering and index ids).
- If factor file has no explicit "index", we align by list position (0..n-1).
"""

import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Config: set your paths
# -------------------------
FACTORS_PATH = "./results/HCM-3k/responses_gpt-4.1-mini.json"          # your file containing "factors" per sample
EVAL_RAW_PATH = "./results/HCM-3k/eval_raw_gpt-4.1-mini.json"        # raw-input evaluation results
EVAL_CONV_PATH = "./results/HCM-3k/eval_converted_gpt-4.1-mini.json" # converted-input evaluation results
OUTDIR = "./figs/"


# -------------------------
# Utilities
# -------------------------
def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def standard_error(x: List[float]) -> float:
    if len(x) <= 1:
        return 0.0
    arr = np.asarray(x, dtype=float)
    return float(arr.std(ddof=1) / math.sqrt(len(arr)))

def mean_and_se(x: List[float]) -> Tuple[float, float, int]:
    if not x:
        return (float("nan"), float("nan"), 0)
    return (float(np.mean(x)), standard_error(x), len(x))

def safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


# -------------------------
# Loading + alignment
# -------------------------
def build_factor_by_index(factors_data: Any) -> Dict[int, Dict[str, bool]]:
    """
    Returns: idx -> factor_dict
    Supports:
      - factors_data is list of sample dicts
      - sample may have "index" field; else align by position
    """
    if not isinstance(factors_data, list):
        raise ValueError("FACTORS_PATH should point to a JSON list of samples.")

    out: Dict[int, Dict[str, bool]] = {}
    for i, item in enumerate(factors_data):
        idx = item.get("index", i)
        f = item.get("factors", {})
        if not isinstance(f, dict):
            f = {}
        out[int(idx)] = {k: bool(v) for k, v in f.items()}
    return out

def build_metrics_by_index(eval_data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Returns: idx -> metrics dict (from per_sample[*].metrics)
    """
    per = eval_data.get("per_sample", [])
    out: Dict[int, Dict[str, Any]] = {}
    for row in per:
        idx = int(row["index"])
        out[idx] = row.get("metrics", {})
    return out

def collect_all_factor_names(factor_by_idx: Dict[int, Dict[str, bool]]) -> List[str]:
    names = set()
    for f in factor_by_idx.values():
        names.update(f.keys())
    # stable-ish ordering: match your slide order if you want by providing a manual list;
    # otherwise use sorted.
    return sorted(names)


# -------------------------
# Metric extraction (per sample)
# -------------------------
def metric_value(metrics: Dict[str, Any], metric: str) -> Optional[float]:
    """
    Extract one scalar metric from a per-sample 'metrics' dict.
    Supported:
      - "plausibility"
      - "h_precision"
      - "h_coverage"
      - "breadth"
      - "uncertainty_rate" (computed from boolean uncertainty_flag)
    """
    if metric in ("plausibility", "h_precision", "h_coverage", "breadth"):
        v = metrics.get(metric, None)
        return float(v) if v is not None else None

    if metric == "uncertainty_rate":
        # eval stores boolean uncertainty_flag per sample in metrics
        flag = metrics.get("uncertainty_flag", None)
        if flag is None:
            return None
        return 1.0 if bool(flag) else 0.0

    raise ValueError(f"Unknown metric: {metric}")


# -------------------------
# Aggregation: factor -> mean metric for raw/converted
# -------------------------
def factor_level_stats(
    factor_names: List[str],
    factor_by_idx: Dict[int, Dict[str, bool]],
    raw_metrics_by_idx: Dict[int, Dict[str, Any]],
    conv_metrics_by_idx: Dict[int, Dict[str, Any]],
    metric: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Returns:
      stats[factor] = {
        "raw": {"mean":..., "se":..., "n":...},
        "conv": {"mean":..., "se":..., "n":...},
      }
    Only uses samples where that factor is True ("Yes" only).
    """
    stats: Dict[str, Dict[str, Any]] = {}

    # indices present in both eval files (and ideally in factor file)
    common_idxs = set(raw_metrics_by_idx.keys()) & set(conv_metrics_by_idx.keys()) & set(factor_by_idx.keys())

    for fname in factor_names:
        raw_vals: List[float] = []
        conv_vals: List[float] = []

        for idx in common_idxs:
            fdict = factor_by_idx[idx]
            if not fdict.get(fname, False):
                continue

            rv = metric_value(raw_metrics_by_idx[idx], metric)
            cv = metric_value(conv_metrics_by_idx[idx], metric)
            if rv is None or cv is None:
                continue

            raw_vals.append(rv)
            conv_vals.append(cv)

        r_mean, r_se, r_n = mean_and_se(raw_vals)
        c_mean, c_se, c_n = mean_and_se(conv_vals)

        stats[fname] = {
            "raw": {"mean": r_mean, "se": r_se, "n": r_n},
            "conv": {"mean": c_mean, "se": c_se, "n": c_n},
        }

    return stats


# -------------------------
# Plotting
# -------------------------
def plot_grouped_bars(
    stats: Dict[str, Dict[str, Any]],
    title: str,
    ylabel: str,
    outfile: str,
    rotate_xticks: int = 45,
):
    """
    Grouped bar chart with error bars: raw vs converted for each factor.
    """
    factors = list(stats.keys())

    raw_means = [stats[f]["raw"]["mean"] for f in factors]
    raw_ses   = [stats[f]["raw"]["se"]   for f in factors]
    conv_means= [stats[f]["conv"]["mean"] for f in factors]
    conv_ses  = [stats[f]["conv"]["se"]   for f in factors]

    x = np.arange(len(factors))
    width = 0.38

    plt.figure(figsize=(14, 4.8))
    plt.bar(x - width/2, raw_means, width, yerr=raw_ses, capsize=3, label="Original prompt")
    plt.bar(x + width/2, conv_means, width, yerr=conv_ses, capsize=3, label="Converted prompt")

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Prompt-level factor ('Yes' only)")
    plt.xticks(x, [f"{f}" for f in factors], rotation=rotate_xticks, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def main():
    ensure_dir(OUTDIR)

    factors_data = load_json(FACTORS_PATH)
    eval_raw = load_json(EVAL_RAW_PATH)
    eval_conv = load_json(EVAL_CONV_PATH)

    factor_by_idx = build_factor_by_index(factors_data)
    raw_metrics_by_idx = build_metrics_by_index(eval_raw)
    conv_metrics_by_idx = build_metrics_by_index(eval_conv)

    factor_names = collect_all_factor_names(factor_by_idx)

    # If you want the exact slide ordering, uncomment and edit this list:
    factor_names = [
      "has_worried_tone",
      "mentions_urgency_or_severity",
      "mentions_specific",
      "prior_belief_high_confidence",
      "asks_for_list",
      "asks_for_management_or_treatment",
      "contains_irrelevant_details",
      "missing_objective_data",
      "missing_time_course",
    #   "multi_turn_like_context",
      "ambiguous_or_unstructured_question_format",
    ]

    metrics_to_plot = [
        ("h_precision", "H precision by prompt-level factor", "Mean H precision", "h_precision.png"),
        ("h_coverage", "H coverage by prompt-level factor", "Mean H coverage", "h_coverage.png"),
        ("plausibility", "Plausibility by prompt-level factor", "Mean plausibility", "plausibility.png"),
        ("breadth", "Breadth by prompt-level factor", "Mean breadth (# diagnoses)", "breadth.png"),
        ("uncertainty_rate", "Uncertainty rate by prompt-level factor", "Mean uncertainty rate", "uncertainty_rate.png"),
    ]

    for metric, title, ylabel, fname in metrics_to_plot:
        stats = factor_level_stats(
            factor_names=factor_names,
            factor_by_idx=factor_by_idx,
            raw_metrics_by_idx=raw_metrics_by_idx,
            conv_metrics_by_idx=conv_metrics_by_idx,
            metric=metric,
        )
        outpath = str(Path(OUTDIR) / fname)
        plot_grouped_bars(stats, title=title, ylabel=ylabel, outfile=outpath)

    print(f"Saved plots to: {OUTDIR}")
    print("Files:")
    for p in sorted(Path(OUTDIR).glob("*.png")):
        print(" -", p)


if __name__ == "__main__":
    main()
