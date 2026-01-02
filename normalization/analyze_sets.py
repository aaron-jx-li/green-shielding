import json
from collections import defaultdict

def normalize_dx(dx: str) -> str:
    return (dx or "").strip().lower()

def compute_position_stats(samples):
    """
    Position-wise stats:
      pos=0 aggregates the 1st extracted dx of each question that has >=1 dx
      pos=1 aggregates the 2nd extracted dx of each question that has >=2 dx
      ...

    Uses per-sample:
      metrics.extracted_diagnoses (list)
      metrics.in_P / metrics.in_H (list of dicts with key "dx")
    """

    # pos -> counts
    counts = defaultdict(lambda: {"n_total": 0, "n_in_P": 0, "n_in_H": 0})

    for s in samples:
        m = s.get("metrics", {})
        extracted = m.get("extracted_diagnoses", []) or []

        in_P = {normalize_dx(d.get("dx")) for d in m.get("in_P", [])}
        in_H = {normalize_dx(d.get("dx")) for d in m.get("in_H", [])}

        for pos, dx in enumerate(extracted):
            dx_n = normalize_dx(dx)
            if not dx_n:
                continue
            counts[pos]["n_total"] += 1
            if dx_n in in_P:
                counts[pos]["n_in_P"] += 1
            if dx_n in in_H:
                counts[pos]["n_in_H"] += 1

    # pos -> means
    results = {}
    for pos, c in sorted(counts.items()):
        n = c["n_total"]
        results[pos] = {
            "n_questions_with_dx_at_pos": n,
            "mean_plausibility": (c["n_in_P"] / n) if n else 0.0,
            "mean_h_precision": (c["n_in_H"] / n) if n else 0.0,
        }
    return results


# ---- Usage ----
with open("./results/HCM-9k/eval_converted_gpt-4.1-mini.json", "r") as f:
    data = json.load(f)

samples = data["per_sample"] if "per_sample" in data else data

pos_stats = compute_position_stats(samples)

for pos, stats in pos_stats.items():
    print(
        f"pos={pos+1:2d}  "
        f"n={stats['n_questions_with_dx_at_pos']:4d}  "
        f"P={stats['mean_plausibility']:.3f}  "
        f"H={stats['mean_h_precision']:.3f}"
    )