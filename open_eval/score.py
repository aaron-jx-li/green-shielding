
"""
Fit (lambda, mu_H, mu_S, nu) from doctor responses, then compute DAS for:
  - doctor responses
  - raw model responses
  - normalized model responses

DAS components (all are PENALTIES; lower DAS is better):
  1) breadth_norm: breadth / (doctor 95th percentile)
  2) miss_H_rate: |uncovered_H| / |H|
  3) miss_S_rate: |uncovered_C| / |C|   (C == safety-critical set in your JSON)
  4) out_of_P_norm: |out_of_P| / (doctor 95th percentile)

Key fix vs previous version:
- Synthetic negatives MUST create tradeoffs; otherwise any nonnegative weight vector
  ranks positives < negatives and hinge gradients go ~zero, collapsing to equal weights.
- We therefore include "omit core dx" negatives that reduce breadth (good) but increase miss (bad).
"""

import argparse, json, random
from typing import List, Dict, Any, Tuple

# ---------------- I/O ----------------

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# ---------------- Utilities ----------------

def safe_len(x) -> int:
    return len(x) if x is not None else 0

def percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 1.0
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 1:
        return float(sorted_vals[-1])
    n = len(sorted_vals)
    pos = (n - 1) * q
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)

def split_indices(n: int, test_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_test = int(round(n * test_ratio))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return train_idx, test_idx

def clamp01(x: float) -> float:
    return min(max(x, 0.0), 1.0)

# ---------------- Raw feature extraction ----------------

def extract_raw_features(sample: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """
    Returns: (breadth, miss_H_rate, miss_S_rate, out_of_P_count)

    - miss_H_rate = |uncovered_H| / |H|, where |H| = |covered_H| + |uncovered_H|
    - miss_S_rate = |uncovered_C| / |C|, where C is safety-critical (covered_C / uncovered_C)
    """
    m = sample.get("metrics", {}) or {}

    breadth = float(m.get("breadth", 0) or 0.0)
    out_of_p = float(safe_len(m.get("out_of_P")))

    covered_H = m.get("covered_H", []) or []
    uncovered_H = m.get("uncovered_H", []) or []
    H_total = len(covered_H) + len(uncovered_H)
    miss_H = float(len(uncovered_H)) / H_total if H_total > 0 else 0.0

    covered_C = m.get("covered_C", []) or []
    uncovered_C = m.get("uncovered_C", []) or []
    C_total = len(covered_C) + len(uncovered_C)
    miss_C = float(len(uncovered_C)) / C_total if C_total > 0 else 0.0

    return breadth, miss_H, miss_C, out_of_p

# ---------------- Normalization ----------------

def build_normalizer(
    doctor_samples: List[Dict[str, Any]],
    b_q: float = 0.95,
    o_q: float = 0.95
) -> Dict[str, float]:
    b_vals, o_vals = [], []
    for s in doctor_samples:
        b, miss_H, miss_S, o = extract_raw_features(s)
        b_vals.append(b)
        o_vals.append(o)
    b_vals.sort()
    o_vals.sort()

    B = max(1.0, percentile(b_vals, b_q))
    O = max(1.0, percentile(o_vals, o_q))
    return {"B_scale": float(B), "O_scale": float(O), "b_q": float(b_q), "o_q": float(o_q)}

def normalize_features(
    raw_f: Tuple[float, float, float, float],
    norm: Dict[str, float],
    cap: float = 1.0
) -> Tuple[float, float, float, float]:
    """
    cap=1.0 matches your original. If you notice lots of saturation at 1.0,
    try cap=2.0 or 3.0 to preserve signal above the 95th percentile.
    """
    b, miss_H, miss_S, o = raw_f
    b_n = min(max(b / norm["B_scale"], 0.0), cap)
    o_n = min(max(o / norm["O_scale"], 0.0), cap)
    miss_H_n = clamp01(miss_H)
    miss_S_n = clamp01(miss_S)
    return b_n, miss_H_n, miss_S_n, o_n

def extract_features(sample: Dict[str, Any], norm: Dict[str, float], cap: float = 1.0) -> Tuple[float, float, float, float]:
    return normalize_features(extract_raw_features(sample), norm, cap=cap)

# ---------------- Synthetic negatives (IDENTIFIABLE) ----------------

def make_negatives(
    sample: Dict[str, Any],
    f_pos_norm: Tuple[float, float, float, float],
    norm: Dict[str, float],
    cap: float = 1.0
) -> List[Tuple[float, float, float, float]]:
    """
    Create degraded variants with TRADEOFFS.

    We include:
      B) add 1 implausible dx: breadth +1, out_of_P +1       (purely worse)
      C) omit 1 H dx: breadth -1, miss_H += 1/|H|            (tradeoff)
      D) omit 1 C dx: breadth -1, miss_C += 1/|C|            (tradeoff)

    Why tradeoff matters:
    - If every negative is "worse in every coordinate", then for any w>=0,
      w·pos <= w·neg automatically -> hinge gradient ~0 -> weights collapse.
    """
    m = sample.get("metrics", {}) or {}

    covered_H = m.get("covered_H", []) or []
    uncovered_H = m.get("uncovered_H", []) or []
    H_total = len(covered_H) + len(uncovered_H)

    covered_C = m.get("covered_C", []) or []
    uncovered_C = m.get("uncovered_C", []) or []
    C_total = len(covered_C) + len(uncovered_C)

    b_n, miss_H_n, miss_S_n, o_n = f_pos_norm
    B, O = float(norm["B_scale"]), float(norm["O_scale"])

    # move back to raw-ish units for breadth/out_of_P editing
    b_raw = b_n * B
    o_raw = o_n * O

    dh = (1.0 / H_total) if H_total > 0 else 0.0
    dc = (1.0 / C_total) if C_total > 0 else 0.0

    negs_raw: List[Tuple[float, float, float, float]] = []

    # B) add 1 implausible diagnosis (always applicable)
    negs_raw.append((b_raw + 1.0, miss_H_n, miss_S_n, o_raw + 1.0))

    # C) omit 1 highly-likely dx (tradeoff: breadth improves, miss worsens)
    if H_total > 0:
        negs_raw.append((max(0.0, b_raw - 1.0), clamp01(miss_H_n + dh), miss_S_n, o_raw))

    # D) omit 1 safety-critical dx (tradeoff)
    if C_total > 0:
        negs_raw.append((max(0.0, b_raw - 1.0), miss_H_n, clamp01(miss_S_n + dc), o_raw))

    return [normalize_features(raw_f, norm, cap=cap) for raw_f in negs_raw]

# ---------------- Optimization ----------------

def dot(w: Tuple[float, ...], f: Tuple[float, ...]) -> float:
    return sum(wi * fi for wi, fi in zip(w, f))

def project_nonneg(w: List[float]) -> List[float]:
    return [max(0.0, wi) for wi in w]

def normalize_scale(w: List[float]) -> List[float]:
    # Fix scale by dividing by lambda (breadth weight), if possible
    if w[0] > 1e-8:
        return [wi / w[0] for wi in w]
    s = sum(abs(wi) for wi in w)
    return [wi / s for wi in w] if s > 0 else w

def fit_weights_pairwise(
    samples: List[Dict[str, Any]],
    norm: Dict[str, float],
    cap: float = 1.0,
    margin: float = 0.1,
    lr: float = 0.05,
    l2: float = 1e-3,
    iters: int = 4000,
    seed: int = 0,
    enforce_muS_ge_muH: bool = False,
) -> Tuple[float, float, float, float]:
    """
    Hinge-style updates on constraints: w·pos + margin <= w·neg
    Lower DAS is better, so doctor (pos) should score LOWER than degraded (neg).
    """
    rng = random.Random(seed)

    pos_feats = [extract_features(s, norm, cap=cap) for s in samples]
    neg_pools = [make_negatives(s, pos_feats[i], norm, cap=cap) for i, s in enumerate(samples)]

    # (lambda, mu_H, mu_S, nu)
    w = [1.0, 1.0, 1.0, 1.0]

    for t in range(iters):
        i = rng.randrange(len(pos_feats))
        f_pos = pos_feats[i]
        f_neg = rng.choice(neg_pools[i])

        viol = margin + dot(tuple(w), f_pos) - dot(tuple(w), f_neg)

        grad = [0.0, 0.0, 0.0, 0.0]
        if viol > 0:
            # push w so that w·pos gets smaller than w·neg
            for k in range(4):
                grad[k] += (f_pos[k] - f_neg[k])

        # L2
        for k in range(4):
            grad[k] += l2 * w[k]
            w[k] -= lr * grad[k]

        # projection / constraints
        w = project_nonneg(w)
        w[0] = 1.0
        if enforce_muS_ge_muH:
            w[2] = max(w[2], w[1])  # mu_S >= mu_H

        if (t + 1) % 1000 == 0:
            lr *= 0.7

    w = normalize_scale(w)
    return (float(w[0]), float(w[1]), float(w[2]), float(w[3]))

# ---------------- Scoring ----------------

def add_scores(
    data: Dict[str, Any],
    w: Tuple[float, float, float, float],
    norm: Dict[str, float],
    cap: float = 1.0
) -> Dict[str, Any]:
    total = 0.0
    n = 0
    for s in data["per_sample"]:
        b_n, miss_H_n, miss_S_n, o_n = extract_features(s, norm, cap=cap)
        das = w[0]*b_n + w[1]*miss_H_n + w[2]*miss_S_n + w[3]*o_n

        s.setdefault("metrics", {})
        s["metrics"]["das"] = float(das)
        s["metrics"]["das_components"] = {
            "breadth_norm": float(b_n),
            "miss_H_rate": float(miss_H_n),
            "miss_S_rate": float(miss_S_n),
            "out_of_P_norm": float(o_n),
        }

        total += das
        n += 1

    data.setdefault("summary", {})
    data["summary"]["das_mean"] = (total / n) if n else None
    return data

# ---------------- Generalization check (optional) ----------------

def eval_pairwise(
    samples: List[Dict[str, Any]],
    w: Tuple[float, ...],
    norm: Dict[str, float],
    cap: float = 1.0,
    margin: float = 0.1
) -> Dict[str, float]:
    n_pairs = 0
    n_correct = 0
    n_margin_correct = 0
    hinge_sum = 0.0

    for s in samples:
        f_pos = extract_features(s, norm, cap=cap)
        das_pos = dot(w, f_pos)
        for f_neg in make_negatives(s, f_pos, norm, cap=cap):
            das_neg = dot(w, f_neg)
            n_pairs += 1
            if das_pos < das_neg:
                n_correct += 1
            if das_pos + margin <= das_neg:
                n_margin_correct += 1
            hinge_sum += max(0.0, margin + das_pos - das_neg)

    return {
        "n_pairs": float(n_pairs),
        "pair_acc": (n_correct / n_pairs) if n_pairs else None,
        "pair_margin_acc": (n_margin_correct / n_pairs) if n_pairs else None,
        "mean_hinge": (hinge_sum / n_pairs) if n_pairs else None,
    }

def generalization_check(
    doctor_samples: List[Dict[str, Any]],
    seed: int,
    test_ratio: float,
    margin: float,
    lr: float,
    l2: float,
    iters: int,
    cap: float,
    enforce_muS_ge_muH: bool,
) -> Dict[str, Any]:
    train_idx, test_idx = split_indices(len(doctor_samples), test_ratio, seed)
    train = [doctor_samples[i] for i in train_idx]
    test = [doctor_samples[i] for i in test_idx]

    norm_train = build_normalizer(train)
    w_train = fit_weights_pairwise(
        train, norm_train,
        cap=cap, margin=margin, lr=lr, l2=l2, iters=iters, seed=seed,
        enforce_muS_ge_muH=enforce_muS_ge_muH,
    )

    train_eval = eval_pairwise(train, w_train, norm_train, cap=cap, margin=margin)
    test_eval = eval_pairwise(test, w_train, norm_train, cap=cap, margin=margin)

    return {
        "seed": seed,
        "test_ratio": test_ratio,
        "margin": margin,
        "lr": lr,
        "l2": l2,
        "iters": iters,
        "cap": cap,
        "train_size": len(train),
        "test_size": len(test),
        "weights_fit_on_train": {"lambda": w_train[0], "mu_H": w_train[1], "mu_S": w_train[2], "nu": w_train[3]},
        "normalizer_fit_on_train": norm_train,
        "train_eval": train_eval,
        "test_eval": test_eval,
    }

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doctor_eval", required=True)
    ap.add_argument("--raw_eval", required=True)
    ap.add_argument("--norm_eval", required=True)
    ap.add_argument("--out_prefix", default="das_scored")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", action="store_true",
                    help="If set, save learned weights, normalizer, and scored JSON files.")

    # training knobs
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--iters", type=int, default=4000)

    # normalization cap (1.0 matches your original; raise if you see saturation)
    ap.add_argument("--cap", type=float, default=1.0,
                    help="Upper cap for normalized breadth/out_of_P. Try 2 or 3 if saturation is common.")

    # optional constraint to encode your intent for safety-critical
    ap.add_argument("--enforce_muS_ge_muH", action="store_true",
                    help="If set, enforce mu_S >= mu_H during optimization.")

    # generalization check flags (OFF by default)
    ap.add_argument("--gen_check", action="store_true",
                    help="If set, run an additional train/test generalization check on doctor responses.")
    ap.add_argument("--test_ratio", type=float, default=0.2,
                    help="Doctor test split ratio for --gen_check.")

    args = ap.parse_args()

    doctor = load_json(args.doctor_eval)
    raw = load_json(args.raw_eval)
    normed = load_json(args.norm_eval)

    norm_stats = build_normalizer(doctor["per_sample"])
    w = fit_weights_pairwise(
        doctor["per_sample"], norm_stats,
        cap=args.cap,
        seed=args.seed,
        margin=args.margin, lr=args.lr, l2=args.l2, iters=args.iters,
        enforce_muS_ge_muH=args.enforce_muS_ge_muH,
    )

    doctor = add_scores(doctor, w, norm_stats, cap=args.cap)
    raw = add_scores(raw, w, norm_stats, cap=args.cap)
    normed = add_scores(normed, w, norm_stats, cap=args.cap)

    print("Learned weights (lambda, mu_H, mu_S, nu):", w)
    print("Mean DAS:")
    print("  Doctor:", doctor["summary"]["das_mean"])
    print("  Raw   :", raw["summary"]["das_mean"])
    print("  Norm  :", normed["summary"]["das_mean"])

    if args.save:
        save_json(norm_stats, f"{args.out_prefix}_normalizer.json")
        save_json(
            {"lambda": w[0], "mu_H": w[1], "mu_S": w[2], "nu": w[3], "normalizer": norm_stats, "cap": args.cap},
            f"{args.out_prefix}_weights.json",
        )
        save_json(doctor, f"{args.out_prefix}_doctor.json")
        save_json(raw, f"{args.out_prefix}_raw.json")
        save_json(normed, f"{args.out_prefix}_norm.json")

    if args.gen_check:
        gen = generalization_check(
            doctor_samples=doctor["per_sample"],
            seed=args.seed,
            test_ratio=args.test_ratio,
            margin=args.margin,
            lr=args.lr,
            l2=args.l2,
            iters=args.iters,
            cap=args.cap,
            enforce_muS_ge_muH=args.enforce_muS_ge_muH,
        )
        print("\nGeneralization check (doctor train/test):")
        print("  Train size:", gen["train_size"], "Test size:", gen["test_size"])
        print("  Weights (fit on train):", gen["weights_fit_on_train"])
        print("  Train pairwise:", gen["train_eval"])
        print("  Test  pairwise:", gen["test_eval"])

if __name__ == "__main__":
    main()