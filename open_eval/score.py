"""
Fit lambda, mu, nu from doctor responses, then compute DAS for:
  - doctor responses
  - raw model responses
  - normalized model responses

Uses global percentile normalization for breadth and out_of_P,
and core-miss as a rate.
"""

import argparse, json, random
from typing import List, Dict, Any, Tuple

# ---------------- I/O ----------------

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# ---------------- Utilities ----------------

def safe_len(x):
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

# ---------------- Raw feature extraction ----------------

def extract_raw_features(sample) -> Tuple[float, float, float]:
    m = sample.get("metrics", {})
    b = float(m.get("breadth", 0) or 0.0)
    o = float(safe_len(m.get("out_of_P")))
    covered_H = m.get("covered_H", []) or []
    uncovered = m.get("uncovered_H", []) or []
    H_total = len(covered_H) + len(uncovered)
    miss = float(len(uncovered)) / H_total if H_total > 0 else 0.0
    return b, miss, o

# ---------------- Normalization ----------------

def build_normalizer(doctor_samples: List[Dict[str, Any]], b_q=0.95, o_q=0.95):
    b_vals, o_vals = [], []
    for s in doctor_samples:
        b, miss, o = extract_raw_features(s)
        b_vals.append(b)
        o_vals.append(o)
    b_vals.sort()
    o_vals.sort()
    B = max(1.0, percentile(b_vals, b_q))
    O = max(1.0, percentile(o_vals, o_q))
    return {"B_scale": float(B), "O_scale": float(O), "b_q": b_q, "o_q": o_q}

def normalize_features(raw_f, norm):
    b, miss, o = raw_f
    b_n = min(max(b / norm["B_scale"], 0.0), 1.0)
    o_n = min(max(o / norm["O_scale"], 0.0), 1.0)
    miss_n = min(max(miss, 0.0), 1.0)
    return b_n, miss_n, o_n

def extract_features(sample, norm):
    return normalize_features(extract_raw_features(sample), norm)

# ---------------- Synthetic negatives ----------------

def make_negatives(f_pos_norm, norm):
    b_n, miss_n, o_n = f_pos_norm
    B, O = norm["B_scale"], norm["O_scale"]
    b_raw, o_raw = b_n * B, o_n * O

    return [
        normalize_features((b_raw + 1.0, miss_n, o_raw + 1.0), norm),
        normalize_features((b_raw + 1.0, miss_n, o_raw), norm),
        normalize_features((max(0.0, b_raw - 1.0), 1.0, o_raw), norm),
    ]

# ---------------- Optimization ----------------

def dot(w, f):
    return sum(wi * fi for wi, fi in zip(w, f))

def project_nonneg(w):
    return [max(0.0, wi) for wi in w]

def normalize_scale(w):
    if w[0] > 1e-8:
        return [wi / w[0] for wi in w]
    s = sum(abs(wi) for wi in w)
    return [wi / s for wi in w] if s > 0 else w

def fit_weights_pairwise(samples, norm, margin=0.1, lr=0.05, l2=1e-3, iters=4000, seed=0):
    random.seed(seed)
    pos_feats = [extract_features(s, norm) for s in samples]
    neg_pools = [make_negatives(f, norm) for f in pos_feats]
    w = [1.0, 1.0, 1.0]

    for t in range(iters):
        i = random.randrange(len(pos_feats))
        f_pos = pos_feats[i]
        f_neg = random.choice(neg_pools[i])
        viol = margin + dot(w, f_pos) - dot(w, f_neg)
        grad = [0.0, 0.0, 0.0]
        if viol > 0:
            for k in range(3):
                grad[k] += f_pos[k] - f_neg[k]
        for k in range(3):
            grad[k] += l2 * w[k]
            w[k] -= lr * grad[k]
        w = project_nonneg(w)
        if (t + 1) % 1000 == 0:
            lr *= 0.7

    return tuple(normalize_scale(w))

# ---------------- Scoring ----------------

def add_scores(data, w, norm):
    total = 0.0
    n = 0
    for s in data["per_sample"]:
        b_n, miss_n, o_n = extract_features(s, norm)
        das = w[0]*b_n + w[1]*miss_n + w[2]*o_n
        s.setdefault("metrics", {})
        s["metrics"]["das"] = das
        s["metrics"]["das_components"] = {
            "breadth_norm": b_n,
            "miss_core_rate": miss_n,
            "out_of_P_norm": o_n,
        }
        total += das
        n += 1
    data.setdefault("summary", {})
    data["summary"]["das_mean"] = total / n if n else None
    return data

# ---------------- Generalization check (additional; does not change main pipeline) ----------------

def eval_pairwise(samples, w, norm, margin=0.1) -> Dict[str, float]:
    n_pairs = 0
    n_correct = 0
    n_margin_correct = 0
    hinge_sum = 0.0

    for s in samples:
        f_pos = extract_features(s, norm)
        das_pos = dot(w, f_pos)
        for f_neg in make_negatives(f_pos, norm):
            das_neg = dot(w, f_neg)
            n_pairs += 1
            if das_pos < das_neg:
                n_correct += 1
            if das_pos + margin <= das_neg:
                n_margin_correct += 1
            hinge_sum += max(0.0, margin + das_pos - das_neg)

    return {
        "n_pairs": float(n_pairs),
        "pair_acc": n_correct / n_pairs if n_pairs else None,
        "pair_margin_acc": n_margin_correct / n_pairs if n_pairs else None,
        "mean_hinge": hinge_sum / n_pairs if n_pairs else None,
    }

def generalization_check(doctor_samples, seed: int, test_ratio: float,
                         margin: float, lr: float, l2: float, iters: int) -> Dict[str, Any]:
    train_idx, test_idx = split_indices(len(doctor_samples), test_ratio, seed)
    train = [doctor_samples[i] for i in train_idx]
    test = [doctor_samples[i] for i in test_idx]

    # IMPORTANT: for a true generalization check, build normalizer on TRAIN only
    norm_train = build_normalizer(train)

    w_train = fit_weights_pairwise(
        train, norm_train,
        margin=margin, lr=lr, l2=l2, iters=iters, seed=seed
    )

    train_eval = eval_pairwise(train, w_train, norm_train, margin=margin)
    test_eval = eval_pairwise(test, w_train, norm_train, margin=margin)

    return {
        "seed": seed,
        "test_ratio": test_ratio,
        "margin": margin,
        "lr": lr,
        "l2": l2,
        "iters": iters,
        "train_size": len(train),
        "test_size": len(test),
        "weights_fit_on_train": {"lambda": w_train[0], "mu": w_train[1], "nu": w_train[2]},
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
    ap.add_argument("--save", action="store_true", help="If set, save learned weights, normalizer, and scored JSON files.")

    # knobs for training
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--iters", type=int, default=4000)

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

    w = fit_weights_pairwise(doctor["per_sample"], norm_stats, seed=args.seed)

    doctor = add_scores(doctor, w, norm_stats)
    raw = add_scores(raw, w, norm_stats)
    normed = add_scores(normed, w, norm_stats)
    if args.save:
        save_json(norm_stats, f"{args.out_prefix}_normalizer.json")
        save_json({"lambda": w[0], "mu": w[1], "nu": w[2], "normalizer": norm_stats}, f"{args.out_prefix}_weights.json")
        save_json(doctor, f"{args.out_prefix}_doctor.json")
        save_json(raw, f"{args.out_prefix}_raw.json")
        save_json(normed, f"{args.out_prefix}_norm.json")

    print("Learned weights:", w)
    print("Mean DAS:")
    print("  Doctor:", doctor["summary"]["das_mean"])
    print("  Raw:", raw["summary"]["das_mean"])
    print("  Norm:", normed["summary"]["das_mean"])

    # -------- additional generalization check (optional) --------
    if args.gen_check:
        gen = generalization_check(
            doctor_samples=doctor["per_sample"],
            seed=args.seed,
            test_ratio=args.test_ratio,
            margin=args.margin,
            lr=args.lr,
            l2=args.l2,
            iters=args.iters
        )
        print("\nGeneralization check (doctor train/test):")
        print("  Train size:", gen["train_size"], "Test size:", gen["test_size"])
        print("  Weights (fit on train):", gen["weights_fit_on_train"])
        print("  Train pairwise:", gen["train_eval"])
        print("  Test  pairwise:", gen["test_eval"])

if __name__ == "__main__":
    main()
