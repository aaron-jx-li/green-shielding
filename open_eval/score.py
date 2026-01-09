"""
Fit lambda, mu, nu from doctor responses, then compute DIS for:
  - doctor responses
  - raw model responses
  - normalized model responses
"""

import argparse, json, random
from typing import Dict, Any, List, Tuple

# ---------------- I/O ----------------

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# ---------------- Feature extraction ----------------

def safe_len(x):
    return len(x) if x is not None else 0

def extract_features(sample):
    m = sample.get("metrics", {})
    breadth = float(m.get("breadth", 0) or 0)
    outP = float(safe_len(m.get("out_of_P")))
    uncovered_H = m.get("uncovered_H")
    miss_core = 1.0 if safe_len(uncovered_H) > 0 else 0.0
    return breadth, miss_core, outP

# ---------------- Synthetic negatives ----------------

def make_negatives(f_pos):
    b, miss, o = f_pos
    return [
        (b + 1, miss, o + 1),               # add implausible dx
        (b + 1, miss, o),                   # add plausible dx
        (max(0, b - 1), 1, o),              # drop core dx
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

def fit_weights_pairwise(samples, margin=1.0, lr=0.01, l2=1e-3, iters=2000, seed=0):
    random.seed(seed)
    pos_feats = [extract_features(s) for s in samples]
    neg_pools = [make_negatives(f) for f in pos_feats]

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

        if (t + 1) % 500 == 0:
            lr *= 0.7

    return tuple(normalize_scale(w))

# ---------------- Scoring ----------------

def add_dis_scores(data, w):
    total = 0.0
    n = 0
    for s in data["per_sample"]:
        b, miss, o = extract_features(s)
        dis = w[0]*b + w[1]*miss + w[2]*o
        s.setdefault("metrics", {})
        s["metrics"]["dis"] = dis
        s["metrics"]["dis_terms"] = {"breadth": b, "miss_core": miss, "out_of_P": o}
        total += dis
        n += 1

    data.setdefault("summary", {})
    data["summary"]["dis_mean"] = total / n if n > 0 else None
    return data

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doctor_eval", required=True)
    ap.add_argument("--raw_eval", required=True)
    ap.add_argument("--norm_eval", required=True)
    ap.add_argument("--out_prefix", default="dis_scored")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    doctor = load_json(args.doctor_eval)
    raw = load_json(args.raw_eval)
    norm = load_json(args.norm_eval)

    w = fit_weights_pairwise(doctor["per_sample"], seed=args.seed)

    print("Learned weights:", w)

    weights_obj = {"lambda": w[0], "mu": w[1], "nu": w[2]}
    save_json(weights_obj, f"{args.out_prefix}_weights.json")

    doctor_scored = add_dis_scores(doctor, w)
    raw_scored = add_dis_scores(raw, w)
    norm_scored = add_dis_scores(norm, w)

    save_json(doctor_scored, f"{args.out_prefix}_doctor.json")
    save_json(raw_scored, f"{args.out_prefix}_raw.json")
    save_json(norm_scored, f"{args.out_prefix}_norm.json")

    print("Mean DIS:")
    print("  Doctor:", doctor_scored["summary"]["dis_mean"])
    print("  Raw:", raw_scored["summary"]["dis_mean"])
    print("  Norm:", norm_scored["summary"]["dis_mean"])

if __name__ == "__main__":
    main()
