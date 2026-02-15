"""
Compare doctor vs LLM evaluation results on the same dataset.

Computes:
  - dx_agreement (primary): Diagnosis-level semantic agreement between
    extracted diagnoses, using SemanticMatcher from evaluate.py.
  - set_agreement (supplementary): Set-level agreement on H/C coverage
    and P/H precision using exact string matching on canonical names.

Usage:
  python llm_doctor.py \
    --doctor_path ../results/HCM-3k/reference/eval_doctor.json \
    --llm_path ../results/HCM-3k/exp_4/eval_raw_1_gpt-4.1-mini.json \
    --output_path ../results/HCM-3k/comparison_doctor_vs_llm.json \
    --sem_match_model gpt-4.1-mini \
    --sem_cache_path ../results/HCM-3k/sem_cache_comparison.json
"""

import json
import argparse
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

from evaluate import (
    SemanticMatcher,
    normalize_text,
    save_json,
)


# ============================================================
# I/O helpers
# ============================================================

def load_eval_json(path: str) -> Dict[str, Any]:
    """Load an evaluation JSON produced by evaluate.py."""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict) or "per_sample" not in obj:
        raise ValueError(f"Expected eval JSON with 'per_sample' key in {path}")
    return obj


# ============================================================
# Set helpers
# ============================================================

def extract_covered_names(covered_list: List[Dict[str, Any]], key: str) -> Set[str]:
    """Extract canonical set-item names from covered_H or covered_C lists.

    covered_H items have key "h", covered_C items have key "c".
    """
    names = set()
    for item in covered_list:
        if isinstance(item, dict) and key in item:
            names.add(item[key])
    return names


def jaccard(a: Set[str], b: Set[str]) -> Optional[float]:
    """Jaccard similarity.  Returns None if both sets are empty."""
    if not a and not b:
        return None
    return len(a & b) / len(a | b)


def overlap_coeff(a: Set[str], b: Set[str]) -> Optional[float]:
    """Overlap coefficient.  Returns None if either set is empty."""
    if not a or not b:
        return None
    return len(a & b) / min(len(a), len(b))


# ============================================================
# dx_agreement  (Group 2 – primary)
# ============================================================

def compute_dx_agreement(
    doctor_dx: List[str],
    llm_dx: List[str],
    matcher: SemanticMatcher,
    max_pairs_per_call: int = 50,
) -> Dict[str, Any]:
    """Compute diagnosis-level semantic agreement via greedy 1-to-1 matching."""

    if not doctor_dx and not llm_dx:
        return {
            "doctor_extracted": [],
            "llm_extracted": [],
            "matched_pairs": [],
            "doctor_unique": [],
            "llm_unique": [],
            "shared_count": 0,
            "semantic_jaccard": None,
            "overlap_coefficient": None,
        }

    # Pre-compute all pairwise semantic decisions (batched, cached)
    pairs = [(d, l) for d in doctor_dx for l in llm_dx]
    if pairs:
        matcher.batch_match_pairs(pairs, max_pairs_per_call=max_pairs_per_call)

    # Greedy 1-to-1 matching: for each doctor dx, find first unmatched LLM dx
    matched_pairs: List[Dict[str, Any]] = []
    matched_llm_indices: Set[int] = set()

    for d_dx in doctor_dx:
        for j, l_dx in enumerate(llm_dx):
            if j in matched_llm_indices:
                continue
            k = matcher._key(d_dx, l_dx)
            info = matcher.cache.get(k, {})
            if info.get("match", False):
                matched_pairs.append({
                    "doctor_dx": d_dx,
                    "llm_dx": l_dx,
                    "match_info": info,
                })
                matched_llm_indices.add(j)
                break

    matched_doctor_set = {p["doctor_dx"] for p in matched_pairs}
    matched_llm_set = {p["llm_dx"] for p in matched_pairs}

    doctor_unique = [d for d in doctor_dx if d not in matched_doctor_set]
    llm_unique = [l for l in llm_dx if l not in matched_llm_set]

    shared = len(matched_pairs)
    total_union = len(doctor_dx) + len(llm_dx) - shared

    sem_jaccard = shared / total_union if total_union > 0 else None
    ov = shared / min(len(doctor_dx), len(llm_dx)) if min(len(doctor_dx), len(llm_dx)) > 0 else None

    return {
        "doctor_extracted": doctor_dx,
        "llm_extracted": llm_dx,
        "matched_pairs": matched_pairs,
        "doctor_unique": doctor_unique,
        "llm_unique": llm_unique,
        "shared_count": shared,
        "semantic_jaccard": sem_jaccard,
        "overlap_coefficient": ov,
    }


# ============================================================
# set_agreement  (Group 1 – supplementary)
# ============================================================

def compute_set_agreement(
    doctor_metrics: Dict[str, Any],
    llm_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare coverage of H / C sets and P / H precision.

    Uses exact string equality on canonical reference-set names (both
    eval files share the same judge_dx_space).
    """

    # Covered H / C from each side
    doc_H = extract_covered_names(doctor_metrics.get("covered_H", []), "h")
    llm_H = extract_covered_names(llm_metrics.get("covered_H", []), "h")

    doc_C = extract_covered_names(doctor_metrics.get("covered_C", []), "c")
    llm_C = extract_covered_names(llm_metrics.get("covered_C", []), "c")

    # H-coverage agreement
    h_both = sorted(doc_H & llm_H)
    h_doctor_only = sorted(doc_H - llm_H)
    h_llm_only = sorted(llm_H - doc_H)
    h_jac = jaccard(doc_H, llm_H)

    # C-coverage agreement
    c_both = sorted(doc_C & llm_C)
    c_doctor_only = sorted(doc_C - llm_C)
    c_llm_only = sorted(llm_C - doc_C)
    c_jac = jaccard(doc_C, llm_C)

    # Scalar comparisons
    plaus_doc = doctor_metrics.get("plausibility")
    plaus_llm = llm_metrics.get("plausibility")
    plaus_diff = None
    if plaus_doc is not None and plaus_llm is not None:
        plaus_diff = plaus_llm - plaus_doc

    hprec_doc = doctor_metrics.get("h_precision")
    hprec_llm = llm_metrics.get("h_precision")
    hprec_diff = None
    if hprec_doc is not None and hprec_llm is not None:
        hprec_diff = hprec_llm - hprec_doc

    return {
        "plausibility_doctor": plaus_doc,
        "plausibility_llm": plaus_llm,
        "plausibility_diff": plaus_diff,

        "covered_H_both": h_both,
        "covered_H_doctor_only": h_doctor_only,
        "covered_H_llm_only": h_llm_only,
        "h_coverage_jaccard": h_jac,

        "covered_C_both": c_both,
        "covered_C_doctor_only": c_doctor_only,
        "covered_C_llm_only": c_llm_only,
        "c_coverage_jaccard": c_jac,

        "h_precision_doctor": hprec_doc,
        "h_precision_llm": hprec_llm,
        "h_precision_diff": hprec_diff,
    }


# ============================================================
# Summary aggregation
# ============================================================

def build_summary(per_sample: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-sample results into summary statistics."""

    n = len(per_sample)
    if n == 0:
        return {"num_samples": 0, "dx_agreement": {}, "set_agreement": {}}

    # -- dx_agreement accumulators --
    sum_sem_jac = 0.0;  n_sem_jac = 0
    sum_ov = 0.0;       n_ov = 0
    sum_doc_unique = 0
    sum_llm_unique = 0
    sum_shared = 0

    # -- set_agreement accumulators --
    sum_plaus_doc = 0.0;  n_plaus_doc = 0
    sum_plaus_llm = 0.0;  n_plaus_llm = 0
    sum_h_jac = 0.0;      n_h_jac = 0
    sum_c_jac = 0.0;      n_c_jac = 0
    sum_hprec_doc = 0.0;  n_hprec_doc = 0
    sum_hprec_llm = 0.0;  n_hprec_llm = 0
    cnt_c_missed_by_llm = 0
    cnt_c_caught_by_llm = 0

    for s in per_sample:
        dx = s.get("dx_agreement", {})
        sa = s.get("set_agreement", {})

        # dx_agreement
        if dx.get("semantic_jaccard") is not None:
            sum_sem_jac += dx["semantic_jaccard"]
            n_sem_jac += 1
        if dx.get("overlap_coefficient") is not None:
            sum_ov += dx["overlap_coefficient"]
            n_ov += 1
        sum_doc_unique += len(dx.get("doctor_unique", []))
        sum_llm_unique += len(dx.get("llm_unique", []))
        sum_shared += dx.get("shared_count", 0)

        # set_agreement
        if sa.get("plausibility_doctor") is not None:
            sum_plaus_doc += sa["plausibility_doctor"]
            n_plaus_doc += 1
        if sa.get("plausibility_llm") is not None:
            sum_plaus_llm += sa["plausibility_llm"]
            n_plaus_llm += 1
        if sa.get("h_coverage_jaccard") is not None:
            sum_h_jac += sa["h_coverage_jaccard"]
            n_h_jac += 1
        if sa.get("c_coverage_jaccard") is not None:
            sum_c_jac += sa["c_coverage_jaccard"]
            n_c_jac += 1
        if sa.get("h_precision_doctor") is not None:
            sum_hprec_doc += sa["h_precision_doctor"]
            n_hprec_doc += 1
        if sa.get("h_precision_llm") is not None:
            sum_hprec_llm += sa["h_precision_llm"]
            n_hprec_llm += 1
        if sa.get("covered_C_doctor_only"):
            cnt_c_missed_by_llm += 1
        if sa.get("covered_C_llm_only"):
            cnt_c_caught_by_llm += 1

    safe_div = lambda s, c: (s / c) if c else None

    return {
        "num_samples": n,
        "dx_agreement": {
            "mean_semantic_jaccard": safe_div(sum_sem_jac, n_sem_jac),
            "mean_overlap_coefficient": safe_div(sum_ov, n_ov),
            "mean_doctor_unique_count": sum_doc_unique / n,
            "mean_llm_unique_count": sum_llm_unique / n,
            "mean_shared_count": sum_shared / n,
        },
        "set_agreement": {
            "mean_plausibility_doctor": safe_div(sum_plaus_doc, n_plaus_doc),
            "mean_plausibility_llm": safe_div(sum_plaus_llm, n_plaus_llm),
            "mean_plausibility_diff": (
                safe_div(sum_plaus_llm, n_plaus_llm) - safe_div(sum_plaus_doc, n_plaus_doc)
                if n_plaus_doc and n_plaus_llm else None
            ),
            "mean_h_coverage_jaccard": safe_div(sum_h_jac, n_h_jac),
            "mean_c_coverage_jaccard": safe_div(sum_c_jac, n_c_jac),
            "mean_h_precision_doctor": safe_div(sum_hprec_doc, n_hprec_doc),
            "mean_h_precision_llm": safe_div(sum_hprec_llm, n_hprec_llm),
            "mean_h_precision_diff": (
                safe_div(sum_hprec_llm, n_hprec_llm) - safe_div(sum_hprec_doc, n_hprec_doc)
                if n_hprec_doc and n_hprec_llm else None
            ),
            "c_missed_by_llm_only_rate": cnt_c_missed_by_llm / n,
            "c_caught_by_llm_only_rate": cnt_c_caught_by_llm / n,
        },
    }


# ============================================================
# CLI + main
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Compare doctor vs LLM evaluation results: dx_agreement + set_agreement."
    )
    ap.add_argument("--doctor_path", type=str, required=True,
                    help="Path to doctor eval JSON (output of evaluate.py).")
    ap.add_argument("--llm_path", type=str, required=True,
                    help="Path to LLM eval JSON (output of evaluate.py).")
    ap.add_argument("--output_path", type=str, required=True,
                    help="Where to save comparison JSON.")

    ap.add_argument("--sem_match_model", type=str, default="gpt-4.1-mini",
                    help="Model for semantic matching of extracted diagnoses.")
    ap.add_argument("--sem_cache_path", type=str, default="",
                    help="Optional path to save/load semantic match cache JSON.")
    ap.add_argument("--sem_max_pairs_per_call", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max", type=int, default=0,
                    help="Max number of samples to process (0 = all).")
    ap.add_argument("--save_every", type=int, default=100,
                    help="Checkpoint every N samples.")

    return ap.parse_args()


def main():
    args = parse_args()

    print(f"Loading doctor eval:  {args.doctor_path}")
    doctor_eval = load_eval_json(args.doctor_path)
    print(f"Loading LLM eval:     {args.llm_path}")
    llm_eval = load_eval_json(args.llm_path)

    doc_samples = doctor_eval["per_sample"]
    llm_samples = llm_eval["per_sample"]

    # Index maps for O(1) lookup
    doc_by_idx = {s["index"]: s for s in doc_samples if isinstance(s, dict) and "index" in s}
    llm_by_idx = {s["index"]: s for s in llm_samples if isinstance(s, dict) and "index" in s}

    common_indices = sorted(set(doc_by_idx.keys()) & set(llm_by_idx.keys()))
    print(f"Doctor samples: {len(doc_by_idx)}, "
          f"LLM samples: {len(llm_by_idx)}, "
          f"Common indices: {len(common_indices)}")

    matcher = SemanticMatcher(
        model=args.sem_match_model,
        temperature=args.temperature,
        cache_path=args.sem_cache_path or None,
    )

    per_sample: List[Dict[str, Any]] = []

    if args.max > 0:
        common_indices = common_indices[:args.max]

    for idx in tqdm(common_indices, desc="Comparing"):
        doc_s = doc_by_idx[idx]
        llm_s = llm_by_idx[idx]

        # Both must have valid P/H/C and metrics
        if not doc_s.get("has_pxhx") or not llm_s.get("has_pxhx"):
            continue

        doc_m = doc_s.get("metrics")
        llm_m = llm_s.get("metrics")
        if not isinstance(doc_m, dict) or not isinstance(llm_m, dict):
            continue

        # ---- dx_agreement (primary) ----
        doc_dx = doc_m.get("extracted_diagnoses_top_k",
                           doc_m.get("extracted_diagnoses", []))
        llm_dx = llm_m.get("extracted_diagnoses_top_k",
                           llm_m.get("extracted_diagnoses", []))

        dx_ag = compute_dx_agreement(
            doc_dx, llm_dx, matcher,
            max_pairs_per_call=args.sem_max_pairs_per_call,
        )

        # ---- set_agreement (supplementary) ----
        set_ag = compute_set_agreement(doc_m, llm_m)

        per_sample.append({
            "index": idx,
            "input": doc_s.get("input", ""),
            "reference_diagnosis": doc_s.get("reference_diagnosis", ""),
            "judge_dx_space": doc_s.get("judge_dx_space", {}),
            "dx_agreement": dx_ag,
            "set_agreement": set_ag,
        })

        # Periodic checkpoint
        if len(per_sample) % args.save_every == 0:
            if args.sem_cache_path:
                matcher.save_cache()
            summary = build_summary(per_sample)
            output = {"summary": summary, "per_sample": per_sample}
            save_json(args.output_path, output)
            print(f"  [checkpoint] {len(per_sample)} samples saved")

    # Final save
    if args.sem_cache_path:
        matcher.save_cache()

    summary = build_summary(per_sample)
    output = {"summary": summary, "per_sample": per_sample}
    save_json(args.output_path, output)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(per_sample)} comparisons to {args.output_path}")


if __name__ == "__main__":
    main()
