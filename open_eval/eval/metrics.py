from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from open_eval.core.io import save_json
from open_eval.core.text import normalize_text

if TYPE_CHECKING:
    from open_eval.eval.semantic_match import SemanticMatcher


def dedup_preserve_order(xs: List[str]) -> List[str]:
    out, seen = [], set()
    for x in xs:
        k = normalize_text(x)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(x.strip())
    return out


def compute_breadth_metrics(extracted: List[str], P: List[str]) -> Dict[str, Any]:
    breadth = len(extracted)
    norm_breadth = None
    if isinstance(P, list) and len(P) > 0:
        norm_breadth = breadth / len(P)
    return {
        "breadth": breadth,
        "normalized_breadth": norm_breadth,
        "P_size": len(P) if isinstance(P, list) else None,
    }


def compute_grounding_metrics(grounding_obj: Dict[str, Any], extracted: List[str]) -> Dict[str, Any]:
    per = grounding_obj.get("per_diagnosis", [])
    if not isinstance(per, list):
        per = []

    per_by_norm = {normalize_text(d.get("diagnosis", "")): d for d in per if isinstance(d, dict)}

    ordered = []
    for dx in extracted:
        d = per_by_norm.get(normalize_text(dx))
        if d is None:
            ordered.append(
                {
                    "diagnosis": dx,
                    "input_support_quotes": [],
                    "has_support": False,
                    "indirect_inference_claims": [],
                    "has_indirect_inference": False,
                }
            )
        else:
            ordered.append(
                {
                    "diagnosis": dx,
                    "input_support_quotes": d.get("input_support_quotes", [])
                    if isinstance(d.get("input_support_quotes", []), list)
                    else [],
                    "has_support": bool(d.get("has_support", False)),
                    "indirect_inference_claims": d.get("indirect_inference_claims", [])
                    if isinstance(d.get("indirect_inference_claims", []), list)
                    else [],
                    "has_indirect_inference": bool(d.get("has_indirect_inference", False)),
                }
            )

    n = len(ordered)
    if n == 0:
        return {
            "per_diagnosis": [],
            "support_rate": 1.0,
            "indirect_inference_rate": 0.0,
        }

    support_rate = sum(1 for d in ordered if d["has_support"]) / n
    indirect_inference_rate = sum(1 for d in ordered if d["has_indirect_inference"]) / n

    return {
        "per_diagnosis": ordered,
        "support_rate": support_rate,
        "indirect_inference_rate": indirect_inference_rate,
    }


def recompute_aggregates_from_per_sample(per_sample: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_total = 0
    n_with_pxhx = 0

    sum_plausibility = 0.0

    sum_hcov = 0.0
    n_hcov_defined = 0

    sum_ccov = 0.0
    n_ccov_defined = 0

    sum_h_precision = 0.0

    sum_support_rate = 0.0
    sum_indirect_inference_rate = 0.0

    sum_breadth = 0.0
    sum_norm_breadth = 0.0
    n_norm_breadth_defined = 0

    n_uncertain = 0

    for r in per_sample:
        if not isinstance(r, dict):
            continue

        if isinstance(r.get("index", None), int):
            n_total += 1

        if not r.get("has_pxhx", False):
            continue

        metrics = r.get("metrics", None)
        if not isinstance(metrics, dict):
            continue

        n_with_pxhx += 1

        plaus = metrics.get("plausibility", None)
        if plaus is not None:
            sum_plausibility += float(plaus)

        hc = metrics.get("h_coverage", None)
        if hc is not None:
            sum_hcov += float(hc)
            n_hcov_defined += 1

        cc = metrics.get("c_coverage", None)
        if cc is not None:
            sum_ccov += float(cc)
            n_ccov_defined += 1

        hp = metrics.get("h_precision", None)
        if hp is not None:
            sum_h_precision += float(hp)

        sum_support_rate += float(metrics.get("support_rate", 0.0))
        sum_indirect_inference_rate += float(metrics.get("indirect_inference_rate", 0.0))

        sum_breadth += float(metrics.get("breadth", 0.0))
        nb = metrics.get("normalized_breadth", None)
        if nb is not None:
            sum_norm_breadth += float(nb)
            n_norm_breadth_defined += 1

        if bool(metrics.get("uncertainty_flag", False)):
            n_uncertain += 1

    return {
        "n_total": n_total,
        "n_with_pxhx": n_with_pxhx,
        "sum_plausibility": sum_plausibility,
        "sum_hcov": sum_hcov,
        "n_hcov_defined": n_hcov_defined,
        "sum_ccov": sum_ccov,
        "n_ccov_defined": n_ccov_defined,
        "sum_h_precision": sum_h_precision,
        "sum_support_rate": sum_support_rate,
        "sum_indirect_inference_rate": sum_indirect_inference_rate,
        "sum_breadth": sum_breadth,
        "sum_norm_breadth": sum_norm_breadth,
        "n_norm_breadth_defined": n_norm_breadth_defined,
        "n_uncertain": n_uncertain,
    }


def make_summary(agg: Dict[str, Any], matcher_cache_entries: int, models: Dict[str, str]) -> Dict[str, Any]:
    n_total = agg["n_total"]
    n_with_pxhx = agg["n_with_pxhx"]

    return {
        "num_total_processed": n_total,
        "num_with_pxhx": n_with_pxhx,
        "rate_with_pxhx": (n_with_pxhx / n_total) if n_total else None,
        "mean_plausibility": (agg["sum_plausibility"] / n_with_pxhx) if n_with_pxhx else None,
        "mean_h_coverage": (agg["sum_hcov"] / agg["n_hcov_defined"]) if agg["n_hcov_defined"] else None,
        "h_coverage_defined_count": agg["n_hcov_defined"],
        "mean_c_coverage": (agg["sum_ccov"] / agg["n_ccov_defined"]) if agg["n_ccov_defined"] else None,
        "c_coverage_defined_count": agg["n_ccov_defined"],
        "mean_h_precision": (agg["sum_h_precision"] / n_with_pxhx) if n_with_pxhx else None,
        "mean_support_rate": (agg["sum_support_rate"] / n_with_pxhx) if n_with_pxhx else None,
        "mean_indirect_inference_rate": (agg["sum_indirect_inference_rate"] / n_with_pxhx) if n_with_pxhx else None,
        "mean_breadth": (agg["sum_breadth"] / n_with_pxhx) if n_with_pxhx else None,
        "mean_normalized_breadth": (agg["sum_norm_breadth"] / agg["n_norm_breadth_defined"])
        if agg["n_norm_breadth_defined"]
        else None,
        "normalized_breadth_defined_count": agg["n_norm_breadth_defined"],
        "uncertainty_rate": (agg["n_uncertain"] / n_with_pxhx) if n_with_pxhx else None,
        "models": models,
        "sem_match_cache_entries": matcher_cache_entries,
    }


def checkpoint_save(
    path: str,
    per_sample: List[Dict[str, Any]],
    agg: Dict[str, Any],
    matcher: SemanticMatcher,
    models: Dict[str, str],
) -> None:
    summary = make_summary(agg, len(matcher.cache), models)
    output = {"summary": summary, "per_sample": per_sample}
    save_json(path, output)
    print(f"[checkpoint] Saved {len(per_sample)} entries to {path}")


def extract_P_H_C_from_truth_record(truth_rec: Dict[str, Any]) -> Tuple[Optional[List[str]], Optional[List[str]], Optional[List[str]]]:
    """
    Supports a few possible schemas:
    - truth_rec["ground_truth_space_majority"]["plausible_set"/"highly_likely_set"/"cannot_miss_set"]
    - truth_rec["ground_truth_space"]["plausible_set"/"highly_likely_set"/"cannot_miss_set"]
    - truth_rec["judge_dx_space"]["plausible_set"/"highly_likely_set"/"cannot_miss_set"]  (legacy compatibility)
    """
    if not isinstance(truth_rec, dict):
        return None, None, None

    if isinstance(truth_rec.get("ground_truth_space_majority"), dict):
        gt = truth_rec["ground_truth_space_majority"]
        P = gt.get("plausible_set", None)
        H = gt.get("highly_likely_set", None)
        C = gt.get("cannot_miss_set", None)
        return (P if isinstance(P, list) else None), (H if isinstance(H, list) else None), (C if isinstance(C, list) else None)

    if isinstance(truth_rec.get("ground_truth_space"), dict):
        gt = truth_rec["ground_truth_space"]
        P = gt.get("plausible_set", None)
        H = gt.get("highly_likely_set", None)
        C = gt.get("cannot_miss_set", None)
        return (P if isinstance(P, list) else None), (H if isinstance(H, list) else None), (C if isinstance(C, list) else None)

    if isinstance(truth_rec.get("judge_dx_space"), dict):
        gt = truth_rec["judge_dx_space"]
        P = gt.get("plausible_set", None)
        H = gt.get("highly_likely_set", None)
        C = gt.get("cannot_miss_set", None)
        return (P if isinstance(P, list) else None), (H if isinstance(H, list) else None), (C if isinstance(C, list) else None)

    return None, None, None
