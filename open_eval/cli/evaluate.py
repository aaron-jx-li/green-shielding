from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from open_eval.core.io import (
    build_done_index_set,
    load_existing_output,
    load_json_array,
)
from open_eval.eval.metrics import (
    checkpoint_save,
    compute_breadth_metrics,
    compute_grounding_metrics,
    dedup_preserve_order,
    extract_P_H_C_from_truth_record,
    make_summary,
    recompute_aggregates_from_per_sample,
)
from open_eval.eval.semantic_match import SemanticMatcher
from open_eval.llm.client import call_json_judge
from open_eval.llm.prompts import DX_EXTRACT_SYSTEM, GROUNDING_SYSTEM, UNCERTAINTY_SYSTEM


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Evaluate model responses against merged P/H/C sets with multi-metric LLM judging + semantic matching."
    )
    ap.add_argument("--input_path", type=str, required=True, help="JSON array with fields: raw_input, and target response column.")
    ap.add_argument("--pxhx_path", type=str, required=True, help="Merged ground-truth JSON array (each item has ground_truth_space_majority).")
    ap.add_argument("--output_path", type=str, required=True, help="Where to save metrics JSON.")
    ap.add_argument("--col_name", type=str, required=True, help="Column name for target response (e.g., 'original_output' or 'model_response').")

    ap.add_argument("--dx_extract_model", type=str, default="gpt-4.1-mini")
    ap.add_argument("--uncertainty_model", type=str, default="gpt-4.1-mini")
    ap.add_argument("--grounding_model", type=str, default="gpt-4.1-mini")
    ap.add_argument("--sem_match_model", type=str, default="gpt-4.1-mini")
    ap.add_argument("--sem_cache_path", type=str, default="", help="Optional path to save/load semantic match cache JSON.")

    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--resume_path", type=str, default="")

    ap.add_argument("--sem_max_pairs_per_call", type=int, default=50)
    ap.add_argument("--max_grounding_dx", type=int, default=8, help="Max extracted diagnoses to send to grounding judge per sample.")
    ap.add_argument("--top_k_dx", type=int, default=8, help="Number of top diagnoses to use for coverage/plausibility (0 = use all).")
    ap.add_argument("--skip_grounding", action="store_true", help="Skip grounding judge to speed up (metrics will be defaulted).")
    ap.add_argument("--skip_uncertainty", action="store_true", help="Skip uncertainty judge to speed up (flag will be false).")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    data = load_json_array(args.input_path)
    truth = load_json_array(args.pxhx_path)

    matcher = SemanticMatcher(
        model=args.sem_match_model,
        temperature=args.temperature,
        cache_path=(args.sem_cache_path or None),
    )

    print(f"Loaded {len(data)} samples from input_path.")
    print(f"Loaded {len(truth)} samples from merged truth file.")

    n_limit = min(len(data), len(truth))
    if len(truth) != len(data):
        print(f"[warn] truth has {len(truth)} records but input has {len(data)}. Will score first {n_limit} by index.")

    resume_path = args.resume_path or args.output_path
    existing = load_existing_output(resume_path) if resume_path else None

    per_sample: List[Dict[str, Any]] = []
    done_idx = set()

    if existing is not None:
        per_sample = existing.get("per_sample", [])
        done_idx = build_done_index_set(per_sample)
        print(f"[resume] Loaded {len(per_sample)} previous per_sample entries from {resume_path}.")
        print(f"[resume] Will skip {len(done_idx)} indices already completed.")

    agg = recompute_aggregates_from_per_sample(per_sample)
    agg.setdefault("sum_h_precision", 0.0)

    models_dict = {
        "dx_extract_model": args.dx_extract_model,
        "uncertainty_model": args.uncertainty_model,
        "grounding_model": args.grounding_model,
        "sem_match_model": args.sem_match_model,
    }

    for idx in tqdm(range(n_limit), desc="Scoring"):
        if idx in done_idx:
            continue
        if args.max and idx >= args.max:
            break

        sample = data[idx]
        truth_rec = truth[idx]

        question = (sample.get("raw_input") or sample.get("input") or "").strip()
        out_col = args.col_name
        model_answer = (sample.get(out_col) or "").strip()

        if not question or not model_answer:
            continue

        agg["n_total"] += 1

        P, H, C = extract_P_H_C_from_truth_record(truth_rec)
        if not isinstance(P, list) or not isinstance(H, list):
            per_sample.append(
                {
                    "index": idx,
                    "has_pxhx": False,
                    "input": question,
                    "model_response": model_answer,
                    "metrics": None,
                }
            )
            continue

        if C is None:
            C = []

        agg["n_with_pxhx"] += 1

        dx_user = (
            f"QUESTION:\n{question}\n\n"
            f"MODEL_ANSWER:\n{model_answer}\n\n"
            f"TOP_K:\n{int(args.top_k_dx)}\n\n"
            "Return STRICT JSON."
        )
        dx_obj = call_json_judge(
            model=args.dx_extract_model,
            system_prompt=DX_EXTRACT_SYSTEM,
            user_prompt=dx_user,
            temperature=args.temperature,
        )
        extracted_all = dx_obj.get("extracted_diagnoses", [])
        if not isinstance(extracted_all, list):
            extracted_all = []
        extracted_all = dedup_preserve_order([str(x) for x in extracted_all if str(x).strip()])

        extracted_top_k = dx_obj.get("top_k_diagnoses", [])
        if not isinstance(extracted_top_k, list):
            extracted_top_k = []
        extracted_top_k = dedup_preserve_order([str(x) for x in extracted_top_k if str(x).strip()])

        if args.top_k_dx and int(args.top_k_dx) > 0:
            k = int(args.top_k_dx)
            if not extracted_top_k:
                extracted_top_k = extracted_all[:k]
            else:
                extracted_top_k = extracted_top_k[:k]
        else:
            extracted_top_k = extracted_all[:]

        pairs_DP = [(dx, p) for dx in extracted_top_k for p in P]

        pair_decisions = matcher.batch_match_pairs(
            pairs_DP,
            max_pairs_per_call=args.sem_max_pairs_per_call,
        )

        def _is_match(a: str, b: str) -> Tuple[bool, Dict[str, Any]]:
            k = matcher._key(a, b)
            info = pair_decisions.get(k) or matcher.cache.get(k)
            if not info:
                return False, {"match": False, "relation": "different", "note": "missing"}
            return bool(info.get("match", False)), info

        inP = []
        outP = []
        matched_P_by_dx: Dict[str, Optional[str]] = {}
        for dx in extracted_top_k:
            matched_P = None
            matched_info = None
            for p in P:
                ok, info = _is_match(dx, p)
                if ok:
                    matched_P = p
                    matched_info = info
                    break
            if matched_P is None:
                outP.append(dx)
            else:
                inP.append({"dx": dx, "matched_P": matched_P, "match_info": matched_info})
            matched_P_by_dx[dx] = matched_P

        plausibility = 1.0 if len(extracted_top_k) == 0 else (len(inP) / len(extracted_top_k))

        covered_H = []
        uncovered_H = []
        for h in H:
            matched_dx = None
            matched_info = None
            for dx in extracted_top_k:
                if matched_P_by_dx.get(dx) == h:
                    matched_dx = dx
                    matched_info = {"match": True, "relation": "same", "note": "via_P"}
                    break
            if matched_dx is None:
                uncovered_H.append(h)
            else:
                covered_H.append({"h": h, "matched_dx": matched_dx, "match_info": matched_info})

        h_coverage = None if len(H) == 0 else (len(covered_H) / len(H))

        covered_C = []
        uncovered_C = []
        for c in C:
            matched_dx = None
            matched_info = None
            for dx in extracted_top_k:
                if matched_P_by_dx.get(dx) == c:
                    matched_dx = dx
                    matched_info = {"match": True, "relation": "same", "note": "via_P"}
                    break
            if matched_dx is None:
                uncovered_C.append(c)
            else:
                covered_C.append({"c": c, "matched_dx": matched_dx, "match_info": matched_info})

        c_coverage = None if len(C) == 0 else (len(covered_C) / len(C))

        in_H = []
        out_of_H = []
        H_set = set(H)
        for dx in extracted_top_k:
            matched_H = matched_P_by_dx.get(dx)
            if matched_H in H_set:
                in_H.append({"dx": dx, "matched_H": matched_H, "match_info": {"match": True, "relation": "same", "note": "via_P"}})
            else:
                out_of_H.append(dx)

        h_precision = 1.0 if len(extracted_top_k) == 0 else (len(in_H) / len(extracted_top_k))

        if args.skip_uncertainty:
            uncertainty_flag = False
        else:
            unc_user = f"QUESTION:\n{question}\n\nMODEL_ANSWER:\n{model_answer}\n\nReturn STRICT JSON."
            unc_obj = call_json_judge(
                model=args.uncertainty_model,
                system_prompt=UNCERTAINTY_SYSTEM,
                user_prompt=unc_user,
                temperature=args.temperature,
            )
            uncertainty_flag = bool(unc_obj.get("uncertainty_flag", False))

        breadth_metrics = compute_breadth_metrics(extracted_all, P)

        extracted_for_grounding = extracted_top_k[: max(0, int(args.max_grounding_dx))]

        if args.skip_grounding or len(extracted_for_grounding) == 0:
            grounding_metrics = {
                "per_diagnosis": [],
                "support_rate": 1.0 if len(extracted_for_grounding) == 0 else 0.0,
                "indirect_inference_rate": 0.0 if len(extracted_for_grounding) == 0 else 0.0,
            }
        else:
            grounding_user = (
                f"QUESTION:\n{question}\n\n"
                f"MODEL_ANSWER:\n{model_answer}\n\n"
                f"EXTRACTED_DIAGNOSES:\n{json.dumps(extracted_for_grounding, ensure_ascii=False)}\n\n"
                "Return STRICT JSON."
            )

            try:
                grounding_obj = call_json_judge(
                    model=args.grounding_model,
                    system_prompt=GROUNDING_SYSTEM,
                    user_prompt=grounding_user,
                    temperature=args.temperature,
                )
                grounding_metrics = compute_grounding_metrics(grounding_obj, extracted_for_grounding)
            except Exception:
                grounding_metrics = {
                    "per_diagnosis": [],
                    "support_rate": 1.0 if len(extracted_for_grounding) == 0 else 0.0,
                    "indirect_inference_rate": 0.0 if len(extracted_for_grounding) == 0 else 0.0,
                }

        agg["sum_plausibility"] += float(plausibility)

        if h_coverage is not None:
            agg["sum_hcov"] += float(h_coverage)
            agg["n_hcov_defined"] += 1

        if c_coverage is not None:
            agg["sum_ccov"] += float(c_coverage)
            agg["n_ccov_defined"] += 1

        agg["sum_h_precision"] += float(h_precision)
        agg["sum_support_rate"] += float(grounding_metrics["support_rate"])
        agg["sum_indirect_inference_rate"] += float(grounding_metrics["indirect_inference_rate"])

        agg["sum_breadth"] += float(breadth_metrics["breadth"])
        if breadth_metrics["normalized_breadth"] is not None:
            agg["sum_norm_breadth"] += float(breadth_metrics["normalized_breadth"])
            agg["n_norm_breadth_defined"] += 1

        if uncertainty_flag:
            agg["n_uncertain"] += 1

        ref_dx = sample.get("reference_diagnosis", None)

        per_sample.append(
            {
                "index": idx,
                "has_pxhx": True,
                "input": question,
                "model_response": model_answer,
                "reference_diagnosis": ref_dx,
                "judge_dx_space": {
                    "plausible_set": P,
                    "highly_likely_set": H,
                    "cannot_miss_set": C,
                },
                "metrics": {
                    "plausibility": plausibility,
                    "h_coverage": h_coverage,
                    "covered_H": covered_H,
                    "uncovered_H": uncovered_H,
                    "c_coverage": c_coverage,
                    "covered_C": covered_C,
                    "uncovered_C": uncovered_C,
                    "h_precision": h_precision,
                    "extracted_diagnoses": extracted_all,
                    "extracted_diagnoses_top_k": extracted_top_k,
                    "in_P": inP,
                    "out_of_P": outP,
                    "in_H": in_H,
                    "out_of_H": out_of_H,
                    "uncertainty_flag": uncertainty_flag,
                    "breadth": breadth_metrics["breadth"],
                    "normalized_breadth": breadth_metrics["normalized_breadth"],
                    "support_rate": grounding_metrics["support_rate"],
                    "indirect_inference_rate": grounding_metrics["indirect_inference_rate"],
                    "grounding_per_diagnosis": grounding_metrics["per_diagnosis"],
                },
            }
        )

        done_idx.add(idx)

        if (idx + 1) % args.save_every == 0:
            if args.sem_cache_path:
                matcher.save_cache()
            checkpoint_save(resume_path, per_sample, agg, matcher, models_dict)

    if args.sem_cache_path:
        matcher.save_cache()

    checkpoint_save(args.output_path, per_sample, agg, matcher, models_dict)

    print("=== SUMMARY ===")
    print(json.dumps(make_summary(agg, len(matcher.cache), models_dict), indent=2, ensure_ascii=False))
    print(f"\nSaved to {args.output_path}")


if __name__ == "__main__":
    main()
