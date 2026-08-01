#!/usr/bin/env python3
"""
Score management_reasoning diagnosis text against P/H/C truth (Gemini Flash-Lite).

Same metric definitions as ``open_eval/cli/evaluate.py`` (plausibility, H/S coverage,
breadth, evidence/support, inference, uncertainty). Does not modify open_eval.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from open_eval.eval.metrics import (  # noqa: E402  — read-only reuse of pure aggregators
    checkpoint_save,
    compute_breadth_metrics,
    compute_grounding_metrics,
    dedup_preserve_order,
    extract_P_H_C_from_truth_record,
    make_summary,
    recompute_aggregates_from_per_sample,
)

from management_reasoning.clients.vertex import (  # noqa: E402
    VertexConfigError,
    get_location,
    get_project,
    make_vertex_client,
)
from management_reasoning.config import models_config  # noqa: E402
from management_reasoning.eval.judge import call_json_judge  # noqa: E402
from management_reasoning.eval.prepare_inputs import build_eval_inputs  # noqa: E402
from management_reasoning.eval.prompts import (  # noqa: E402
    DX_EXTRACT_SYSTEM,
    GROUNDING_SYSTEM,
    UNCERTAINTY_SYSTEM,
)
from management_reasoning.eval.semantic_match import SemanticMatcher  # noqa: E402


def _load_json_array(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected JSON array: {path}")
    return data


def _load_existing(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _done_indices(per_sample: List[Dict[str, Any]]) -> set:
    done = set()
    for r in per_sample:
        if isinstance(r, dict) and isinstance(r.get("index"), int):
            done.add(r["index"])
    return done


def score_samples(
    *,
    data: List[Dict[str, Any]],
    truth: List[Dict[str, Any]],
    model: str,
    client: Any,
    project: str,
    location: str,
    thinking_level: str,
    temperature: float,
    top_k_dx: int,
    max_grounding_dx: int,
    sem_max_pairs_per_call: int,
    sem_cache_path: str,
    skip_grounding: bool,
    skip_uncertainty: bool,
    max_n: int,
    save_every: int,
    output_path: str,
    resume_path: str,
) -> Dict[str, Any]:
    matcher = SemanticMatcher(
        model=model,
        temperature=temperature,
        cache_path=(sem_cache_path or None),
        client=client,
        project=project,
        location=location,
        thinking_level=thinking_level,
    )

    n_limit = min(len(data), len(truth))
    if len(truth) != len(data):
        print(
            f"[warn] truth has {len(truth)} records but input has {len(data)}. "
            f"Will score first {n_limit} by index."
        )

    resume = resume_path or output_path
    existing = _load_existing(resume)
    per_sample: List[Dict[str, Any]] = []
    done_idx = set()
    if existing is not None:
        per_sample = existing.get("per_sample", []) or []
        done_idx = _done_indices(per_sample)
        print(f"[resume] {len(done_idx)} indices already done from {resume}")

    agg = recompute_aggregates_from_per_sample(per_sample)
    agg.setdefault("sum_h_precision", 0.0)

    models_dict = {
        "judge_model": model,
        "judge_provider": "vertex_gemini",
        "thinking_level": thinking_level,
        "track": "management_reasoning.eval",
    }

    end = n_limit
    if max_n and int(max_n) > 0:
        end = min(n_limit, int(max_n))

    for idx in tqdm(range(end), desc="mr-eval"):
        if idx in done_idx:
            continue
        # end already encodes --max; keep guard for clarity
        if max_n and idx >= max_n:
            break

        sample = data[idx]
        truth_rec = truth[idx]

        question = (sample.get("raw_input") or sample.get("input") or "").strip()
        model_answer = (sample.get("model_response") or "").strip()

        if not question or not model_answer:
            per_sample.append(
                {
                    "index": idx,
                    "sample_id": sample.get("sample_id", idx),
                    "has_pxhx": False,
                    "input": question,
                    "model_response": model_answer,
                    "metrics": None,
                    "skip_reason": "missing_question_or_diagnosis",
                }
            )
            agg["n_total"] += 1
            continue

        agg["n_total"] += 1

        P, H, C = extract_P_H_C_from_truth_record(truth_rec)
        if not isinstance(P, list) or not isinstance(H, list):
            per_sample.append(
                {
                    "index": idx,
                    "sample_id": sample.get("sample_id", idx),
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
            f"TOP_K:\n{int(top_k_dx)}\n\n"
            "Return STRICT JSON."
        )
        dx_obj = call_json_judge(
            model=model,
            system_prompt=DX_EXTRACT_SYSTEM,
            user_prompt=dx_user,
            temperature=temperature,
            client=client,
            project=project,
            location=location,
            thinking_level=thinking_level,
        )
        extracted_all = dx_obj.get("extracted_diagnoses", [])
        if not isinstance(extracted_all, list):
            extracted_all = []
        extracted_all = dedup_preserve_order([str(x) for x in extracted_all if str(x).strip()])

        extracted_top_k = dx_obj.get("top_k_diagnoses", [])
        if not isinstance(extracted_top_k, list):
            extracted_top_k = []
        extracted_top_k = dedup_preserve_order(
            [str(x) for x in extracted_top_k if str(x).strip()]
        )

        if top_k_dx and int(top_k_dx) > 0:
            k = int(top_k_dx)
            if not extracted_top_k:
                extracted_top_k = extracted_all[:k]
            else:
                extracted_top_k = extracted_top_k[:k]
        else:
            extracted_top_k = extracted_all[:]

        pairs_DP = [(dx, p) for dx in extracted_top_k for p in P]
        pair_decisions = matcher.batch_match_pairs(
            pairs_DP, max_pairs_per_call=sem_max_pairs_per_call
        )

        def _is_match(a: str, b: str) -> Tuple[bool, Dict[str, Any]]:
            key = matcher._key(a, b)
            info = pair_decisions.get(key) or matcher.cache.get(key)
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
                covered_H.append(
                    {"h": h, "matched_dx": matched_dx, "match_info": matched_info}
                )
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
                covered_C.append(
                    {"c": c, "matched_dx": matched_dx, "match_info": matched_info}
                )
        c_coverage = None if len(C) == 0 else (len(covered_C) / len(C))

        in_H = []
        out_of_H = []
        H_set = set(H)
        for dx in extracted_top_k:
            matched_H = matched_P_by_dx.get(dx)
            if matched_H in H_set:
                in_H.append(
                    {
                        "dx": dx,
                        "matched_H": matched_H,
                        "match_info": {"match": True, "relation": "same", "note": "via_P"},
                    }
                )
            else:
                out_of_H.append(dx)
        h_precision = 1.0 if len(extracted_top_k) == 0 else (len(in_H) / len(extracted_top_k))

        if skip_uncertainty:
            uncertainty_flag = False
        else:
            unc_user = (
                f"QUESTION:\n{question}\n\nMODEL_ANSWER:\n{model_answer}\n\n"
                "Return STRICT JSON."
            )
            unc_obj = call_json_judge(
                model=model,
                system_prompt=UNCERTAINTY_SYSTEM,
                user_prompt=unc_user,
                temperature=temperature,
                client=client,
                project=project,
                location=location,
                thinking_level=thinking_level,
            )
            uncertainty_flag = bool(unc_obj.get("uncertainty_flag", False))

        breadth_metrics = compute_breadth_metrics(extracted_all, P)
        extracted_for_grounding = extracted_top_k[: max(0, int(max_grounding_dx))]

        if skip_grounding or len(extracted_for_grounding) == 0:
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
                    model=model,
                    system_prompt=GROUNDING_SYSTEM,
                    user_prompt=grounding_user,
                    temperature=temperature,
                    client=client,
                    project=project,
                    location=location,
                    thinking_level=thinking_level,
                )
                grounding_metrics = compute_grounding_metrics(
                    grounding_obj, extracted_for_grounding
                )
            except Exception:
                grounding_metrics = {
                    "per_diagnosis": [],
                    "support_rate": 0.0,
                    "indirect_inference_rate": 0.0,
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
        agg["sum_indirect_inference_rate"] += float(
            grounding_metrics["indirect_inference_rate"]
        )
        agg["sum_breadth"] += float(breadth_metrics["breadth"])
        if breadth_metrics["normalized_breadth"] is not None:
            agg["sum_norm_breadth"] += float(breadth_metrics["normalized_breadth"])
            agg["n_norm_breadth_defined"] += 1
        if uncertainty_flag:
            agg["n_uncertain"] += 1

        per_sample.append(
            {
                "index": idx,
                "sample_id": sample.get("sample_id", idx),
                "arm": sample.get("arm"),
                "has_pxhx": True,
                "input": question,
                "model_response": model_answer,
                "reference_diagnosis": sample.get("reference_diagnosis"),
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

        if (idx + 1) % save_every == 0:
            if sem_cache_path:
                matcher.save_cache()
            checkpoint_save(resume, per_sample, agg, matcher, models_dict)

    if sem_cache_path:
        matcher.save_cache()
    checkpoint_save(output_path, per_sample, agg, matcher, models_dict)
    summary = make_summary(agg, len(matcher.cache), models_dict)
    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved to {output_path}")
    return summary


def main() -> None:
    cfg = models_config()
    default_judge = cfg.get("gemini_judge", "gemini-3.1-flash-lite")
    default_location = cfg.get("location_default", "global")
    default_thinking = cfg.get("gemini_judge_thinking_level", "HIGH")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--responses_jsonl",
        default="",
        help="management_reasoning responses.jsonl (preferred with --arm)",
    )
    ap.add_argument(
        "--input_path",
        default="",
        help="Prebuilt eval input JSON from prepare_inputs.py (alternative to --responses_jsonl)",
    )
    ap.add_argument(
        "--cohort_json",
        default="./results/management_reasoning/data/hcm_full_inputs.json",
    )
    ap.add_argument("--arm", choices=("raw", "neutralized"), default="raw")
    ap.add_argument(
        "--pxhx_path",
        default="./results/HCM-3k/truth/merged_truth_new.json",
    )
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--resume_path", default="")
    ap.add_argument("--model", default=default_judge)
    ap.add_argument("--project", default=None)
    ap.add_argument("--location", default=None)
    ap.add_argument("--thinking_level", default=default_thinking)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_k_dx", type=int, default=8)
    ap.add_argument("--max_grounding_dx", type=int, default=8)
    ap.add_argument("--sem_max_pairs_per_call", type=int, default=50)
    ap.add_argument("--sem_cache_path", default="")
    ap.add_argument("--save_every", type=int, default=25)
    ap.add_argument("--max", type=int, default=0, help="Score only first N indices (smoke)")
    ap.add_argument("--skip_grounding", action="store_true")
    ap.add_argument("--skip_uncertainty", action="store_true")
    args = ap.parse_args()

    try:
        project = get_project(args.project)
        location = get_location(
            args.location or os.environ.get("GOOGLE_CLOUD_LOCATION") or default_location
        )
    except VertexConfigError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)

    if args.responses_jsonl:
        data = build_eval_inputs(
            responses_jsonl=args.responses_jsonl,
            cohort_json=args.cohort_json,
            arm=args.arm,
            require_parse_ok=True,
        )
    elif args.input_path:
        data = _load_json_array(args.input_path)
    else:
        raise SystemExit("Provide --responses_jsonl or --input_path")

    truth = _load_json_array(args.pxhx_path)
    client = make_vertex_client(project=project, location=location)

    print(
        f"management_reasoning.eval: model={args.model} thinking={args.thinking_level} "
        f"project={project} location={location} arm={args.arm} "
        f"n_input={len(data)} n_truth={len(truth)} max={args.max or 'all'}"
    )

    score_samples(
        data=data,
        truth=truth,
        model=args.model,
        client=client,
        project=project,
        location=location,
        thinking_level=args.thinking_level,
        temperature=args.temperature,
        top_k_dx=args.top_k_dx,
        max_grounding_dx=args.max_grounding_dx,
        sem_max_pairs_per_call=args.sem_max_pairs_per_call,
        sem_cache_path=args.sem_cache_path,
        skip_grounding=args.skip_grounding,
        skip_uncertainty=args.skip_uncertainty,
        max_n=args.max,
        save_every=args.save_every,
        output_path=args.output_path,
        resume_path=args.resume_path or args.output_path,
    )


if __name__ == "__main__":
    main()
