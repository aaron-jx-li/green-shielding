"""Aggregate collected judge Batch stages into open_eval-compatible metrics JSON."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from open_eval.eval.metrics import (
    compute_breadth_metrics,
    compute_grounding_metrics,
    dedup_preserve_order,
    extract_P_H_C_from_truth_record,
    make_summary,
)

from management_reasoning.eval.batch.paths import (
    DEFAULT_COHORT,
    DEFAULT_TRUTH,
    answer_mode_for,
    local_collected,
    local_eval_out,
    local_pair_index,
    question_filter_for,
    response_path_for,
)
from management_reasoning.eval.prepare_inputs import build_eval_inputs
from management_reasoning.eval.text import fuzzy_match, normalize_text


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_json_array(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(path)
    return data


def _sem_key(a: str, b: str) -> str:
    a_n, b_n = normalize_text(a), normalize_text(b)
    return f"{a_n}|||{b_n}" if a_n <= b_n else f"{b_n}|||{a_n}"


def _build_match_table(
    suite: str,
    target: str,
    arm: str,
) -> Dict[int, Dict[str, bool]]:
    """sample_id → {pair_key → match}."""
    index = _load_jsonl(local_pair_index(suite, target, arm))
    sem_rows = {
        r["custom_id"]: r
        for r in _load_jsonl(local_collected(suite, target, arm, "sem"))
        if r.get("custom_id")
    }
    by_sample: Dict[int, Dict[str, bool]] = {}

    for meta in index:
        sid = int(meta["sample_id"])
        by_sample.setdefault(sid, {})
        if meta.get("chunk") == -1:
            for am in meta.get("auto_matches") or []:
                k = _sem_key(am["dx_a"], am["dx_b"])
                by_sample[sid][k] = True
            continue
        cid = meta["custom_id"]
        pairs = meta.get("pairs") or []
        matches = (sem_rows.get(cid) or {}).get("matches") or []
        if len(matches) < len(pairs):
            matches = list(matches) + [False] * (len(pairs) - len(matches))
        for pair, m in zip(pairs, matches):
            k = _sem_key(pair["dx_a"], pair["dx_b"])
            by_sample[sid][k] = bool(m)
    return by_sample


def aggregate_arm(
    *,
    suite: str,
    target: str,
    arm: str,
    cohort_json: str = DEFAULT_COHORT,
    truth_path: str = DEFAULT_TRUTH,
    top_k_dx: int = 8,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
) -> Dict[str, Any]:
    truth = _load_json_array(truth_path)
    responses = response_path_for(suite, target, arm)
    data = build_eval_inputs(
        responses_jsonl=responses,
        cohort_json=cohort_json,
        arm=arm,
        require_parse_ok=True,
        answer_mode=answer_mode_for(suite),
        question_filter=question_filter_for(suite),
    )
    if end_idx is None:
        end_idx = len(data) - 1
    data = data[start_idx : end_idx + 1]

    extracts = {
        r["sample_id"]: r
        for r in _load_jsonl(local_collected(suite, target, arm, "extract"))
        if r.get("parse_ok")
    }
    uncs = {
        r["sample_id"]: r
        for r in _load_jsonl(local_collected(suite, target, arm, "unc"))
        if r.get("parse_ok")
    }
    grounds = {
        r["sample_id"]: r
        for r in _load_jsonl(local_collected(suite, target, arm, "ground"))
        if r.get("parse_ok")
    }
    match_table = _build_match_table(suite, target, arm)

    per_sample: List[Dict[str, Any]] = []
    agg = {
        "n_total": 0,
        "n_with_pxhx": 0,
        "sum_plausibility": 0.0,
        "sum_hcov": 0.0,
        "n_hcov_defined": 0,
        "sum_ccov": 0.0,
        "n_ccov_defined": 0,
        "sum_h_precision": 0.0,
        "sum_support_rate": 0.0,
        "sum_indirect_inference_rate": 0.0,
        "sum_breadth": 0.0,
        "sum_norm_breadth": 0.0,
        "n_norm_breadth_defined": 0,
        "n_uncertain": 0,
    }

    for sample in data:
        idx = int(sample["index"])
        sid = int(sample["sample_id"])
        question = (sample.get("raw_input") or "").strip()
        model_answer = (sample.get("model_response") or "").strip()
        agg["n_total"] += 1

        if not question or not model_answer or sid not in extracts:
            per_sample.append(
                {
                    "index": idx,
                    "sample_id": sid,
                    "has_pxhx": False,
                    "input": question,
                    "model_response": model_answer,
                    "metrics": None,
                    "skip_reason": "missing_question_diagnosis_or_extract",
                }
            )
            continue

        truth_rec = truth[sid] if sid < len(truth) else {}
        P, H, C = extract_P_H_C_from_truth_record(truth_rec)
        if not isinstance(P, list) or not isinstance(H, list):
            per_sample.append(
                {
                    "index": idx,
                    "sample_id": sid,
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
        ex = extracts[sid]
        extracted_all = dedup_preserve_order(
            [str(x) for x in (ex.get("extracted_diagnoses") or []) if str(x).strip()]
        )
        extracted_top_k = dedup_preserve_order(
            [str(x) for x in (ex.get("top_k_diagnoses") or []) if str(x).strip()]
        )
        if top_k_dx > 0:
            if not extracted_top_k:
                extracted_top_k = extracted_all[:top_k_dx]
            else:
                extracted_top_k = extracted_top_k[:top_k_dx]
        else:
            extracted_top_k = extracted_all[:]

        mt = match_table.get(sid, {})

        def is_match(a: str, b: str) -> bool:
            if fuzzy_match(a, b):
                return True
            return bool(mt.get(_sem_key(a, b), False))

        inP = []
        outP = []
        matched_P_by_dx: Dict[str, Optional[str]] = {}
        for dx in extracted_top_k:
            matched_P = None
            for p in P:
                if is_match(dx, str(p)):
                    matched_P = str(p)
                    break
            if matched_P is None:
                outP.append(dx)
            else:
                inP.append({"dx": dx, "matched_P": matched_P})
            matched_P_by_dx[dx] = matched_P

        plausibility = 1.0 if not extracted_top_k else len(inP) / len(extracted_top_k)

        covered_H, uncovered_H = [], []
        for h in H:
            hit = None
            for dx in extracted_top_k:
                if matched_P_by_dx.get(dx) == h:
                    hit = dx
                    break
            if hit is None:
                uncovered_H.append(h)
            else:
                covered_H.append({"h": h, "matched_dx": hit})
        h_coverage = None if not H else len(covered_H) / len(H)

        covered_C, uncovered_C = [], []
        for c in C:
            hit = None
            for dx in extracted_top_k:
                if matched_P_by_dx.get(dx) == c:
                    hit = dx
                    break
            if hit is None:
                uncovered_C.append(c)
            else:
                covered_C.append({"c": c, "matched_dx": hit})
        c_coverage = None if not C else len(covered_C) / len(C)

        H_set = set(H)
        in_H, out_of_H = [], []
        for dx in extracted_top_k:
            mh = matched_P_by_dx.get(dx)
            if mh in H_set:
                in_H.append({"dx": dx, "matched_H": mh})
            else:
                out_of_H.append(dx)
        h_precision = 1.0 if not extracted_top_k else len(in_H) / len(extracted_top_k)

        uncertainty_flag = bool((uncs.get(sid) or {}).get("uncertainty_flag", False))
        breadth_metrics = compute_breadth_metrics(extracted_all, P)

        g_obj = grounds.get(sid) or {}
        if g_obj.get("per_diagnosis") is not None:
            grounding_metrics = compute_grounding_metrics(
                {"per_diagnosis": g_obj.get("per_diagnosis") or []},
                extracted_top_k,
            )
        else:
            grounding_metrics = {
                "per_diagnosis": [],
                "support_rate": 1.0 if not extracted_top_k else 0.0,
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
                "sample_id": sid,
                "arm": arm,
                "target": target,
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

    models = {
        "judge_model": "gemini-3.1-flash-lite",
        "judge_provider": "vertex_gemini_batch",
        "target": target,
        "arm": arm,
        "suite": suite,
        "track": "management_reasoning.eval.batch",
    }
    summary = make_summary(agg, 0, models)
    out = {"summary": summary, "per_sample": per_sample}
    out_path = local_eval_out(suite, target, arm)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return {"out_path": out_path, "summary": summary}
