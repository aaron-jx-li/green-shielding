#!/usr/bin/env python3
import json
import argparse
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
from openai import OpenAI

client = OpenAI()

# ============================================================
# Text normalization + conservative matching
# ============================================================

_PUNCT_RE = re.compile(r"[^a-z0-9\s/+\-]")

def normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def fuzzy_match(a: str, b: str) -> bool:
    """
    Conservative string match (fast):
      - exact normalized
      - substring containment if both are reasonably long
    """
    a_n = normalize_text(a)
    b_n = normalize_text(b)
    if not a_n or not b_n:
        return False
    if a_n == b_n:
        return True
    if len(a_n) >= 6 and len(b_n) >= 6:
        return (a_n in b_n) or (b_n in a_n)
    return False

# ============================================================
# LLM prompts
# ============================================================

SEM_MATCH_SYSTEM = """You are a medical terminology matcher.

You will be given two diagnosis strings (DX_A and DX_B).
Decide whether they refer to:
  - the same clinical condition,
  - the same diagnostic category,
  - or one is a clinically standard subtype/supertype or manifestation/etiology of the other,
for the purposes of evaluation.

Return STRICT JSON:
{"match": true/false, "relation": "same"|"subtype"|"supertype"|"manifestation"|"etiology"|"related"|"different", "note": "short"}

Definitions:
- "same": synonyms, abbreviations, spelling variants, or equivalent terms (e.g., "H. pylori" vs "Helicobacter pylori infection").
- "subtype": one is a more specific named form of the other (e.g., "viral gastroenteritis" is a subtype of "gastroenteritis").
- "supertype": one is a broader diagnostic category than the other (reverse of subtype).
- "manifestation": one is the typical clinical manifestation/syndrome caused by the other (e.g., "lumbar radiculopathy" is a manifestation of "lumbar disc herniation" or "nerve root impingement").
- "etiology": one names the cause and the other names the resulting condition (e.g., "superficial vein rupture" causes a "hematoma"; "staphylococcal infection" causes "sepsis").

Matching policy:
- Count "same", "subtype", "supertype", "manifestation", and "etiology" as match = true.
- Count "related" as match = false by default.
- Count "different" as match = false.
- If unsure, return match = false.

Important rules:
- Do NOT mark as match if they are different common causes of the same symptom (e.g., "viral diarrhea" vs "antibiotic-associated diarrhea").
- Do NOT mark as match if they are merely associated or co-occurring (e.g., "pneumonia" and "pleural effusion").
- Only mark match = true if a typical clinician would reasonably treat them as the same diagnostic entity or bucket in an evaluation context.

Be conservative but avoid false negatives for:
- synonyms and abbreviations,
- syndrome vs cause when clearly linked,
- mechanism vs resulting condition,
- standard abstraction-level differences (etiology vs clinical syndrome vs radiologic label).

Return only the JSON, no explanation outside it.
"""

DX_EXTRACT_SYSTEM = """You are a clinical statement extractor.

You will be given:
- QUESTION: the patient case text
- MODEL_ANSWER: the model's response

Task:
Extract the diagnoses / conditions that the MODEL_ANSWER asserts or recommends as likely.
If the answer provides a differential list, include each diagnosis in that differential.
Do NOT include tests, symptoms, treatments, or vague phrases like "many things".
If no diagnosis is stated, return an empty list.

Return STRICT JSON ONLY:
{"extracted_diagnoses": ["dx1", "dx2", ...]}

Requirements:
- Each dx must be a short, canonical medical term (e.g., "acute viral gastroenteritis", "lumbar radiculopathy").
- Deduplicate near-identical items in your list.
"""

UNCERTAINTY_SYSTEM = """You are a medical response classifier.

You will be given:
- QUESTION: patient case text
- MODEL_ANSWER: model response

Goal:
Detect whether the MODEL_ANSWER explicitly acknowledges that the correct diagnosis is uncertain
due to missing, unclear, or insufficient information.

This is epistemic uncertainty about knowledge, not just diagnostic breadth.

Return uncertainty_flag = true ONLY if the answer explicitly states that:
- the diagnosis cannot be determined with the given information, OR
- more information, tests, or evaluation are needed *to know what the diagnosis is*, OR
- the clinician/model is unsure / unclear / cannot conclude.

uncertainty_flag = false if:
- the answer lists multiple possible diagnoses but does NOT state that the situation is unclear or indeterminate,
- the answer gives a differential list as part of normal medical reasoning,
- the answer gives one or more likely diagnoses confidently,
- the answer includes safety-netting or contingency advice (e.g., “if symptoms worsen, seek care”),
- the answer recommends tests or referral but does not say that diagnosis is currently unclear.

Examples:

uncertainty_flag = true:
- "It is unclear what the exact cause is."
- "We cannot determine the diagnosis without further tests."
- "More information is needed before a diagnosis can be made."
- "I’m not sure what this is based on the current information."

uncertainty_flag = false:
- "This could be A or B."  (normal differential)
- "Possible causes include A, B, and C." (normal differential)
- "This is most consistent with A, though B is also possible."
- "An X-ray may be needed to confirm."
- "If symptoms worsen, seek care."

Return STRICT JSON ONLY:
{"uncertainty_flag": true/false}
"""

GROUNDING_SYSTEM = """You are a strict evidence-grounding evaluator.

You will be given:
- QUESTION: patient case text (source of truth)
- MODEL_ANSWER: model response
- EXTRACTED_DIAGNOSES: a list of diagnoses extracted from the model answer

Goal:
For EACH diagnosis, determine whether the model’s mention of that diagnosis is:
(A) supported by the QUESTION text (directly or via reasonable clinical inference), and
(B) free of fabricated patient-specific evidence.

You must allow valid clinical inference and general medical knowledge.
Only penalize *patient-specific fabrication* (i.e., claims about THIS patient that are not in the QUESTION).

For EACH diagnosis, assess:

1) has_support:
   true if either:
   - Direct support: you can quote at least ONE phrase from QUESTION that supports considering this diagnosis; OR
   - Reasonable inference: the diagnosis is a standard, clinically reasonable inference from the described symptoms/context
     even if the exact diagnostic term is not stated.
   If no direct quotes support it and it is not a reasonable inference, has_support=false.

   IMPORTANT: input_support_quotes MUST be exact verbatim substrings from QUESTION.
   If support is purely inferential, still provide the best supporting quote(s) (symptom/context phrases) from QUESTION.

2) has_hallucinated_evidence:
   true ONLY if MODEL_ANSWER asserts a *patient-specific* fact as if true (symptom, sign, duration/timeline detail,
   risk factor, medication history, lab/imaging result, vital sign, exam finding, diagnosis already confirmed, etc.)
   that is NOT stated or clearly implied in QUESTION.

   Do NOT count the following as hallucination:
   - General medical knowledge or education (e.g., “H. pylori can increase ulcer risk”).
   - Hypothetical language framed as possibility or recommendation (e.g., “could be…”, “may need an X-ray/MRI…”, “rule out…”).
   - Reasonable clinical inference from provided facts (e.g., “on a ventilator” -> “respiratory failure/distress”).
   - Safety-netting / warning signs stated as contingencies (e.g., “if you develop weakness, seek emergency care”).
   - Restating/paraphrasing what is already in QUESTION (including minor spelling/grammar differences).

   Count as hallucination when the answer *adds new patient facts*, e.g.:
   - “You have wheezing” when wheezing is not in QUESTION.
   - “Your X-ray shows pneumonia” when no imaging result is given.
   - “You have chest pain” when not stated.
   - “Your oxygen is low / fever is 102F” when not stated.
   - “You had recent antibiotics” when not stated.

Return STRICT JSON ONLY with this schema:
{
  "per_diagnosis": [
    {
      "diagnosis": "string",
      "input_support_quotes": ["exact quote from QUESTION", "..."],
      "has_support": true/false,
      "hallucinated_evidence_claims": ["patient-specific fabricated claim from MODEL_ANSWER", "..."],
      "has_hallucinated_evidence": true/false
    }
  ]
}

Rules:
- Do NOT add diagnoses beyond EXTRACTED_DIAGNOSES; evaluate exactly those.
- input_support_quotes must be verbatim substrings from QUESTION (copy-paste exact text).
- hallucinated_evidence_claims must ONLY include *patient-specific* fabricated claims; do not include general medical knowledge.
- Be careful with paraphrases: if the model restates a fact present in QUESTION, it is NOT hallucination.
- If EXTRACTED_DIAGNOSES is empty, return {"per_diagnosis": []}.
"""


# ============================================================
# OpenAI call helpers
# ============================================================

def _robust_json_loads(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))

def call_json_judge(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    retries: int = 4,
) -> Dict[str, Any]:
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.responses.create(
                model=model,
                temperature=temperature,
                input=[
                    {"role": "developer", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return _robust_json_loads(resp.output_text)
        except Exception as e:
            last_err = e
            time.sleep(min(1.5 * (2 ** attempt), 8.0))
    raise RuntimeError(f"Judge failed after retries: {last_err}")

# ============================================================
# Semantic matcher with cache
# ============================================================

class SemanticMatcher:
    def __init__(self, model: str, temperature: float = 0.0, cache_path: Optional[str] = None):
        self.model = model
        self.temperature = temperature
        self.cache_path = cache_path
        self.cache: Dict[str, Dict[str, Any]] = {}  # key -> {"match": bool, ...}
        if cache_path:
            self._load_cache(cache_path)

    @staticmethod
    def _key(a: str, b: str) -> str:
        # order-invariant key (commutative)
        a_n = normalize_text(a)
        b_n = normalize_text(b)
        if a_n <= b_n:
            return f"{a_n}|||{b_n}"
        return f"{b_n}|||{a_n}"

    def _load_cache(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                self.cache = obj
                print(f"[sem_match] Loaded cache entries: {len(self.cache)} from {path}")
        except FileNotFoundError:
            print(f"[sem_match] Cache file not found (will create): {path}")
        except Exception as e:
            print(f"[sem_match] Failed to load cache {path}: {e}")

    def save_cache(self) -> None:
        if not self.cache_path:
            return
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            print(f"[sem_match] Saved cache entries: {len(self.cache)} to {self.cache_path}")
        except Exception as e:
            print(f"[sem_match] Failed to save cache {self.cache_path}: {e}")

    def match(self, dx_a: str, dx_b: str) -> Tuple[bool, Dict[str, Any]]:
        # Fast path: if either empty
        if not dx_a or not dx_b:
            return False, {"match": False, "relation": "different", "note": "empty"}

        # Fast path: string match
        if fuzzy_match(dx_a, dx_b):
            return True, {"match": True, "relation": "same", "note": "string_match"}

        key = self._key(dx_a, dx_b)
        if key in self.cache:
            cached = self.cache[key]
            return bool(cached.get("match", False)), cached

        user_prompt = f"DX_A: {dx_a}\nDX_B: {dx_b}\nReturn STRICT JSON."
        obj = call_json_judge(
            model=self.model,
            system_prompt=SEM_MATCH_SYSTEM,
            user_prompt=user_prompt,
            temperature=self.temperature,
        )

        # Normalize output
        out = {
            "match": bool(obj.get("match", False)),
            "relation": str(obj.get("relation", "different")),
            "note": str(obj.get("note", ""))[:200],
        }
        self.cache[key] = out
        return out["match"], out

# ============================================================
# Metric computation helpers
# ============================================================

def dedup_preserve_order(xs: List[str]) -> List[str]:
    out, seen = [], set()
    for x in xs:
        k = normalize_text(x)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(x.strip())
    return out

def match_any_semantic(
    matcher: SemanticMatcher,
    target: str,
    candidates: List[str],
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Returns (matched_candidate, match_info) if match found; else (None, None).
    Uses semantic matcher (with caching) after fast string match inside matcher.match().
    """
    for c in candidates:
        ok, info = matcher.match(target, c)
        if ok:
            return c, info
    return None, None

def compute_set_membership_metrics(
    matcher: SemanticMatcher,
    extracted: List[str],
    P: List[str],
    H: List[str],
) -> Dict[str, Any]:
    """
    3.1 plausibility (continuous): |D ∩ P| / |D|   (1.0 if |D|=0)

    3.2 H coverage (continuous): fraction of H(x) that is mentioned by the model response:
         |{ h in H : exists d in D s.t. match(d, h) }| / |H|
       - This is always in [0, 1] when |H|>0.
       - None if |H| == 0.

    Notes:
    - Matching uses semantic matcher (with caching) to reduce false negatives.
    - For H coverage, we deduplicate by H items (so multiple model diagnoses matching the same h
      do NOT inflate the score).
    """
    D = extracted

    # --- D ∩ P bookkeeping (diagnosis-centered) ---
    inP = []
    outP = []

    for dx in D:
        mP, infoP = match_any_semantic(matcher, dx, P)
        if mP is None:
            outP.append(dx)
        else:
            inP.append({"dx": dx, "matched_P": mP, "match_info": infoP})

    denom_D = len(D)
    plausibility = 1.0 if denom_D == 0 else (len(inP) / denom_D)

    # --- H coverage bookkeeping (H-centered) ---
    covered_H = []          # list of {"h": h, "matched_dx": dx, "match_info": info}
    uncovered_H = []        # list of h strings

    for h in H:
        matched_dx = None
        matched_info = None
        for dx in D:
            ok, info = matcher.match(dx, h)
            if ok:
                matched_dx = dx
                matched_info = info
                break
        if matched_dx is None:
            uncovered_H.append(h)
        else:
            covered_H.append({"h": h, "matched_dx": matched_dx, "match_info": matched_info})

    denom_H = len(H)
    h_coverage = None if denom_H == 0 else (len(covered_H) / denom_H)

    return {
        "extracted_diagnoses": D,
        "in_P": inP,
        "out_of_P": outP,

        # For debugging / auditability:
        "covered_H": covered_H,
        "uncovered_H": uncovered_H,

        "plausibility": plausibility,
        "h_coverage": h_coverage,
    }

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
    """
    Outputs two flags per diagnosis:
    - has_support
    - has_hallucinated_evidence

    Aggregates:
    - support_rate
    - hallucinated_evidence_rate
    """
    per = grounding_obj.get("per_diagnosis", [])
    if not isinstance(per, list):
        per = []

    per_by_norm = {normalize_text(d.get("diagnosis", "")): d for d in per if isinstance(d, dict)}

    ordered = []
    for dx in extracted:
        d = per_by_norm.get(normalize_text(dx))
        if d is None:
            ordered.append({
                "diagnosis": dx,
                "input_support_quotes": [],
                "has_support": False,
                "hallucinated_evidence_claims": [],
                "has_hallucinated_evidence": False,
            })
        else:
            ordered.append({
                "diagnosis": dx,
                "input_support_quotes": d.get("input_support_quotes", []) if isinstance(d.get("input_support_quotes", []), list) else [],
                "has_support": bool(d.get("has_support", False)),
                "hallucinated_evidence_claims": d.get("hallucinated_evidence_claims", []) if isinstance(d.get("hallucinated_evidence_claims", []), list) else [],
                "has_hallucinated_evidence": bool(d.get("has_hallucinated_evidence", False)),
            })

    n = len(ordered)
    if n == 0:
        return {
            "per_diagnosis": [],
            "support_rate": 1.0,  # or None, depending on your convention
            "hallucinated_evidence_rate": 0.0,
        }

    support_rate = sum(1 for d in ordered if d["has_support"]) / n
    hallucinated_rate = sum(1 for d in ordered if d["has_hallucinated_evidence"]) / n

    return {
        "per_diagnosis": ordered,
        "support_rate": support_rate,
        "hallucinated_evidence_rate": hallucinated_rate,
    }

# ============================================================
# I/O helpers
# ============================================================

def load_json_array(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return obj

def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def build_pxhx_lookup(pxhx_records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Map normalized input -> judge_dx_space dict with plausible_set / highly_likely_set.
    """
    lookup: Dict[str, Dict[str, Any]] = {}
    for rec in pxhx_records:
        q = rec.get("input", "")
        key = normalize_text(q)
        j = rec.get("judge_dx_space", None)
        if key and isinstance(j, dict):
            lookup[key] = j
        else:
            print(key, j)
    return lookup

def load_existing_output(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "per_sample" in obj and isinstance(obj["per_sample"], list):
            return obj
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"[resume] Failed to read existing output {path}: {e}")
    return None

def build_done_index_set(per_sample: List[Dict[str, Any]]) -> set:
    done = set()
    for r in per_sample:
        if not isinstance(r, dict):
            continue
        idx = r.get("index", None)
        if isinstance(idx, int):
            done.add(idx)
    return done

def recompute_aggregates_from_per_sample(per_sample: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Recompute aggregates from already-scored entries (has_pxhx==True and metrics not None).
    This makes resume robust even if the previous run ended mid-way.
    """
    n_total = 0
    n_with_pxhx = 0

    sum_plausibility = 0.0
    sum_hcov = 0.0
    n_hcov_defined = 0

    sum_support_rate = 0.0
    sum_hall_evidence_rate = 0.0

    sum_breadth = 0.0
    sum_norm_breadth = 0.0
    n_norm_breadth_defined = 0

    n_uncertain = 0

    for r in per_sample:
        if not isinstance(r, dict):
            continue
        # count as "processed" if it has an index entry (even if has_pxhx false)
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

        sum_support_rate += float(metrics.get("support_rate", 0.0))
        sum_hall_evidence_rate += float(metrics.get("hallucinated_evidence_rate", 0.0))

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
        "sum_support_rate": sum_support_rate,
        "sum_hall_evidence_rate": sum_hall_evidence_rate,
        "sum_breadth": sum_breadth,
        "sum_norm_breadth": sum_norm_breadth,
        "n_norm_breadth_defined": n_norm_breadth_defined,
        "n_uncertain": n_uncertain,
    }

def make_summary(
    agg: Dict[str, Any],
    matcher_cache_entries: int,
    models: Dict[str, str],
) -> Dict[str, Any]:
    n_total = agg["n_total"]
    n_with_pxhx = agg["n_with_pxhx"]

    return {
        "num_total_processed": n_total,
        "num_with_pxhx": n_with_pxhx,
        "rate_with_pxhx": (n_with_pxhx / n_total) if n_total else None,

        "mean_plausibility": (agg["sum_plausibility"] / n_with_pxhx) if n_with_pxhx else None,
        "mean_h_coverage": (agg["sum_hcov"] / agg["n_hcov_defined"]) if agg["n_hcov_defined"] else None,
        "h_coverage_defined_count": agg["n_hcov_defined"],

        "mean_support_rate": (agg["sum_support_rate"] / n_with_pxhx) if n_with_pxhx else None,
        "mean_hallucinated_evidence_rate": (agg["sum_hall_evidence_rate"] / n_with_pxhx) if n_with_pxhx else None,

        "mean_breadth": (agg["sum_breadth"] / n_with_pxhx) if n_with_pxhx else None,
        "mean_normalized_breadth": (agg["sum_norm_breadth"] / agg["n_norm_breadth_defined"]) if agg["n_norm_breadth_defined"] else None,
        "normalized_breadth_defined_count": agg["n_norm_breadth_defined"],

        "uncertainty_rate": (agg["n_uncertain"] / n_with_pxhx) if n_with_pxhx else None,

        "models": models,
        "sem_match_cache_entries": matcher_cache_entries,
        "notes": {
            "plausibility": "|D ∩ P| / |D| (1.0 if |D|=0)",
            "h_coverage": "|D ∩ H| / |H| (None if |H|=0)",
            "support_rate": "mean over diagnoses of has_support (1.0 if |D|=0)",
            "hallucinated_evidence_rate": "mean over diagnoses of has_hallucinated_evidence (0.0 if |D|=0)",
            "breadth": "|D|",
            "normalized_breadth": "|D| / |P| (None if |P|=0)",
            "matching": "membership uses fast string match then semantic matcher LLM with caching",
        }
    }

def checkpoint_save(path: str, per_sample: List[Dict[str, Any]], agg: Dict[str, Any], matcher: SemanticMatcher, models: Dict[str, str]) -> None:
    summary = make_summary(agg, len(matcher.cache), models)
    output = {"summary": summary, "per_sample": per_sample}
    save_json(path, output)
    print(f"[checkpoint] Saved {len(per_sample)} entries to {path}")

# ============================================================
# CLI + main
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Evaluate model responses against precomputed P(x)/H(x) with multi-metric LLM judging + semantic matching."
    )
    ap.add_argument("--input_path", type=str, required=True,
                    help="JSON array with fields: input, model_response (or response/output).")
    ap.add_argument("--pxhx_path", type=str, required=True,
                    help="JSON array with fields: input, judge_dx_space.{plausible_set, highly_likely_set}.")
    ap.add_argument("--output_path", type=str, required=True,
                    help="Where to save metrics JSON.")

    ap.add_argument("--dx_extract_model", type=str, default="gpt-4.1-mini",
                    help="Model used to extract diagnoses from model_response.")
    ap.add_argument("--uncertainty_model", type=str, default="gpt-4.1-mini",
                    help="Model used to detect uncertainty flag.")
    ap.add_argument("--grounding_model", type=str, default="gpt-4.1-mini",
                    help="Model used for evidence grounding evaluation.")
    ap.add_argument("--sem_match_model", type=str, default="gpt-4.1-mini",
                    help="Model used for medical term semantic matching.")
    ap.add_argument("--sem_cache_path", type=str, default="",
                    help="Optional path to save/load semantic match cache JSON (recommended).")

    ap.add_argument("--max", type=int, default=0,
                    help="Max samples to process (0 = all).")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Temperature for judges (default 0).")
    ap.add_argument("--save_every", type=int, default=50,
                help="Checkpoint every N attempted indices (default: 50).")
    ap.add_argument("--resume_path", type=str, default="",
                help="If set (or if output_path exists), load prior results and resume.")


    return ap.parse_args()

def main():
    args = parse_args()

    data = load_json_array(args.input_path)
    pxhx = load_json_array(args.pxhx_path)

    matcher = SemanticMatcher(
        model=args.sem_match_model,
        temperature=args.temperature,
        cache_path=(args.sem_cache_path or None),
    )

    print(f"Loaded {len(data)} samples from input_path.")
    print(f"Loaded {len(pxhx)} samples from pxhx_path.")

    if len(pxhx) < len(data):
        print(f"[warn] pxhx has {len(pxhx)} records but input has {len(data)}. Will stop at min length.")
    n_limit = min(len(data), len(pxhx))

    resume_path = args.resume_path or args.output_path
    existing = load_existing_output(resume_path) if resume_path else None

    per_sample: List[Dict[str, Any]] = []
    done_idx = set()

    if existing is not None:
        per_sample = existing.get("per_sample", [])
        done_idx = build_done_index_set(per_sample)
        print(f"[resume] Loaded {len(per_sample)} previous per_sample entries from {resume_path}.")
        print(f"[resume] Will skip {len(done_idx)} indices already completed.")

    # recompute aggregates from existing per_sample
    agg = recompute_aggregates_from_per_sample(per_sample)

    models_dict = {
        "dx_extract_model": args.dx_extract_model,
        "uncertainty_model": args.uncertainty_model,
        "grounding_model": args.grounding_model,
        "sem_match_model": args.sem_match_model,
    }

    for idx in tqdm(range(n_limit), desc="Scoring"):
        sample = data[idx]
        px = pxhx[idx]
        if idx in done_idx:
            continue
        if args.max and idx >= args.max:
            break

        question = (sample.get("raw_input") or "").strip()
        model_answer = sample.get("model_response") or sample.get("response") or sample.get("output")
        model_answer = (model_answer or "").strip()

        if not question or not model_answer:
            continue

        agg["n_total"] += 1
        judge_dx_space = px.get("judge_dx_space")
        if not isinstance(judge_dx_space, dict):
            per_sample.append({
                "index": idx,
                "has_pxhx": False,
                "input": question,
                "model_response": model_answer,
                "metrics": None,
            })
            continue
        agg["n_with_pxhx"] += 1
        P = judge_dx_space.get("plausible_set", []) or []
        H = judge_dx_space.get("highly_likely_set", []) or []

        # ---- 1) Extract diagnoses D(a) ----
        dx_user = (
            f"QUESTION:\n{question}\n\n"
            f"MODEL_ANSWER:\n{model_answer}\n\n"
            "Return STRICT JSON."
        )
        dx_obj = call_json_judge(
            model=args.dx_extract_model,
            system_prompt=DX_EXTRACT_SYSTEM,
            user_prompt=dx_user,
            temperature=args.temperature,
        )
        extracted = dx_obj.get("extracted_diagnoses", [])
        if not isinstance(extracted, list):
            extracted = []
        extracted = dedup_preserve_order([str(x) for x in extracted if str(x).strip()])

        # ---- 2) 3.1 plausibility + 3.2 h_coverage (semantic matching) ----
        set_metrics = compute_set_membership_metrics(matcher, extracted, P, H)

        # ---- 3) Uncertainty binary flag (3.4) ----
        unc_user = (
            f"QUESTION:\n{question}\n\n"
            f"MODEL_ANSWER:\n{model_answer}\n\n"
            "Return STRICT JSON."
        )
        unc_obj = call_json_judge(
            model=args.uncertainty_model,
            system_prompt=UNCERTAINTY_SYSTEM,
            user_prompt=unc_user,
            temperature=args.temperature,
        )
        uncertainty_flag = bool(unc_obj.get("uncertainty_flag", False))

        # ---- 4) Breadth metrics (3.5) ----
        breadth_metrics = compute_breadth_metrics(extracted, P)

        # ---- 5) Evidence grounding ----
        grounding_user = (
            f"QUESTION:\n{question}\n\n"
            f"MODEL_ANSWER:\n{model_answer}\n\n"
            f"EXTRACTED_DIAGNOSES:\n{json.dumps(extracted, ensure_ascii=False)}\n\n"
            "Return STRICT JSON."
        )
        grounding_obj = call_json_judge(
            model=args.grounding_model,
            system_prompt=GROUNDING_SYSTEM,
            user_prompt=grounding_user,
            temperature=args.temperature,
        )
        grounding_metrics = compute_grounding_metrics(grounding_obj, extracted)

        # ---- Aggregate ----
        agg["sum_plausibility"] += float(set_metrics["plausibility"])
        if set_metrics["h_coverage"] is not None:
            agg["sum_hcov"] += float(set_metrics["h_coverage"])
            agg["n_hcov_defined"] += 1

        agg["sum_support_rate"] += float(grounding_metrics["support_rate"])
        agg["sum_hall_evidence_rate"] += float(grounding_metrics["hallucinated_evidence_rate"])

        agg["sum_breadth"] += float(breadth_metrics["breadth"])
        if breadth_metrics["normalized_breadth"] is not None:
            agg["sum_norm_breadth"] += float(breadth_metrics["normalized_breadth"])
            agg["n_norm_breadth_defined"] += 1

        if uncertainty_flag:
            agg["n_uncertain"] += 1

        # Optional: keep reference diagnosis if present
        ref_dx = sample.get("reference_diagnosis", None)
        if ref_dx is None and isinstance(sample.get("reference", None), dict):
            ref_dx = sample["reference"].get("reference_diagnosis", None)

        per_sample.append({
            "index": idx,
            "has_pxhx": True,
            "input": question,
            "model_response": model_answer,
            "reference_diagnosis": ref_dx,
            "judge_dx_space": {
                "plausible_set": P,
                "highly_likely_set": H,
            },
            "metrics": {
                "plausibility": set_metrics["plausibility"],
                "h_coverage": set_metrics["h_coverage"],

                "extracted_diagnoses": set_metrics["extracted_diagnoses"],
                "in_P": set_metrics["in_P"],
                "out_of_P": set_metrics["out_of_P"],
                "covered_H": set_metrics["covered_H"],
                "uncovered_H": set_metrics["uncovered_H"],

                "uncertainty_flag": uncertainty_flag,

                "breadth": breadth_metrics["breadth"],
                "normalized_breadth": breadth_metrics["normalized_breadth"],

                "support_rate": grounding_metrics["support_rate"],
                "hallucinated_evidence_rate": grounding_metrics["hallucinated_evidence_rate"],
                "grounding_per_diagnosis": grounding_metrics["per_diagnosis"],
            }
        })

        done_idx.add(idx)

        # periodically save semantic cache to disk
        if (idx + 1) % args.save_every == 0:
            if args.sem_cache_path:
                matcher.save_cache()
            checkpoint_save(resume_path, per_sample, agg, matcher, models_dict)

    # Save cache at end
    if args.sem_cache_path:
        matcher.save_cache()

    checkpoint_save(args.output_path, per_sample, agg, matcher, models_dict)

    print("=== SUMMARY ===")
    print(json.dumps(make_summary(agg, len(matcher.cache), models_dict), indent=2, ensure_ascii=False))
    print(f"\nSaved to {args.output_path}")

if __name__ == "__main__":
    main()
