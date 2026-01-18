#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
import math
import hashlib
from typing import Any, Dict, List, Optional, Set

from tqdm import tqdm
from openai import OpenAI


# ============================================================
# Aggregator prompt: N-model semantic clustering + majority vote
# ============================================================

AGG_SYSTEM = """You are a medical terminology normalizer and voting aggregator.

You will be given:
- vote_threshold: an integer
- P_lists: list[list[str]]  (one list per model)
- H_lists: list[list[str]]  (one list per model)
- C_lists: list[list[str]]  (one list per model)  # C = cannot-miss
- H_evidence_maps: list[dict[str, list[str]]] (one per model; evidence strings for each dx in that model's H)
- C_evidence_maps: list[dict[str, list[str]]] (one per model; evidence strings for each dx in that model's C)

Task:
1) For PLAUSIBLE diagnoses:
   - Group diagnosis strings across models into buckets that refer to the SAME diagnostic entity for evaluation.
   - Treat the following relations as SAME bucket: synonyms, spelling variants, abbreviations,
     subtype/supertype, manifestation, etiology.
   - Do NOT merge items that are merely "related" or co-occurring.
   - For each bucket choose ONE canonical name (short, standard).
   - Support of a bucket = number of DISTINCT models that contributed at least one string to the bucket.
   - Keep a bucket if support >= vote_threshold.

2) Repeat the same for HIGHLY LIKELY diagnoses (H_lists),
   BUT keep at most 3 HIGHLY LIKELY diagnoses total.
   If more than 3 buckets satisfy the vote_threshold:
     - prefer buckets with higher support (more models),
     - if tie, prefer buckets with stronger consensus across models,
     - if still tie, prefer more specific / clinically decisive diagnoses.

3) Repeat the same for CANNOT MISS diagnoses (C_lists),
   BUT keep this set SMALL and safety-focused (typically 0–3).
   If more than 3 buckets satisfy the vote_threshold:
     - prefer buckets with higher support,
     - if tie, prefer diagnoses that are more urgent / time-sensitive / high-risk to miss.

4) Enforce subset constraints by bucket identity:
   - H ⊆ P: If a kept H bucket is not present among kept P buckets, add its canonical name to plausible_set.
   - C ⊆ P: If a kept C bucket is not present among kept P buckets, add its canonical name to plausible_set.

5) Evidence aggregation:
   - For each kept H canonical diagnosis, return up to 3 evidence strings.
   - For each kept C canonical diagnosis, return up to 3 evidence strings.
   - Evidence strings must come from the provided *_evidence_maps values (do NOT invent evidence).
   - Prefer evidence strings supported by multiple models; otherwise choose concise ones.

Hard constraints:
- Do NOT invent diagnoses not present in any input list.
- Do NOT return more than 3 items in highly_likely_set.
- Do NOT return more than 3 items in cannot_miss_set.
- Return STRICT JSON only.
- Use the schema exactly:
{
  "plausible_set": [ ...canonical dx... ],
  "highly_likely_set": [ ...canonical dx... ],
  "cannot_miss_set": [ ...canonical dx... ],
  "highly_likely_evidence": { "<canonical dx>": ["e1","e2",...], ... },
  "cannot_miss_evidence": { "<canonical dx>": ["e1","e2",...], ... },
  "support": {
    "plausible_set": { "<canonical dx>": [model_ids...], ... },
    "highly_likely_set": { "<canonical dx>": [model_ids...], ... },
    "cannot_miss_set": { "<canonical dx>": [model_ids...], ... }
  }
}
"""


# ============================================================
# Utilities
# ============================================================

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def parse_json_strict(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))

def to_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    s = str(x).strip()
    return [s] if s else []

def dedup_preserve(xs: List[str]) -> List[str]:
    out, seen = [], set()
    for x in xs:
        x = (x or "").strip()
        if not x:
            continue
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out

def get_idx(rec: Dict[str, Any], fallback: int) -> int:
    if isinstance(rec, dict) and "__idx" in rec:
        try:
            return int(rec["__idx"])
        except Exception:
            return fallback
    return fallback

def safe_get_gt(rec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(rec, dict):
        return {}
    gt = rec.get("ground_truth_space", {})
    return gt if isinstance(gt, dict) else {}

def index_records(recs: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for i, r in enumerate(recs):
        if isinstance(r, dict):
            out[get_idx(r, i)] = r
    return out

def stable_signature(obj: Any) -> str:
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ============================================================
# Aggregator LLM (1 call per record) + optional cache
# ============================================================

class Aggregator:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        retries: int = 4,
        base_delay: float = 0.8,
        cache_path: Optional[str] = None,
    ):
        self.client = client
        self.model = model
        self.retries = retries
        self.base_delay = base_delay
        self.cache_path = cache_path
        self.cache: Dict[str, Dict[str, Any]] = {}

        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        sig = obj.get("sig")
                        out = obj.get("out")
                        if isinstance(sig, str) and isinstance(out, dict):
                            self.cache[sig] = out
            except Exception:
                pass

    def _append_cache(self, sig: str, out: Dict[str, Any]) -> None:
        if not self.cache_path:
            return
        try:
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sig": sig, "out": out}, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def aggregate(
        self,
        vote_threshold: int,
        P_lists: List[List[str]],
        H_lists: List[List[str]],
        C_lists: List[List[str]],
        H_evidence_maps: List[Dict[str, List[str]]],
        C_evidence_maps: List[Dict[str, List[str]]],
    ) -> Dict[str, Any]:
        payload = {
            "vote_threshold": int(vote_threshold),
            "P_lists": P_lists,
            "H_lists": H_lists,
            "C_lists": C_lists,
            "H_evidence_maps": H_evidence_maps,
            "C_evidence_maps": C_evidence_maps,
        }
        sig = stable_signature(payload)
        if sig in self.cache:
            return self.cache[sig]

        user_msg = "INPUT JSON:\n" + json.dumps(payload, ensure_ascii=False, indent=2)

        last_err: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                resp = self.client.responses.create(
                    model=self.model,
                    temperature=0.0,
                    input=[
                        {"role": "system", "content": AGG_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                )
                out = parse_json_strict(resp.output_text)

                plausible = dedup_preserve(to_list(out.get("plausible_set", [])))
                likely = dedup_preserve(to_list(out.get("highly_likely_set", [])))[:3]
                cannot_miss = dedup_preserve(to_list(out.get("cannot_miss_set", [])))[:3]

                ev_h = out.get("highly_likely_evidence", {}) or {}
                if not isinstance(ev_h, dict):
                    ev_h = {}
                ev_c = out.get("cannot_miss_evidence", {}) or {}
                if not isinstance(ev_c, dict):
                    ev_c = {}

                ev_h_clean: Dict[str, List[str]] = {}
                for dx in likely:
                    ev_h_clean[dx] = dedup_preserve(to_list(ev_h.get(dx, [])))[:3]

                ev_c_clean: Dict[str, List[str]] = {}
                for dx in cannot_miss:
                    ev_c_clean[dx] = dedup_preserve(to_list(ev_c.get(dx, [])))[:3]

                # ensure H ⊆ P and C ⊆ P
                p_lower = {x.lower() for x in plausible}
                for dx in likely:
                    if dx.lower() not in p_lower:
                        plausible.append(dx)
                        p_lower.add(dx.lower())
                for dx in cannot_miss:
                    if dx.lower() not in p_lower:
                        plausible.append(dx)
                        p_lower.add(dx.lower())

                support = out.get("support", {}) or {}
                if not isinstance(support, dict):
                    support = {}

                final = {
                    "plausible_set": plausible,
                    "highly_likely_set": likely,
                    "cannot_miss_set": cannot_miss,
                    "highly_likely_evidence": ev_h_clean,
                    "cannot_miss_evidence": ev_c_clean,
                    "support": support,
                }

                self.cache[sig] = final
                self._append_cache(sig, final)
                return final
            except Exception as e:
                last_err = e
                time.sleep(min(self.base_delay * (2 ** attempt), 8.0))

        raise RuntimeError(f"Aggregator failed after retries: {last_err}")


# ============================================================
# Merge records
# ============================================================

def merge_n_models_fast(
    indexed: List[Dict[int, Dict[str, Any]]],
    aggregator: Aggregator,
    vote_threshold: int,
    keep_fields_from: int = 0,
    max_items: int = 0,
) -> List[Dict[str, Any]]:
    all_idx: Set[int] = set()
    for m in indexed:
        all_idx |= set(m.keys())

    out: List[Dict[str, Any]] = []
    K = len(indexed)

    for j, idx in enumerate(tqdm(sorted(all_idx), desc="Merging majority ground truth")):
        if max_items and j >= max_items:
            break

        recs = [m.get(idx) for m in indexed]

        base: Dict[str, Any] = {"__idx": idx}
        base_src = recs[keep_fields_from] if 0 <= keep_fields_from < K else None
        if not isinstance(base_src, dict):
            base_src = next((r for r in recs if isinstance(r, dict)), None)

        if isinstance(base_src, dict):
            base.update({k: v for k, v in base_src.items() if k not in ("ground_truth_space", "errors")})

        P_lists: List[List[str]] = []
        H_lists: List[List[str]] = []
        C_lists: List[List[str]] = []
        H_evidence_maps: List[Dict[str, List[str]]] = []
        C_evidence_maps: List[Dict[str, List[str]]] = []

        for r in recs:
            gt = safe_get_gt(r)
            P_lists.append(dedup_preserve(to_list(gt.get("plausible_set"))))
            H_lists.append(dedup_preserve(to_list(gt.get("highly_likely_set"))))
            C_lists.append(dedup_preserve(to_list(gt.get("cannot_miss_set"))))

            ev_h = gt.get("highly_likely_evidence", {}) or {}
            H_evidence_maps.append(ev_h if isinstance(ev_h, dict) else {})

            ev_c = gt.get("cannot_miss_evidence", {}) or {}
            C_evidence_maps.append(ev_c if isinstance(ev_c, dict) else {})

        try:
            merged_gt = aggregator.aggregate(
                vote_threshold=vote_threshold,
                P_lists=P_lists,
                H_lists=H_lists,
                C_lists=C_lists,
                H_evidence_maps=H_evidence_maps,
                C_evidence_maps=C_evidence_maps,
            )

            base["ground_truth_space_majority"] = {
                "plausible_set": merged_gt["plausible_set"],
                "highly_likely_set": merged_gt["highly_likely_set"],
                "cannot_miss_set": merged_gt["cannot_miss_set"],
                "highly_likely_evidence": merged_gt["highly_likely_evidence"],
                "cannot_miss_evidence": merged_gt["cannot_miss_evidence"],
            }

        except Exception as e:
            base["ground_truth_space_majority"] = {
                "plausible_set": [],
                "highly_likely_set": [],
                "cannot_miss_set": [],
                "highly_likely_evidence": {},
                "cannot_miss_evidence": {},
            }
            base["errors"] = [f"merge_error: {repr(e)}"]

        out.append(base)

    return out


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="List of model output JSON files (each is a JSON array with ground_truth_space).",
    )
    ap.add_argument("--out", required=True, help="Output merged majority JSON")

    ap.add_argument("--agg_model", default="gpt-4.1-mini", help="LLM model for 1-call aggregation")
    ap.add_argument("--agg_cache", default=None, help="Optional JSONL cache for aggregation calls")

    ap.add_argument(
        "--vote_threshold",
        type=int,
        default=2,
        help="Dx kept if supported by >= this many models. Default: majority ceil(N/2).",
    )
    ap.add_argument(
        "--keep_fields_from",
        type=int,
        default=0,
        help="Which input file index to copy non-gt fields from (default: 0).",
    )
    ap.add_argument("--max_items", type=int, default=0, help="0=all, else limit records")

    args = ap.parse_args()

    data_list: List[List[Dict[str, Any]]] = []
    for p in args.inputs:
        data = load_json(p)
        if not isinstance(data, list):
            raise ValueError(f"Input {p} must be a JSON array (list).")
        data_list.append(data)

    N = len(data_list)
    if N < 2:
        raise ValueError("Need at least 2 input files for majority voting.")

    vote_threshold = args.vote_threshold if args.vote_threshold > 0 else int(math.ceil(N / 2))

    indexed = [index_records(d) for d in data_list]

    client = OpenAI()
    aggregator = Aggregator(client=client, model=args.agg_model, cache_path=args.agg_cache)

    merged = merge_n_models_fast(
        indexed=indexed,
        aggregator=aggregator,
        vote_threshold=vote_threshold,
        keep_fields_from=args.keep_fields_from,
        max_items=args.max_items,
    )

    save_json(args.out, merged)
    print(f"Wrote merged majority ground truth to: {args.out}")
    print(f"Models: {N}, vote_threshold: {vote_threshold}")

if __name__ == "__main__":
    main()
