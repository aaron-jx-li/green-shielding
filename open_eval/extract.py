#!/usr/bin/env python3
import argparse
import json
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm

CLIENT = OpenAI()

# =========================
# Extraction judge prompt
# =========================
SYSTEM_PROMPT = """You are a careful clinical information extractor.

You will be given:
- raw_input: a patient's original message (verbatim)

Your task:
Extract ONLY information present in raw_input into a JSON dict with EXACT keys:
{
  "demographics": [ ... ],
  "S": [ ... ],
  "O": [ ... ]
}

Definitions:
- demographics: patient attributes that are explicitly stated OR clearly and directly inferable from the text,
  such as age, sex/gender, weight, pregnancy status.
  Sex/gender may be inferred only if trivial and unambiguous, including from:
    - explicit statements ("33 male"),
    - patient self-reference ("I am pregnant"),
    - or pronouns / kinship terms that clearly refer to the patient ("he is 40", "my son is 1 year old").
  Do NOT infer from stereotypes, symptoms, or context.
  Do NOT include relationship itself (e.g., "brother"), only use it if needed to infer sex.
  Do NOT guess.

- S (Subjective): symptoms/complaints/feelings experienced by the patient, including symptom modifiers such as
  triggers, relievers, or temporal patterns (e.g., "burning improves with water", "pain worse at night").
  Do NOT include requests, intentions, questions, plans, or logistics.
- O (Objective): explicitly stated measurable findings, clinician-labeled results or diagnoses already given,
  clinician statements or recommendations, procedures already done, medications already taken,
  test/imaging results already reported.
  Examples: "HBV found in blood", "biopsy shows...", "two doctors recommended liver transplant",
  "X-ray normal", "partial root canal 36 hours ago", "temporary filling placed".

Critical constraints:
- COVER ALL presented clinically relevant information: every clinically relevant fact in raw_input must appear in
  either demographics, S, or O.
- DO NOT fabricate or perform medical reasoning: do not add facts not present (no staging, no likely diagnoses, no missing info lists).
- Do not restate the same fact in multiple sections.
- Prefer short, atomic bullet strings, but MERGE overlapping or redundant symptom descriptions into a single item
  when they describe the same phenomenon.
- If a test/procedure is mentioned but no result is provided, still include it in O (e.g., "biopsy performed (result not provided)").
- If demographics cannot be reasonably inferred, use an empty list [] rather than guessing.

Output rules:
- Return STRICT JSON ONLY (no markdown, no code fences, no extra keys).
"""

# =========================
# IO helpers
# =========================
def load_records(path: str) -> List[Dict[str, Any]]:
    """Load JSON array or JSONL."""
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]

def save_json(path: str, data: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# =========================
# OpenAI call
# =========================
def _get_output_text(resp: Any) -> str:
    # Newer Responses API
    if hasattr(resp, "output_text") and isinstance(resp.output_text, str):
        return resp.output_text
    # Dict-like fallback
    if isinstance(resp, dict) and isinstance(resp.get("output_text"), str):
        return resp["output_text"]
    # Last resort
    return str(resp)

def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text
    # Remove leading/trailing fences
    text = text.strip().strip("`").strip()
    lines = text.splitlines()
    # If first line is a language tag like "json", drop it
    if lines and lines[0].strip().lower() in {"json", "jsonl"}:
        lines = lines[1:]
    return "\n".join(lines).strip()

def call_extractor(
    model: str,
    raw_input: str,
    max_retries: int = 6,
    base_delay: float = 1.0,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Call LLM to extract demographics/S/O with strict no-inference rules."""
    payload = {"raw_input": raw_input}

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = CLIENT.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                temperature=temperature,
            )

            text = _strip_code_fences(_get_output_text(resp)).strip()
            obj = json.loads(text)

            # Strict schema validation
            if not isinstance(obj, dict):
                raise ValueError("Extractor did not return a JSON object.")

            required = {"demographics", "S", "O"}
            missing = required - set(obj.keys())
            if missing:
                raise ValueError(f"Missing keys: {sorted(missing)}")

            extra = set(obj.keys()) - required
            if extra:
                raise ValueError(f"Unexpected top-level keys: {sorted(extra)}")

            if not all(isinstance(obj[k], list) for k in ["demographics", "S", "O"]):
                raise ValueError("demographics, S, O must all be lists.")

            for k in ["demographics", "S", "O"]:
                for i, item in enumerate(obj[k]):
                    if not isinstance(item, str):
                        raise ValueError(f"{k}[{i}] is not a string.")

            return obj

        except Exception as e:
            last_err = e
            time.sleep(base_delay * (2 ** attempt))

    raise RuntimeError(f"Extractor failed after {max_retries} attempts: {last_err}")

# =========================
# Main runner
# =========================
def run_extraction(
    records: List[Dict[str, Any]],
    model: str,
    max_examples: int = 0,
    raw_key: str = "raw_input",
    attach_key: str = "extracted_state",
) -> List[Dict[str, Any]]:
    """Attach extracted_state to each record."""
    out: List[Dict[str, Any]] = []
    n = 0
    for rec in tqdm(records, desc="Extracting"):
        if max_examples and n >= max_examples:
            break

        raw = rec.get(raw_key, "")
        raw = "" if raw is None else str(raw)

        extracted = call_extractor(model=model, raw_input=raw)
        out.append({**rec, attach_key: extracted})
        n += 1

    return out

def parse_args():
    ap = argparse.ArgumentParser(
        description="LLM extraction: demographics(list) + S + O from raw_input (no inference, no relationship)."
    )
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSON array or JSONL")
    ap.add_argument("--out", dest="out_path", required=True, help="Output JSON array with extracted_state")
    ap.add_argument("--model", default="gpt-4.1", help="Extractor model (default: gpt-4.1)")
    ap.add_argument("--max", dest="max_examples", type=int, default=0, help="Max records (0=all)")
    ap.add_argument("--raw_key", default="raw_input", help="Key for raw input text (default: raw_input)")
    ap.add_argument("--attach_key", default="extracted", help="Key to store extracted dict (default: extracted_state)")
    return ap.parse_args()

def main():
    args = parse_args()
    records = load_records(args.in_path)
    print(f"Loaded {len(records)} records from {args.in_path}")

    extracted_records = run_extraction(
        records=records,
        model=args.model,
        max_examples=args.max_examples,
        raw_key=args.raw_key,
        attach_key=args.attach_key,
    )

    for rec in extracted_records:
        rec.pop("_explicit_dx_ask", None)

    save_json(args.out_path, extracted_records)
    print(f"Saved {len(extracted_records)} records with extractions to {args.out_path}")

if __name__ == "__main__":
    main()
