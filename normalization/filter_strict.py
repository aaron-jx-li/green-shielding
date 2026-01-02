#!/usr/bin/env python3
import argparse
import json
import time
from typing import Dict, Any, Iterable, List, Optional
from openai import OpenAI
from tqdm import tqdm

CLIENT = OpenAI()

# =========================
# Judge prompt
# =========================
SYSTEM_PROMPT = """You are a meticulous evaluator for a physician Q&A dataset.

You will be given:
- raw_input: the patient's original message (verbatim)

Task:
Decide whether the raw_input contains an EXPLICIT request for a diagnosis / cause.

Return "yes" ONLY if the patient explicitly asks for diagnosis/cause using language like:
- "what is the diagnosis", "what could this be", "what is causing this",
- "what condition do I have", "what might be wrong", "most likely diagnosis",
- or clearly asks the doctor to identify the condition/cause.

Return "no" if the patient only:
- asks what to do / how to treat / whether it's serious,
- asks for general info or prognosis,
- asks about safety (e.g., flying/diving) WITHOUT explicitly asking what it is,
- or the diagnostic ask is only implied but not explicitly requested.

Important:
- If the message includes BOTH treatment/safety questions AND an explicit "what is it/diagnosis/cause" question, answer "yes".
- Be conservative: when unsure, answer "no".

Return STRICT JSON ONLY:
{
  "explicit_diagnosis_ask": "yes" | "no",
  "confidence": 1-5,
  "rationale": "<= 2 short sentences, cite the exact phrase(s) that triggered your decision"
}
"""

# =========================
# IO
# =========================
def load_records(path: str) -> List[Dict[str, Any]]:
    """
    Load JSON array or JSONL.
    """
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        else:
            return [json.loads(line) for line in f if line.strip()]

def save_json(path: str, data: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# =========================
# Judge call utilities
# =========================
def _extract_output_text(resp: Any) -> str:
    """
    Try multiple access patterns across SDK versions.
    """
    # Newer Responses API
    if hasattr(resp, "output_text") and isinstance(resp.output_text, str):
        return resp.output_text
    # Fallback: try dict-like
    if isinstance(resp, dict) and "output_text" in resp and isinstance(resp["output_text"], str):
        return resp["output_text"]
    # Last resort: stringify
    return str(resp)

def call_judge(
    model: str,
    raw_input: str,
    max_retries: int = 6,
    base_delay: float = 1.0,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Query an LLM judge with exponential backoff.
    Returns parsed JSON.
    """
    user_payload = {"raw_input": raw_input}

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = CLIENT.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Evaluate:\n{json.dumps(user_payload, ensure_ascii=False)}",
                    },
                ],
                temperature=temperature,
            )

            text = _extract_output_text(resp).strip()

            # Some models may wrap JSON in code fences; strip if present.
            if text.startswith("```"):
                text = text.strip("`")
                # If it was ```json\n...\n``` form, try to pull the middle.
                lines = text.splitlines()
                if len(lines) >= 2:
                    # remove a leading language tag if present
                    if lines[0].lower().startswith("json"):
                        lines = lines[1:]
                    text = "\n".join(lines).strip()

            return json.loads(text)

        except Exception as e:
            last_err = e
            time.sleep(base_delay * (2 ** attempt))

    raise RuntimeError(f"Judge call failed after {max_retries} attempts: {last_err}")

def normalize_yesno(val: Any) -> str:
    s = str(val).strip().lower()
    return "yes" if s in {"y", "yes", "true", "1"} else "no"

# =========================
# Main filtering logic
# =========================
def filter_explicit_diagnosis_asks(
    records: List[Dict[str, Any]],
    model: str,
    max_examples: int = 0,
    keep_judge_fields: bool = True,
) -> List[Dict[str, Any]]:
    """
    Keeps only samples whose raw_input contains an explicit diagnosis request.
    Adds judge fields if keep_judge_fields=True.
    """
    filtered: List[Dict[str, Any]] = []
    n = 0

    for rec in tqdm(records, desc="Judging"):
        if max_examples and n >= max_examples:
            break

        raw = rec.get("raw_input", "")
        if not isinstance(raw, str):
            raw = "" if raw is None else str(raw)

        result = call_judge(model=model, raw_input=raw)

        flag = normalize_yesno(result.get("explicit_diagnosis_ask", "no"))
        conf = int(result.get("confidence", 3)) if str(result.get("confidence", "")).strip() else 3
        rat = str(result.get("rationale", "")).strip()

        if flag == "yes":
            if keep_judge_fields:
                rec_out = {
                    **rec,
                    "_explicit_dx_ask": {
                        "explicit": True,
                        "confidence": conf,
                        "rationale": rat,
                        "judge_model": model,
                    },
                }
            else:
                rec_out = rec
            filtered.append(rec_out)

        n += 1

    return filtered

def parse_args():
    ap = argparse.ArgumentParser(
        description="Filter dataset: keep only samples whose raw_input explicitly asks for a diagnosis (LLM judge)."
    )
    ap.add_argument("--in", dest="in_path", required=True, help="Input data (JSON array or JSONL)")
    ap.add_argument("--out", dest="out_path", required=True, help="Output JSON (filtered)")
    ap.add_argument("--model", default="gpt-4.1", help="Judge model (default: gpt-4.1)")
    ap.add_argument("--max", dest="max_examples", type=int, default=0, help="Max examples (0 = all)")
    ap.add_argument("--no_judge_fields", action="store_true", help="Do not attach judge metadata to kept records")
    return ap.parse_args()

def main():
    args = parse_args()
    records = load_records(args.in_path)
    print(f"Loaded {len(records)} records from {args.in_path}")

    filtered = filter_explicit_diagnosis_asks(
        records=records,
        model=args.model,
        max_examples=args.max_examples,
        keep_judge_fields=(not args.no_judge_fields),
    )

    print(f"Kept {len(filtered)} records with explicit diagnosis ask in raw_input.")
    save_json(args.out_path, filtered)
    print(f"Saved filtered JSON to {args.out_path}")

if __name__ == "__main__":
    main()
