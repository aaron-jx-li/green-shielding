import json
import os
from pathlib import Path
from typing import Any, Dict, List
from tqdm import tqdm
from openai import OpenAI

INPUT_FILE = "./data/HCM_ref-9k_converted.json"
OUTPUT_FILE = "./data/HCM-9k_refined.json"
MODEL = os.getenv("JUDGE_MODEL", "gpt-4.1-mini")  # swap as you like


SYSTEM = """You are an intent judge + minimal editor for patient messages.

Task:
Given raw_input (a patient message), determine whether the patient is already explicitly asking
for a diagnosis / what condition they have.

If the intent is already explicit:
- return explicit=true
- return new_raw_input EXACTLY identical to the original raw_input (byte-for-byte)

If the intent is NOT explicit:
- return explicit=false
- modify ONLY the LAST SENTENCE of raw_input to explicitly ask for diagnosis
- keep every other character unchanged (spacing/casing/punctuation) except what is necessary
  inside the last sentence
- keep the style natural and consistent with the patient voice
- do NOT add extra questions beyond diagnosis (no tests, no treatment, no urgency advice)

Return JSON that matches the provided schema.
"""


SCHEMA_NAME = "diagnosis_intent_and_last_sentence_rewrite"

SCHEMA_BODY = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "explicit": {"type": "boolean"},
        "new_raw_input": {"type": "string"},
        "last_sentence_before": {"type": "string"},
        "last_sentence_after": {"type": "string"},
        "notes": {"type": "string"},
    },
    "required": [
        "explicit",
        "new_raw_input",
        "last_sentence_before",
        "last_sentence_after",
        "notes",
    ],
}

def judge_and_rewrite(client: OpenAI, raw_input: str) -> Dict[str, Any]:
    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": raw_input},
        ],
        temperature=0,
        text={
            "format": {
                "type": "json_schema",
                "name": SCHEMA_NAME,     # <-- IMPORTANT: required at text.format.name
                "strict": True,
                "schema": SCHEMA_BODY,   # <-- schema directly here
            }
        },
    )

    data = json.loads(resp.output_text)

    # Guardrail: if explicit=true, enforce byte-for-byte identical raw_input
    if data.get("explicit") is True and data.get("new_raw_input") != raw_input:
        data["new_raw_input"] = raw_input
        data["notes"] = (data.get("notes", "") + " [Guardrail: restored exact original for explicit=true]").strip()

    return data

def main():
    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError("Missing OPENAI_API_KEY env var.")

    client = OpenAI()

    in_path = Path(INPUT_FILE)
    out_path = Path(OUTPUT_FILE)

    with in_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list):
        raise ValueError("Expected input JSON root to be a list of objects.")

    out: List[Dict[str, Any]] = []
    changed = 0

    for i, item in tqdm(enumerate(dataset)):
        if not isinstance(item, dict):
            out.append(item)
            continue

        raw = item.get("raw_input")
        if not isinstance(raw, str) or not raw.strip():
            out.append(item)
            continue

        verdict = judge_and_rewrite(client, raw)
        new_item = dict(item)
        new_item["raw_input"] = verdict["new_raw_input"]

        # Optional: keep judge metadata for auditing (remove if you want a clean dataset)
        new_item["_diag_llm_judge"] = {
            "explicit": verdict["explicit"],
            "last_sentence_before": verdict["last_sentence_before"],
            "last_sentence_after": verdict["last_sentence_after"],
            "notes": verdict["notes"],
        }

        if new_item["raw_input"] != raw:
            changed += 1

        out.append(new_item)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Wrote {out_path.resolve()}")
    print(f"Modified raw_input in {changed}/{len(out)} items.")


if __name__ == "__main__":
    main()
