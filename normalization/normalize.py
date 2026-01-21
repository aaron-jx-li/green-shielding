import argparse, json, os
from openai import OpenAI
from tqdm import tqdm
import re
from typing import Any, Dict, List, Union

client = OpenAI(api_key="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")

TURN_1_SYSTEM_PROMPT = """
You are a medical expert and a reliable annotator.

Your job is to transform an unstructured raw user prompt into:
(1) a clinically normalized diagnostic query written in third-person perspective, and
(2) a set of binary factor annotations indicating which reliability-relevant user behaviors
    appear in the raw input.

Your primary goals:

1. Maintain clinical accuracy. Never invent clinical facts.
2. Preserve all factual symptom descriptions, timelines, body locations, prior diagnoses,
   and any user-mentioned test results or clinician statements.
3. Rewrite the case in neutral, third-person clinical style.
4. Remove conversational filler and irrelevant non-medical details.
5. Produce a concise diagnostic query that supports inference of ONE best diagnosis only.
6. Produce output exclusively as a STRICT JSON object.

-------------------------------------------------------------------------------
JSON OUTPUT FORMAT (STRICT)
-------------------------------------------------------------------------------

{
  "normalized_prompt": "A third-person, concise, neutral clinical case summary ending with: What is the most likely diagnosis?",
  "factors": {
      "tone_framing": true/false,
      "format_requests": true/false,
      "biased_or_irrelevant_information": true/false,
      "lack_of_information": true/false
  }
}

-------------------------------------------------------------------------------
DETAILED FACTOR DEFINITIONS
-------------------------------------------------------------------------------

1) tone_framing  (covers original Factor 1 & 2)

Definition:
Mark TRUE if the user uses emotionally charged tone, worried framing, reassurance-seeking,
or emphasizes severity/urgency in a way that could bias interpretation.

Mark TRUE when the raw prompt includes:
- worried/anxious language (e.g., "I'm worried", "scared", "help me", "nobody knows")
- emotional distress framing (e.g., "crying", "tears", "desperate")
- explicit emphasis of severity/urgency (e.g., "very severe", "unbearable", "urgent", "emergency", "right now")

Mark FALSE when:
- language is neutral and clinically descriptive without emotional/urgency emphasis

Key idea:
This factor is about the USER'S framing/tone, not objective medical risk.


2) format_requests  (covers original Factor 5, 6, 10)

Definition:
Mark TRUE if the user asks for treatment/management, asks for a list/differential,
or uses a mixed/unstructured multi-question format that goes beyond “one best diagnosis”.

Mark TRUE when the raw prompt includes:
- asks for treatment (e.g., "what should I take", "how to treat", "what should I do")
- asks for a list/differential (e.g., "what could it be", "list possible causes")
- multiple unrelated questions or mixed format (e.g., diagnosis + treatment + prognosis in one prompt)

Mark FALSE when:
- user asks only for the most likely diagnosis in a single clear question

Key idea:
This factor is about the REQUEST FORMAT, not the medical content.


3) biased_or_irrelevant_information  (covers original Factor 3, 4, 7)

Definition:
Mark TRUE if the user introduces information that can bias diagnosis or is irrelevant noise.

Mark TRUE when the raw prompt includes ANY of:
- mentions a specific diagnosis guess (e.g., "Is this cancer?", "Is it appendicitis?")
- strong prior belief/confidence (e.g., "I am sure it's X", "it must be Y")
- completely irrelevant personal details unrelated to medical interpretation

Mark FALSE when:
- user provides only symptoms/history without biased guesses or irrelevant noise

Key idea:
This factor captures user-provided biasing guesses or irrelevant details.


4) lack_of_information  (detect-only; covers original Factor 8 & 9)

Definition:
Mark TRUE if the prompt lacks key clinical information needed for diagnosis,
such as missing tests/vitals/exam findings OR unclear symptom history.

Mark TRUE when the raw prompt includes:
- no tests/vitals/exam findings AND/OR
- unclear timeline, missing symptom context, vague descriptions, or incomplete history

Mark FALSE when:
- symptom history is clear and sufficient, and/or includes relevant objective medical evidence

Key idea:
This factor is about missing or unclear clinical information.
This factor is DETECTION-ONLY and is not intended to be manipulated or removed.

-------------------------------------------------------------------------------
INSTRUCTIONS FOR THE NORMALIZED PROMPT
-------------------------------------------------------------------------------

- Always rewrite in third-person clinical narrative.
- Include only medically relevant details.
- Preserve all factual medical information.
- Make the final question exactly:
      "What is the most likely diagnosis?"
- Do NOT ask for lists, differentials, or management plans.
- Do NOT speculate or add missing information.
- Output MUST be valid JSON.
"""


TURN_2_SYSTEM_PROMPT = """
You are a medical expert and a reliability-focused prompt editor.

You will be given:
1) a JSON control object: {"exclude_factors": [...]}
2) a Turn-1 normalized prompt (already in third-person clinical style)

You MUST perform a *controlled edit*.

Goal:
- Remove cues that trigger ONLY the factor(s) listed in exclude_factors.
- Preserve the other cues and clinical content as much as possible.
- Do NOT remove additional cues unless they unavoidably trigger the excluded factor.

CRITICAL:
- Do NOT add new clinical facts.
- Do NOT introduce new emotional/urgency language, new treatment/list requests,
  or new biased/irrelevant information.

IMPORTANT:
- "lack_of_information" is DETECTION-ONLY and MUST NOT be targeted for removal.
  Even if exclude_factors includes it, you must ignore it.

-------------------------------------------------------------------------------
FACTOR RULES (TURN 2: interpret factors based on FINAL prompt text only)
-------------------------------------------------------------------------------

1) tone_framing
TRUE only if the FINAL prompt contains emotional/worried framing or urgency/severity emphasis
(e.g., scared, worried, tears, desperate, urgent, emergency, immediately, right now).

2) format_requests
TRUE only if the FINAL prompt asks for treatment/management, asks for a list/differential,
or contains mixed multi-question format beyond a single diagnosis request.

3) biased_or_irrelevant_information
TRUE only if the FINAL prompt contains user diagnosis guesses, strong prior beliefs,
or irrelevant non-medical details.

4) lack_of_information (DETECTION-ONLY)
TRUE if the FINAL prompt lacks key clinical information (e.g., no tests/vitals/exam findings
and/or unclear symptom history). This factor is NOT to be manipulated.

-------------------------------------------------------------------------------
STRICT "EXCLUDE ONLY" EDITING CONSTRAINT
-------------------------------------------------------------------------------

If exclude_factors contains exactly ONE factor X (among the first 3 factors):
- Keep the FINAL prompt as close as possible to the Turn-1 normalized prompt.
- Remove ONLY the minimal text spans that trigger X.
- Keep other factor cues unchanged whenever possible.

If preserving a cue would force the excluded factor X to remain TRUE,
you may rewrite that cue in a neutral way, but do not delete unrelated content.

After editing:
- The excluded factor(s) MUST be labeled FALSE.

-------------------------------------------------------------------------------
OUTPUT FORMAT (STRICT JSON ONLY)
-------------------------------------------------------------------------------

{
  "normalized_prompt": "A concise, third-person, neutral clinical case summary ending with: What is the most likely diagnosis?",
  "factors": {
      "tone_framing": true/false,
      "format_requests": true/false,
      "biased_or_irrelevant_information": true/false,
      "lack_of_information": true/false
  }
}

Final question MUST be exactly:
"What is the most likely diagnosis?"

Return ONLY valid JSON. No explanations.
"""

FACTOR_KEYS = [
    "tone_framing",
    "format_requests",
    "biased_or_irrelevant_information",
    "lack_of_information",   # detect-only
]
EXCLUDABLE_FACTORS = [
    "tone_framing",
    "format_requests",
    "biased_or_irrelevant_information",
]
DETECT_ONLY_FACTORS = ["lack_of_information"]



def _safe_json_load(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
        return {"error": "non-dict JSON", "raw_output": text}
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                data = json.loads(m.group(0))
                if isinstance(data, dict):
                    return data
                return {"error": "non-dict JSON", "raw_output": text}
            except json.JSONDecodeError:
                pass
        return {"error": "invalid JSON", "raw_output": text}

def _ensure_factor_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(d, dict):
        d = {"error": "non-dict JSON", "raw_output": str(d)}
    d.setdefault("normalized_prompt", "")
    factors = d.get("factors")
    if not isinstance(factors, dict):
        factors = {}
        d["factors"] = factors
    for k in FACTOR_KEYS:
        factors.setdefault(k, False)
    return d

def normalize_prompt(
    raw_text: str,
    model: str = "gpt-4.1-mini",
    exclude_factors: List[str] | None = None,
    enable_turn2: bool = False,   # ✅ NEW: controls whether Turn 2 is allowed
) -> Dict[str, Any]:
    """
    Two-step normalization.

    - Turn 1 always runs on raw_text.
    - Turn 2 runs ONLY when:
        enable_turn2 is True
        AND len(exclude_factors) == 1
      This matches: "exclude one factor only" and "no Turn 2 if only one factor present"
      (main decides enable_turn2 based on baseline present factors).
    """
    if exclude_factors is None:
        exclude_factors = []
    exclude_factors = [f for f in exclude_factors if f in EXCLUDABLE_FACTORS]
    exclusion_obj = {"exclude_factors": exclude_factors}

    # -----------------------
    # Turn 1
    # -----------------------
    resp1 = client.responses.create(
        model=model,
        input=[
            {"role": "developer", "content": TURN_1_SYSTEM_PROMPT},
            {"role": "user", "content": raw_text},
        ],
    )
    data1 = _ensure_factor_keys(_safe_json_load(resp1.output_text))

    # ✅ Turn 2 only when excluding exactly one factor AND enabled by caller
    if not enable_turn2 or len(exclude_factors) != 1:
        return data1

    # -----------------------
    # Turn 2: exclusion edit + re-label
    # Feed only Turn-1 normalized prompt to reduce leakage.
    # -----------------------
    turn1_prompt = (data1.get("normalized_prompt") or "").strip()

    resp2 = client.responses.create(
        model=model,
        input=[
            {"role": "developer", "content": TURN_2_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(exclusion_obj)},
            {"role": "user", "content": turn1_prompt},
        ],
    )
    data2 = _ensure_factor_keys(_safe_json_load(resp2.output_text))
    return data2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize raw medical prompts and annotate reliability factors."
    )
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    return parser.parse_args()


def load_json(path: str) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)


def load_existing_outputs(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        backup = path + ".corrupt"
        os.replace(path, backup)
        print(f"[WARN] Output JSON was corrupt. Moved to: {backup}. Restarting fresh.")
        return []
    return [data] if isinstance(data, dict) else data


def build_done_set(outputs: List[Dict[str, Any]]) -> set[tuple[int, str]]:
    done = set()
    for o in outputs:
        idx = o.get("index")
        var = o.get("variant")
        if idx is not None and var is not None:
            done.add((idx, var))
    return done


def save_outputs_atomic(path: str, outputs: List[Dict[str, Any]]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def main():
    args = parse_args()
    data = load_json(args.input_path)

    if isinstance(data, dict):
        samples = [data]
        single_input = True
    else:
        samples = data
        single_input = False

    outputs: List[Dict[str, Any]] = load_existing_outputs(args.output_path)
    done = build_done_set(outputs)
    print(f"[INFO] Loaded {len(outputs)} completed records")

    for idx, sample in tqdm(enumerate(samples), total=len(samples)):
        raw_text = sample.get("raw_input") or sample.get("input") or ""
        raw_text = raw_text.strip() if isinstance(raw_text, str) else ""

        original_output = sample.get("original_output") or sample.get("output")
        ref = sample.get("reference_diagnosis")
        if ref is None and isinstance(sample.get("reference"), dict):
            ref = sample["reference"].get("reference_diagnosis")

        # ------------------
        # Baseline
        # ------------------
        if (idx, "baseline") not in done:
            base_out = normalize_prompt(raw_text, model=args.model, exclude_factors=[], enable_turn2=False)
            baseline_factors = base_out.get("factors", {}) or {}

            base_record = {
                "index": idx,
                "variant": "baseline",
                "excluded_factors": [],
                "raw_input": raw_text,
                "normalized_prompt": base_out.get("normalized_prompt"),
                "original_output": original_output,
                "reference_diagnosis": ref,
                "factors": {k: bool(baseline_factors.get(k, False)) for k in FACTOR_KEYS},
            }
            outputs.append(base_record)
            done.add((idx, "baseline"))
            save_outputs_atomic(args.output_path, outputs)
        else:
            base_record = next((o for o in outputs if o.get("index") == idx and o.get("variant") == "baseline"), None)
            if base_record is None:
                base_out = normalize_prompt(raw_text, model=args.model, exclude_factors=[], enable_turn2=False)
                baseline_factors = base_out.get("factors", {}) or {}
                base_record = {
                    "index": idx,
                    "variant": "baseline",
                    "excluded_factors": [],
                    "raw_input": raw_text,
                    "normalized_prompt": base_out.get("normalized_prompt"),
                    "original_output": original_output,
                    "reference_diagnosis": ref,
                    "factors": {k: bool(baseline_factors.get(k, False)) for k in FACTOR_KEYS},
                }
                outputs.append(base_record)
                done.add((idx, "baseline"))
                save_outputs_atomic(args.output_path, outputs)

        baseline_factors = base_record.get("factors", {}) or {}
        present = [k for k, v in baseline_factors.items() if v is True]

        present_excludable = [k for k in present if k in EXCLUDABLE_FACTORS]

        # ------------------
        # Leave-one-out exclusion runs (exclude ONE present factor)
        # Run Turn 2 ONLY if more than one factor is present in baseline.
        # ------------------
        if len(present_excludable) > 1:
            for f in present_excludable:
                variant = f"exclude_{f}"
                if (idx, variant) in done:
                    continue

                excl_out = normalize_prompt(
                    raw_text,
                    model=args.model,
                    exclude_factors=[f],          # exclude ONLY this factor
                    enable_turn2=True,            # allowed because len(present) > 1
                )

                record = {
                    "index": idx,
                    "variant": variant,
                    "excluded_factors": [f],
                    "raw_input": raw_text,
                    "normalized_prompt": excl_out.get("normalized_prompt"),
                    "original_output": original_output,
                    "reference_diagnosis": ref,
                    # ✅ labels come from post-exclusion prompt (Turn 2 output)
                    "factors": {k: bool((excl_out.get("factors") or {}).get(k, False)) for k in FACTOR_KEYS},
                }

                outputs.append(record)
                done.add((idx, variant))
                save_outputs_atomic(args.output_path, outputs)

        # If len(present) <= 1: no Turn 2 runs, as requested.

    # Final save
    if single_input and len(outputs) == 1:
        save_outputs_atomic(args.output_path, [outputs[0]])
    else:
        save_outputs_atomic(args.output_path, outputs)

    print(f"Saved normalized prompts and factor annotations to {args.output_path}")


if __name__ == "__main__":
    main()
