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
      "explicit_urgency_framing": true/false,
      "emotional_distress_framing": true/false,
      "medical_evidence_integration": true/false
  }
}

-------------------------------------------------------------------------------
DETAILED FACTOR DEFINITIONS
-------------------------------------------------------------------------------

1) explicit_urgency_framing

Definition:
Mark TRUE if the user explicitly frames the situation as requiring immediate attention,
emergency action, or urgent triage.

Mark TRUE when the raw prompt includes:
- Explicit urgency language (e.g., "urgent", "emergency", "right now", "immediately")
- Triage questions (e.g., "Should I go to the ER?", "Do I need immediate help?")
- Language implying imminent danger or rapid deterioration

Mark FALSE when:
- Symptoms are severe but not framed as urgent
- Chronic symptoms without emergency language
- High-risk medical history (e.g., cancer) without urgency wording

Key idea:
This factor captures urgency in the USER’S FRAMING, not objective medical risk.


2) emotional_distress_framing

Definition:
Mark TRUE if the user expresses emotional distress, anxiety, reassurance-seeking,
or psychologically charged framing that could bias interpretation.

Mark TRUE when the raw prompt includes:
- Emotional expressions (e.g., "scared", "crying", "tears", "desperate", "overwhelmed")
- Reassurance-seeking language (e.g., "please help", "nobody can tell me")
- Catastrophic or hopeless framing

Mark FALSE when:
- Language is neutral or clinically descriptive
- Severity is described without emotional language

Key idea:
This factor captures emotional or psychological framing, not symptom severity.


3) medical_evidence_integration

Definition:
Mark TRUE if the user introduces objective medical evidence or prior professional medical
assessment that constrains diagnosis.

Mark TRUE when the raw prompt includes ANY of the following:
- Lab results or values (e.g., "ALT 194", "WBC 15k", "CRP elevated")
- Imaging/test findings (e.g., "ultrasound showed cyst", "MRI showed herniation")
- Vital signs or measurements (e.g., "BP 180/110", "oxygen 88%")
- Prior clinician assessment/diagnosis/recommendation (e.g., "doctor said...", "oncologist advised...")
- Treatment trials and response/non-response (e.g., "antacids help", "NSAIDs didn't help")

Mark FALSE when:
- Only symptoms and timelines are provided without tests, measurements, clinician input, or treatment response
- Duration alone (e.g., "for 2 months") does NOT count as evidence

Key idea:
This factor is about TESTS/MEASUREMENTS/CLINICIAN INPUT/TREATMENT RESPONSE, not timelines.

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

You MUST perform a *controlled edit*:

Goal:
- Remove cues that trigger ONLY the factor(s) listed in exclude_factors.
- Preserve the other cues and clinical content as much as possible.
- Do NOT remove additional cues unless they unavoidably trigger the excluded factor.

CRITICAL:
- Do NOT add new clinical facts.
- Do NOT introduce new urgency language, new emotional language, or new medical evidence.

-------------------------------------------------------------------------------
FACTOR RULES (OVERRIDE ALL PRIOR DEFINITIONS FOR TURN 2)
-------------------------------------------------------------------------------

Interpret factors as properties of the FINAL prompt text only.

1) explicit_urgency_framing
TRUE only if the FINAL prompt contains explicit urgency/triage words
(e.g., urgent, emergency, immediately, right now, ER, life-threatening).

2) emotional_distress_framing
TRUE only if the FINAL prompt contains emotional or reassurance-seeking language
(e.g., scared, worried, tears, desperate, please help, nobody can tell me).
Clinical severity adjectives (e.g., "severe pain") do NOT count as emotional distress.

3) medical_evidence_integration
TRUE only if the FINAL prompt contains ANY of:
- lab values/results
- imaging/test findings
- vital signs/measurements
- prior clinician assessment/recommendation
- treatment response/non-response
NOTE: timelines/duration alone do NOT count as medical evidence.

-------------------------------------------------------------------------------
STRICT "EXCLUDE ONLY" EDITING CONSTRAINT
-------------------------------------------------------------------------------

If exclude_factors contains exactly ONE factor X:
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
      "explicit_urgency_framing": true/false,
      "emotional_distress_framing": true/false,
      "medical_evidence_integration": true/false
  }
}

Final question MUST be exactly:
"What is the most likely diagnosis?"

Return ONLY valid JSON. No explanations.
"""

FACTOR_KEYS = [
    "explicit_urgency_framing",
    "emotional_distress_framing",
    "medical_evidence_integration",
]

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

        # ------------------
        # Leave-one-out exclusion runs (exclude ONE present factor)
        # Run Turn 2 ONLY if more than one factor is present in baseline.
        # ------------------
        if len(present) > 1:
            for f in present:
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
