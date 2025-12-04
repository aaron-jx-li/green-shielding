import json
import argparse
from collections import Counter
from tqdm import tqdm
from openai import OpenAI


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pairwise medical preference judge (4-way labels) without using questions."
    )

    # Input files
    parser.add_argument(
        "--input_a",
        type=str,
        required=True,
        help="JSON file for model A responses.",
    )
    parser.add_argument(
        "--input_b",
        type=str,
        required=True,
        help="JSON file for model B responses.",
    )
    parser.add_argument(
        "--ref_path",
        type=str,
        required=True,
        help="JSON file for reference (doctor) responses.",
    )

    # Output
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Where to save judge results.",
    )

    # Judge model
    parser.add_argument(
        "--judge_model",
        type=str,
        default="gpt-4.1-mini",
        help="Model used for judging.",
    )

    # Field names (user configurable)
    parser.add_argument(
        "--answer_field_a",
        type=str,
        default="model_response",
        help="Field name for model A's answer in input_a.",
    )
    parser.add_argument(
        "--answer_field_b",
        type=str,
        default="model_response",
        help="Field name for model B's answer in input_b.",
    )
    parser.add_argument(
        "--ref_field",
        type=str,
        default="output",
        help="Field name for the reference (doctor) answer in ref_path.",
    )

    return parser.parse_args()


def build_pairwise_judge_messages(answer_a: str, answer_b: str, reference_answer: str):
    """
    Build messages for pairwise preference judging between model A and B,
    given a reference answer from real doctors.

    NOTE: No question is included; judge only sees the three answers.
    """
    msgs = [
        {
            "role": "developer",
            "content": (
                "You are a strict medical evaluation assistant.\n\n"
                "You will be given:\n"
                "- MODEL_A_ANSWER (model A's response)\n"
                "- MODEL_B_ANSWER (model B's response)\n"
                "- REFERENCE_ANSWER (the reference response from experienced clinicians)\n\n"
                "Your task is to compare ONLY the PRIMARY DIAGNOSIS stated or implied "
                "in MODEL_A_ANSWER and MODEL_B_ANSWER. You must evaluate how closely "
                "each model's primary diagnosis aligns with the diagnosis in REFERENCE_ANSWER.\n\n"
                "VERY IMPORTANT RULES:\n"
                "- Focus **exclusively** on diagnostic correctness and diagnostic specificity.\n"
                "- Ignore style, tone, empathy, length, clarity, bedside manner, disclaimers, "
                "safety language, or any other non-diagnostic elements.\n"
                "- Ignore supportive reasoning, pathophysiology explanations, and narrative details "
                "unless they change the explicit or implied diagnosis.\n"
                "- Do NOT reward verbosity or penalize brief answers.\n"
                "- Do NOT consider treatment recommendations unless they directly indicate the diagnosis.\n"
                "- The only question is: Which model gives a diagnosis that is closer to the "
                "reference diagnosis?\n\n"
                "Definitions:\n"
                "- A_BETTER: Model A's PRIMARY diagnosis (or most likely diagnosis) is clearly closer to, "
                "more accurate than, or matches the reference diagnosis more closely than Model B's.\n"
                "- B_BETTER: Model B's PRIMARY diagnosis is clearly closer to the reference diagnosis "
                "than Model A's.\n"
                "- BOTH_GOOD: Both diagnoses correctly match the reference diagnosis at an appropriate "
                "clinical level of specificity.\n"
                "- BOTH_BAD: Both diagnoses are incorrect or do not meaningfully match the reference diagnosis.\n\n"
                "Comparison guidance:\n"
                "- If one model names the correct diagnosis and the other model gives an incorrect, overly broad, "
                "or unrelated diagnosis, choose the correct one.\n"
                "- If both models give the correct diagnosis (even if phrased differently), choose BOTH_GOOD.\n"
                "- If both models fail to identify the correct diagnosis, choose BOTH_BAD.\n\n"
                "Output format (IMPORTANT):\n"
                "Line 1: EXACTLY ONE label in UPPERCASE with no punctuation:\n"
                "        A_BETTER, B_BETTER, BOTH_GOOD, or BOTH_BAD\n"
                "Line 2: A concise explanation focusing ONLY on the diagnostic correctness comparison."
            ),
        },
        {
            "role": "user",
            "content": (
                f"MODEL_A_ANSWER:\n{(answer_a or '').strip()}\n\n"
                f"MODEL_B_ANSWER:\n{(answer_b or '').strip()}\n\n"
                f"REFERENCE_ANSWER:\n{(reference_answer or '').strip()}\n\n"
                "Considering ONLY the diagnostic content and ignoring style and non-diagnostic details, "
                "select the correct label:\n"
                "- A_BETTER\n"
                "- B_BETTER\n"
                "- BOTH_GOOD\n"
                "- BOTH_BAD\n\n"
                "Respond in exactly two parts:\n"
                "Line 1: A_BETTER, B_BETTER, BOTH_GOOD, or BOTH_BAD\n"
                "Line 2: A concise diagnosis-focused explanation."
            ),
        },
    ]
    return msgs


def normalize_pairwise_label(raw_label: str) -> str:
    """
    Map raw label text to one of:
        A_BETTER, B_BETTER, BOTH_GOOD, BOTH_BAD
    Fallback: BOTH_BAD (strict / conservative).
    """
    if not raw_label:
        return "BOTH_BAD"

    text = raw_label.strip().upper()

    # Ideal case: exact match
    if text.startswith("A_BETTER"):
        return "A_BETTER"
    if text.startswith("B_BETTER"):
        return "B_BETTER"
    if text.startswith("BOTH_GOOD"):
        return "BOTH_GOOD"
    if text.startswith("BOTH_BAD"):
        return "BOTH_BAD"

    # Some looser matches
    if "A BETTER" in text or text == "A":
        return "A_BETTER"
    if "B BETTER" in text or text == "B":
        return "B_BETTER"
    if "BOTH GOOD" in text:
        return "BOTH_GOOD"
    if "BOTH BAD" in text:
        return "BOTH_BAD"

    # Fallback
    return "BOTH_BAD"


def call_judge(client, judge_model: str, msgs):
    """
    Call the judge and return (label, reasoning).
    """
    if "gpt-5" in judge_model:
        resp = client.responses.create(
            model=judge_model,
            input=msgs,
        )
    else:
        resp = client.responses.create(
            model=judge_model,
            input=msgs,
            temperature=0.0,
        )

    raw = (resp.output_text or "").strip()
    if not raw:
        return "BOTH_BAD", ""

    lines = raw.splitlines()
    raw_label = lines[0] if lines else ""
    reasoning = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

    label = normalize_pairwise_label(raw_label)
    return label, reasoning


def get_field(sample: dict, field_name: str):
    """Safe getter: returns None if sample is not a dict or field_name is None."""
    if not isinstance(sample, dict) or field_name is None:
        return None
    return sample.get(field_name)


def main():
    args = parse_args()
    client = OpenAI()

    # Load three datasets
    with open(args.input_a, "r") as f:
        data_a = json.load(f)
    with open(args.input_b, "r") as f:
        data_b = json.load(f)
    with open(args.ref_path, "r") as f:
        data_ref = json.load(f)

    if not (len(data_a) == len(data_b) == len(data_ref)):
        raise ValueError(
            f"Input lengths differ: "
            f"A={len(data_a)}, B={len(data_b)}, REF={len(data_ref)}"
        )

    print(f"Loaded {len(data_a)} triplets (A, B, REF).")

    results = []
    num_total = 0
    label_counts = Counter()

    for idx, (sample_a, sample_b, ref_sample) in enumerate(
        tqdm(zip(data_a, data_b, data_ref), total=len(data_a), desc="Judging")
    ):
        # Answers only (no question)
        answer_a = get_field(sample_a, args.answer_field_a)
        answer_b = get_field(sample_b, args.answer_field_b)
        ref_answer = get_field(ref_sample, args.ref_field)

        # If any critical parts missing, skip this sample
        if not answer_a or not answer_b or not ref_answer:
            continue

        msgs = build_pairwise_judge_messages(
            answer_a=answer_a,
            answer_b=answer_b,
            reference_answer=ref_answer,
        )

        label, reasoning = call_judge(client, args.judge_model, msgs)
        label_counts[label] += 1
        num_total += 1

        results.append(
            {
                "index": idx,
                "label": label,  # A_BETTER, B_BETTER, BOTH_GOOD, BOTH_BAD
                "reasoning": reasoning,
                "model_a_response": answer_a,
                "model_b_response": answer_b,
                "reference_answer": ref_answer,
            }
        )

    # Build summary
    num_a_better = label_counts["A_BETTER"]
    num_b_better = label_counts["B_BETTER"]
    num_both_good = label_counts["BOTH_GOOD"]
    num_both_bad = label_counts["BOTH_BAD"]

    num_decisive = num_a_better + num_b_better
    a_win_rate = num_a_better / num_decisive if num_decisive > 0 else None

    summary = {
        "num_total_judged": num_total,
        "num_a_better": num_a_better,
        "num_b_better": num_b_better,
        "num_both_good": num_both_good,
        "num_both_bad": num_both_bad,
        "a_win_rate_excl_ties": a_win_rate,
        "judge_model": args.judge_model,
        "config": {
            "answer_field_a": args.answer_field_a,
            "answer_field_b": args.answer_field_b,
            "ref_field": args.ref_field,
        },
    }

    output = {
        "summary": summary,
        "per_sample": results,
    }

    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("=== JUDGE SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved results to {args.output_path}")


if __name__ == "__main__":
    main()