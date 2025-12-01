#!/usr/bin/env python3
import json
import argparse
from tqdm import tqdm
from openai import OpenAI


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute accuracy using simple CORRECT/WRONG judge."
    )

    parser.add_argument("--input_path", type=str, required=True,
                        help="JSON with model_response and reference_diagnosis.")

    parser.add_argument("--output_path", type=str, required=True,
                        help="Where to save judge results.")

    parser.add_argument("--judge_model", type=str, default="gpt-4.1-mini",
                        help="Model used for judging.")

    return parser.parse_args()


def build_judge_messages(question: str, response: str, truth: str):
    """
    Build messages following the EXACT style the user requested.
    """
    msgs = [
        {
            "role": "developer",
            "content": (
                "You are a strict medical evaluation assistant.\n\n"
                "You will be given:\n"
                "- A QUESTION (the clinical case or query).\n"
                "- A MODEL_ANSWER (the model's response to the question).\n"
                "- A GROUND_TRUTH (the correct diagnosis or answer).\n\n"
                "Your task is to decide whether the PRIMARY diagnosis or answer expressed in "
                "MODEL_ANSWER matches GROUND_TRUTH at a clinically appropriate level of specificity.\n\n"
                "Rules:\n"
                "- If MODEL_ANSWER is exactly the same as GROUND_TRUTH, or is a synonym or paraphrased "
                "statement that a clinician would consider equivalent, output CORRECT.\n"
                "- If MODEL_ANSWER is ambiguous, partially correct, hedged, overly broad, or wrong, output WRONG.\n"
                "- Do NOT give credit for extremely vague or generic categories when GROUND_TRUTH is more specific.\n\n"
                "Output EXACTLY ONE WORD in UPPERCASE with no punctuation: CORRECT or WRONG. Do not explain."
            ),
        },
        {
            "role": "user",
            "content": (
                f"QUESTION:\n{(question or '').strip()}\n\n"
                f"MODEL_ANSWER:\n{(response or '').strip()}\n\n"
                f"GROUND_TRUTH:\n{(truth or '').strip()}\n\n"
                "Given that the correct answer to QUESTION is GROUND_TRUTH, is MODEL_ANSWER correct? "
                "Answer CORRECT or WRONG."
            ),
        },
    ]
    return msgs


def call_judge(client, judge_model: str, msgs) -> str:
    # msgs = build_judge_messages(question, response, truth)

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
            # max_output_tokens=5,
        )

    raw = (resp.output_text or "").strip().upper()
    if raw.startswith("CORRECT"):
        return "CORRECT"
    if raw.startswith("WRONG"):
        return "WRONG"
    # Fallback: treat unexpected output as WRONG
    return "WRONG"


def main():
    args = parse_args()
    client = OpenAI()

    # Load dataset
    with open(args.input_path, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples.")

    results = []
    num_correct = 0
    num_total = 0

    # iterate through samples
    for idx, sample in enumerate(tqdm(data, desc="Judging")):
        if "results/" in args.input_path:
            truth = sample.get("reference_diagnosis", None)
        else:
            reference = sample.get("reference", {})
            truth = reference.get("reference_diagnosis", None)
        model_answer = sample.get("model_response", None)
        question = sample.get("input", "")

        if not truth or not model_answer:
            continue  # cannot judge

        num_total += 1

        messages = build_judge_messages(question, model_answer, truth)
        judge_label = call_judge(client, args.judge_model, messages)

        is_correct = (judge_label == "CORRECT")
        if is_correct:
            num_correct += 1

        results.append(
            {
                "index": idx,
                "judge_label": judge_label,
                "is_correct": is_correct,
                "question": question,
                "model_response": model_answer,
                "reference_diagnosis": truth,
            }
        )

    accuracy = num_correct / num_total if num_total > 0 else None

    summary = {
        "num_total_judged": num_total,
        "num_correct": num_correct,
        "accuracy": accuracy,
        "judge_model": args.judge_model,
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
