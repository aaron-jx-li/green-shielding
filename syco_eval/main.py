from __future__ import annotations
import argparse
from enums import QFormat, QuestionTone
from runner import evaluate_and_save_csv


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="sycoeval",
        description="Sycophancy-style evaluation for MC, binary, and open-ended questions."
    )
    ap.add_argument("--task", required=True, help="Dataset key (e.g., medxpertqa_diag, truthfulqa_mc1, mmlu_elem_math)")
    ap.add_argument("--model", required=True, help="OpenAI model name (e.g., gpt-4.1-mini)")
    ap.add_argument(
        "--format",
        choices=[QFormat.MC.value, QFormat.BINARY.value, QFormat.OPEN.value],
        required=True,
        help="Question format: MC | binary | open-ended",
    )
    ap.add_argument("--out_csv", required=True, help="Path to save the results CSV")
    ap.add_argument("--shuffle_seed", type=int, default=0, help="Seed for any per-item shuffling")
    ap.add_argument("--max_items", type=int, default=None, help="Max items to evaluate (default: 10)")
    ap.add_argument("--include_raw", action="store_true", help="Include raw_* model outputs in the CSV")
    ap.add_argument(
        "--tone",
        choices=[QuestionTone.ORIGINAL.value, QuestionTone.NEUTRAL.value, QuestionTone.WORRIED.value],
        default=QuestionTone.ORIGINAL.value,
        help="Question tone: original | neutral | worried (default: original)"
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.format == "open-ended" and not args.include_raw:
        print("Warning: You are running open-ended evaluation without logging model outputs.")

    # Print tone information
    if args.tone != QuestionTone.ORIGINAL.value:
        print(f"Using {args.tone} tone for questions")

    evaluate_and_save_csv(
        task=args.task,
        model=args.model,
        format=args.format,
        out_csv=args.out_csv,
        shuffle_seed=args.shuffle_seed,
        max_items=args.max_items,
        include_raw_cols=args.include_raw,
        question_tone=args.tone,
    )


if __name__ == "__main__":
    main()
