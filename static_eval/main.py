from __future__ import annotations
import argparse
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from static_eval.enums import Perturbation
from static_eval.runner import evaluate_and_save_csv


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="sycoeval",
        description="Sycophancy-style evaluation with prompt-level perturbations."
    )
    ap.add_argument("--task", required=True, help="Dataset key: medqa_diag | medxpertqa_diag")
    ap.add_argument("--model", required=True, help="OpenAI model name (e.g., gpt-4.1-mini)")
    ap.add_argument(
        "--perturbation",
        choices=[
            Perturbation.SYCOPHANCY.value,
            Perturbation.FORMAT_MC.value,
            Perturbation.FORMAT_BINARY.value,
        ],
        default=Perturbation.SYCOPHANCY.value,
        help="Prompt-level perturbation to apply",
    )
    ap.add_argument("--out_csv", required=True, help="Path to save the results CSV")
    ap.add_argument("--max_items", type=int, default=None, help="Max items to evaluate (default: 10)")
    ap.add_argument("--include_raw", action="store_true", help="Include raw_* model outputs in the CSV")
    ap.add_argument("--judge_template",type=str, default="with_Q", help="Specify the prompt template for the judge")
    ap.add_argument("--num_generations", type=int, default=5, help="Number of generations per prompt")
    ap.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for generations")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    evaluate_and_save_csv(
        task=args.task,
        model=args.model,
        perturbation=args.perturbation,
        out_csv=args.out_csv,
        max_items=args.max_items,
        include_raw_cols=args.include_raw,
        judge_template=args.judge_template,
        num_generations=args.num_generations,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
