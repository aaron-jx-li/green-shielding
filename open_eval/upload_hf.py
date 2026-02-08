import argparse
import json
import os
from typing import Any, Dict, List

from datasets import Dataset


DEFAULT_INPUT = "./results/HCM-3k/truth/merged_truth_new.json"


def load_and_filter_records(input_path: str) -> List[Dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    filtered = []
    for item in records:
        ground_truth = item.get("ground_truth_space_majority") or {}
        filtered.append(
            {
                "raw_input": item.get("raw_input"),
                "plausible": ground_truth.get("plausible_set") or [],
                "highly_likely": ground_truth.get("highly_likely_set") or [],
                "cannot_miss": ground_truth.get("cannot_miss_set") or [],
                "original_output": item.get("original_output"),
            }
        )
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload HCM-3k filtered dataset to Hugging Face Hub.")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to merged_truth_new.json",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face dataset repo id, e.g. username/hcm-3k-benchmark",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split name to use for upload",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Upload as a private dataset (default: public)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    records = load_and_filter_records(args.input)
    dataset = Dataset.from_list(records)

    dataset.push_to_hub(
        args.repo_id,
        split=args.split,
        private=args.private,
    )


if __name__ == "__main__":
    main()
