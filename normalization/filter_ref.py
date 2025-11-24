import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Filter samples where reference_diagnosis is not null.")

    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to input JSON file containing a list of samples.")

    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the filtered JSON file.")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load dataset
    with open(args.input_path, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples.")

    # Filter logic: keep samples where reference_diagnosis is not null
    filtered = []
    for sample in data:
        ref = sample.get("reference", {})
        diagnosis = ref.get("reference_diagnosis", None)

        if diagnosis not in (None, "", "null"):
            filtered.append(sample)

    print(f"Selected {len(filtered)} samples with non-null reference_diagnosis.")

    # Save output
    with open(args.output_path, "w") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    print(f"Saved filtered samples to {args.output_path}")


if __name__ == "__main__":
    main()