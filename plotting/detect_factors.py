import json
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# Ordered list of (json_key, display_name, category)
# category: "content" | "format" | "tone"
FACTOR_SPEC = [
    ("mentions_specific_guess",    "Mentions specific guess",      "content"),
    ("contains_irrelevant_details","Contains irrelevant details",  "content"),
    ("lack_of_objective_data",     "Lack of objective data",       "content"),
    ("lack_of_symptom_history",    "Lack of symptom history",      "content"),
    ("unstructured_question_format","Mixed / unstructured format", "format"),
    ("emotional_or_urgent_tone",   "Worried / emotional tone",     "tone"),
    ("first_person_perspective",   "First-person perspective",     "tone"),
]


def compute_frequencies(data_path):
    with open(data_path) as f:
        records = json.load(f)
    total = len(records)
    counts = {spec[0]: 0 for spec in FACTOR_SPEC}
    for rec in records:
        pf = rec.get("paper_factors", {})
        for key in counts:
            if pf.get(key, False):
                counts[key] += 1
    percentages = [counts[key] / total * 100 for key, _, _ in FACTOR_SPEC]
    return percentages, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to a neutralized-prompts JSON file (e.g. remove_all.json).",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Output path for the figure, e.g., ./factor_frequency.pdf",
    )
    parser.add_argument(
        "--print_mapping",
        action="store_true",
        help="Print F# -> full factor name mapping to stdout.",
    )
    parser.add_argument(
        "--annotate_full_names",
        action="store_true",
        help="Annotate bars with full factor names (keeps y-axis as F#).",
    )
    args = parser.parse_args()

    factor_names = [name  for _, name, _     in FACTOR_SPEC]
    categories   = [cat   for _, _,    cat   in FACTOR_SPEC]

    percentages, total = compute_frequencies(args.data_path)
    print(f"Computed frequencies from {total} records in {args.data_path}")

    # Abbreviations shown on y-axis only
    factor_ids = [f"F{i}" for i in range(1, len(factor_names) + 1)]

    if args.print_mapping:
        for fid, name, pct in zip(factor_ids, factor_names, percentages):
            print(f"{fid}\t{name}\t{pct:.1f}%")

    # Category colors
    content_c = "#9ecae1"
    format_c  = "#fdd0a2"
    tone_c    = "#c7e9c0"
    color_map = {"content": content_c, "format": format_c, "tone": tone_c}
    colors = [color_map[cat] for cat in categories]

    fig, ax = plt.subplots(figsize=(5, 6))
    bars = ax.barh(factor_ids, percentages, color=colors)

    # Annotate each bar with its percentage value
    for bar, pct in zip(bars, percentages):
        ax.text(
            bar.get_width() + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center", ha="left", fontsize=10, color="#333333",
        )

    # Fonts
    ax.set_xlabel("Frequency (%)", fontsize=14)
    ax.set_xlim(0, max(percentages) * 1.22)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=13)

    # Legend
    legend_handles = [
        Patch(facecolor=content_c, label="Content"),
        Patch(facecolor=format_c,  label="Format"),
        Patch(facecolor=tone_c,    label="Tone"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        fontsize=12,
    )

    # Optional: annotate full names to the right of bars (useful for slides; usually skip for paper)
    if args.annotate_full_names:
        for bar, name in zip(bars, factor_names):
            x = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2
            ax.text(
                x + 1.0,
                y,
                name,
                va="center",
                ha="left",
                fontsize=11,
            )

    ax.invert_yaxis()
    fig.tight_layout()

    import os
    stem = os.path.splitext(args.out_path)[0]
    fig.savefig(stem + ".pdf", format="pdf", dpi=500, bbox_inches="tight")
    fig.savefig(stem + ".png", format="png", dpi=300, bbox_inches="tight")
    print(f"Saved  {stem}.pdf")
    print(f"Saved  {stem}.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
