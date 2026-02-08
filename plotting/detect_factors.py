import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Output path for the PDF figure, e.g., ./factor_frequency.pdf",
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

    # Full factor names (kept for mapping + optional annotation)
    factor_names = [
        "Mentions specific guess",
        "Contains irrelevant details",
        "Lack of objective data",
        "Lack of symptom history",
        "Mixed / unstructured format",
        "Worried tone",
        "Urgency / severity",
    ]
    percentages = [33.6, 9.0, 73.5, 16.9, 21.5, 26.8, 9.5]

    # Abbreviations shown on y-axis only
    factor_ids = [f"F{i}" for i in range(1, len(factor_names) + 1)]

    if args.print_mapping:
        for fid, name in zip(factor_ids, factor_names):
            print(f"{fid}\t{name}")

    # Category colors
    content_c = "#9ecae1"
    format_c = "#fdd0a2"
    tone_c = "#c7e9c0"

    # Ordered as requested: content (F1-F4), format (F5), tone (F6-F7)
    colors = [
        content_c, content_c, content_c, content_c,  # content
        format_c,                                    # format
        tone_c, tone_c                               # tone
    ]

    fig, ax = plt.subplots(figsize=(5, 6))
    bars = ax.barh(factor_ids, percentages, color=colors)

    # Fonts
    ax.set_xlabel("Frequency (%)", fontsize=14)
    ax.set_xlim(0, 80)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=13)

    # Legend
    legend_handles = [
        Patch(facecolor=content_c, label="Content"),
        Patch(facecolor=format_c, label="Format"),
        Patch(facecolor=tone_c, label="Tone"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
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
    fig.savefig(args.out_path, format="pdf", dpi=500, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
