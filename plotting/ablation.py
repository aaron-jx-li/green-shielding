import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


RAW_COLOR = "#1a1a1a"   # near-black for maximum contrast with all others

COLORS = {
    "Raw (baseline)":        RAW_COLOR,
    "Remove Content":        "#0571b0",  # vivid cobalt blue
    "Remove Format":         "#d95f02",  # vivid orange   (ColorBrewer Dark2)
    "Remove Tone":           "#1b9e77",  # vivid teal     (ColorBrewer Dark2)
    "Remove Content+Format": "#e7298a",  # vivid magenta  (ColorBrewer Dark2)
    "Remove Content+Tone":   "#7570b3",  # vivid purple   (ColorBrewer Dark2)
    "Remove Format+Tone":    "#66a61e",  # vivid lime     (ColorBrewer Dark2)
    "Remove All":            "#d62728",  # vivid crimson
}

FILES = {
    "Raw (baseline)":        "results/HCM-3k/exp_4/eval_raw_1_gpt-4.1-mini.json",
    "Remove Content":        "results/HCM-3k/exp_5/eval_content_1_4.1-mini.json",
    "Remove Format":         "results/HCM-3k/exp_5/eval_format_1_4.1-mini.json",
    "Remove Tone":           "results/HCM-3k/exp_5/eval_tone_1_4.1-mini.json",
    "Remove Content+Format": "results/HCM-3k/exp_5/eval_content_format_1_4.1-mini.json",
    "Remove Content+Tone":   "results/HCM-3k/exp_5/eval_content_tone_1_4.1-mini.json",
    "Remove Format+Tone":    "results/HCM-3k/exp_5/eval_format_tone_1_4.1-mini.json",
    "Remove All":            "results/HCM-3k/exp_5/eval_all_1_4.1-mini.json",
}

METRICS = [
    ("mean_plausibility",            "Plausibility"),
    ("mean_h_coverage",              "H-coverage"),
    ("mean_c_coverage",              "S-coverage"),
    ("mean_normalized_breadth",      "Breadth"),
    ("mean_support_rate",            "Evidence"),
    ("mean_indirect_inference_rate", "Inference"),
    ("uncertainty_rate",             "Uncertainty"),
]


def load_all(base_dir):
    keys = [m[0] for m in METRICS]
    data = {}
    for label, rel in FILES.items():
        with open(os.path.join(base_dir, rel)) as f:
            s = json.load(f)["summary"]
        data[label] = [s[k] for k in keys]
    return data


def save_fig(fig, stem):
    for ext in ("png", "pdf"):
        path = f"{stem}.{ext}"
        fig.savefig(path, format=ext, dpi=300, bbox_inches="tight")
        print(f"Saved  {path}")


def make_radar(
    data_subset,        # dict: label -> list of metric values
    title,
    y_min=0.0,          # inner radius (raise to zoom in)
    y_max=1.0,
    ring_step=0.2,      # spacing between displayed rings
    dominant=None,      # set of labels to draw bold; rest are faded
    legend_title="",
):
    """Return a (fig, ax) radar chart for the given data subset."""
    dominant = dominant or set(data_subset.keys())
    metric_labels = [m[1] for m in METRICS]
    N = len(METRICS)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    ang_c  = angles + angles[:1]

    plt.rcParams.update({
        "font.family":     "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.linewidth":  0.8,
    })

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True),
                           facecolor="white")
    ax.set_facecolor("#f9f9f9")

    # background (faded) conditions first, dominant on top
    bg_labels  = [l for l in data_subset if l not in dominant]
    dom_labels = [l for l in data_subset if l in dominant]

    for label in bg_labels + dom_labels:
        vals = data_subset[label] + data_subset[label][:1]
        is_dom = label in dominant
        lw         = 2.6 if is_dom else 1.2
        alpha_line = 1.0 if is_dom else 0.45
        alpha_fill = 0.13 if is_dom else 0.04
        zorder_l   = 4   if is_dom else 3
        zorder_f   = 3   if is_dom else 2
        ax.plot(ang_c, vals,
                color=COLORS[label], linewidth=lw, alpha=alpha_line,
                linestyle="-", solid_capstyle="round",
                label=label, zorder=zorder_l)
        ax.fill(ang_c, vals,
                color=COLORS[label], alpha=alpha_fill, zorder=zorder_f)

    # ── grid & rings ─────────────────────────────────────────────────────────
    rings = np.arange(
        ring_step * np.ceil(y_min / ring_step),
        y_max + 1e-9,
        ring_step,
    )
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(rings)
    ax.set_yticklabels([f"{int(round(v*100))}%" for v in rings],
                       size=9, color="#555555")
    ax.yaxis.grid(True, color="#999999", linestyle="-", linewidth=0.9, alpha=0.9)
    ax.xaxis.grid(True, color="#bbbbbb", linestyle="-", linewidth=0.7, alpha=0.8)

    # explicit outer boundary ring at y_max
    theta_full = np.linspace(0, 2 * np.pi, 360)
    ax.plot(theta_full, np.full(360, y_max),
            color="#999999", linewidth=1.0, zorder=1)
    ax.spines["polar"].set_visible(False)

    # ── spoke labels ─────────────────────────────────────────────────────────
    ax.set_xticks(angles)
    ax.set_xticklabels(metric_labels, size=13, fontweight="bold", color="#222222")
    ax.tick_params(axis="x", pad=14)
    ax.set_rlabel_position(20)

    # ── title ────────────────────────────────────────────────────────────────
    ax.set_title(title, size=13, fontweight="bold", color="#222222", pad=20)

    # ── legend ───────────────────────────────────────────────────────────────
    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.18, 1.14),
        title=legend_title,
        title_fontsize=10,
        fontsize=10.5,
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        handlelength=2.0,
        labelspacing=0.5,
    )
    if legend_title:
        legend.get_title().set_fontweight("bold")

    fig.tight_layout(pad=2.0)
    return fig, ax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=".",
                        help="Root of the green-shielding project.")
    parser.add_argument("--out_dir", type=str, default="plotting",
                        help="Directory to write output files into.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = load_all(args.base_dir)

    # ── Plot 1: Raw vs Remove All ────────────────────────────────────────────
    subset1 = {k: data[k] for k in ("Raw (baseline)", "Remove All")}
    fig1, _ = make_radar(
        subset1,
        title="Raw vs. Remove All",
        y_min=0.0, y_max=1.0, ring_step=0.2,
        dominant=set(subset1),
    )
    save_fig(fig1, os.path.join(args.out_dir, "ablation_raw_vs_all"))
    plt.close(fig1)

    # ── Plot 2: Raw vs Remove Content (zoomed in) ────────────────────────────
    subset2 = {k: data[k] for k in ("Raw (baseline)", "Remove Content")}
    # Find the floor of the minimum value across both conditions, rounded to 0.05
    all_vals2 = [v for vs in subset2.values() for v in vs]
    y_floor = max(0.0, np.floor(min(all_vals2) / 0.05) * 0.05 - 0.05)
    fig2, _ = make_radar(
        subset2,
        title="Raw vs. Remove Content",
        y_min=y_floor, y_max=1.0, ring_step=0.1,
        dominant=set(subset2),
    )
    save_fig(fig2, os.path.join(args.out_dir, "ablation_raw_vs_content"))
    plt.close(fig2)

    # ── Plot 3: Raw vs the five other variants ───────────────────────────────
    variant_keys = (
        "Remove Format", "Remove Tone",
        "Remove Content+Format", "Remove Content+Tone", "Remove Format+Tone",
    )
    subset3 = {"Raw (baseline)": data["Raw (baseline)"]}
    subset3.update({k: data[k] for k in variant_keys})
    fig3, _ = make_radar(
        subset3,
        title="Raw vs. Other Ablation Variants",
        y_min=0.0, y_max=1.0, ring_step=0.2,
        dominant={"Raw (baseline)"},
    )
    save_fig(fig3, os.path.join(args.out_dir, "ablation_raw_vs_variants"))
    plt.close(fig3)


if __name__ == "__main__":
    main()
