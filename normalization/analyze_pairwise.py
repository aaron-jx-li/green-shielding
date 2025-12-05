import json
import pandas as pd
import os

# ==== Paths ====
judge1_path = "./results/HCM_ref-9k/gpt-4.1_pairwise_judge.json"  # e.g. gpt-4.1
judge2_path = "./results/HCM_ref-9k/gpt-4.1-mini_pairwise_judge_gpt-5-mini.json"  # e.g. gpt-4.1-mini
fig_save_dir = "./figs/normalization/"
os.makedirs(fig_save_dir, exist_ok=True)

# ==== Load JSONs ====
j1 = json.load(open(judge1_path))
j2 = json.load(open(judge2_path))

per1 = j1["per_sample"]
per2 = j2["per_sample"]

# Turn into DataFrames indexed by 'index'
df1 = pd.DataFrame(per1).set_index("index")
df2 = pd.DataFrame(per2).set_index("index")

# Merge on index, add suffixes to distinguish judges
merged = df1.merge(df2, left_index=True, right_index=True,
                   how="inner", suffixes=("_j1", "_j2"))

# ==== Find disagreements ====
disagree_mask = merged["label_j1"] != merged["label_j2"]
disagreements = merged.loc[disagree_mask].copy()

# Optional: limit to first 50
disagreements = disagreements.head(50)

# ==== Build HTML ====
html_parts = [
    "<html><head><meta charset='UTF-8'>",
    "<title>Judge Disagreement Examples</title>",
    "<style>",
    "body { font-family: Arial, sans-serif; margin: 2rem; }",
    "h1 { color: #1D4ED8; }",
    "table { border-collapse: collapse; width: 100%; }",
    "th, td { border: 1px solid #ccc; padding: 10px; text-align: left; vertical-align: top; }",
    "th { background: #f6f6ff; }",
    ".judge1 { background: #DBEAFE; }",
    ".judge2 { background: #FEE2E2; }",
    ".respA  { background: #ECFDF5; }",
    ".respB  { background: #FEF9C3; }",
    ".ref    { background: #FDE68A; }",
    "</style>",
    "</head><body>",
    "<h1>Examples Where Two Judges Disagree</h1>",
    "<p>",
    "<strong>label_j1</strong> / <strong>reasoning_j1</strong> = gpt-4.1 JSON<br>",
    "<strong>label_j2</strong> / <strong>reasoning_j2</strong> = gpt-5-mini JSON",
    "</p>",
    "<table>",
    "<tr>",
    "<th>#</th>",
    "<th>Judge 1 Decision & Reasoning</th>",
    "<th>Judge 2 Decision & Reasoning</th>",
    "<th>Model A Response</th>",
    "<th>Model B Response</th>",
    "<th>Reference Answer</th>",
    "</tr>",
]

for idx, row in disagreements.iterrows():
    # assume model_a_response / model_b_response / reference_answer
    # are identical across judge files; take from _j1
    model_a = row.get("model_a_response_j1", "")
    model_b = row.get("model_b_response_j1", "")
    ref    = row.get("reference_answer_j1", "")

    html_parts.append(
        "<tr>"
        f"<td>{idx}</td>"
        f"<td class='judge1'><strong>label_j1:</strong> {row['label_j1']}<br>"
        f"<em>{row.get('reasoning_j1', '')}</em></td>"
        f"<td class='judge2'><strong>label_j2:</strong> {row['label_j2']}<br>"
        f"<em>{row.get('reasoning_j2', '')}</em></td>"
        f"<td class='respA'>{model_a}</td>"
        f"<td class='respB'>{model_b}</td>"
        f"<td class='ref'>{ref}</td>"
        "</tr>"
    )

html_parts.extend([
    "</table>",
    "<p>Showing up to 50 examples where the two judges output different labels.</p>",
    "</body></html>",
])

html_text = "\n".join(html_parts)

output_html_path = os.path.join(fig_save_dir, "judge_disagreements_4.1_5.html")
with open(output_html_path, "w", encoding="utf-8") as f:
    f.write(html_text)

print(f"HTML summary written to {output_html_path}")
