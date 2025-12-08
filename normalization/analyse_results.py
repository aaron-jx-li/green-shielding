import pandas as pd
import json



converted_data_path = "/Users/sancheznicolas/Documents/Research/GreenTeam/green_shield_eval/green-shielding/results/HCM_ref-9k/converted_gpt-4.1-mini.json"
converted_data = json.load(open(converted_data_path))
original_data_path = "/Users/sancheznicolas/Documents/Research/GreenTeam/green_shield_eval/green-shielding/results/HCM_ref-9k/gpt-4.1-mini_full.json"
original_data = json.load(open(original_data_path))

judged_converted_data_path = "/Users/sancheznicolas/Documents/Research/GreenTeam/green_shield_eval/green-shielding/results/HCM_ref-9k/converted_gpt-4.1-mini_judged.json"
judged_converted_data = json.load(open(judged_converted_data_path))['per_sample']
judged_original_data_path = "/Users/sancheznicolas/Documents/Research/GreenTeam/green_shield_eval/green-shielding/results/HCM_ref-9k/gpt-4.1-mini_full_judged.json"
judged_original_data = json.load(open(judged_original_data_path))['per_sample']

# get converted_data except for the factors column into a pandas dataframe
converted_data_df = []
factors = []
for item in converted_data:
    factors.append(item["factors"])
    item.pop("factors")
    converted_data_df.append(item)
converted_data_df = pd.DataFrame(converted_data_df)
factors_df = pd.DataFrame(factors)
joined_data_converted_df = pd.concat([converted_data_df, factors_df], axis=1)
fac_cols = factors_df.columns.tolist()

judged_converted_data_df = []
for item in judged_converted_data:
    judged_converted_data_df.append(item)
judged_converted_data_df = pd.DataFrame(judged_converted_data_df)
judged_converted_data_df.set_index("index", inplace=True)

judged_original_data_df = []
for item in judged_original_data:
    judged_original_data_df.append(item)
judged_original_data_df = pd.DataFrame(judged_original_data_df)
judged_original_data_df.set_index("index", inplace=True)


# Use pd.merge to join on DataFrame indices and add suffixes for column differentiation
joined_data_df = pd.merge(
    judged_original_data_df,
    judged_converted_data_df,
    left_index=True,
    right_index=True,
    how="inner",
    suffixes=("_og", "_conv")
)

# Correctly join the DataFrames on model_response_conv == model_response
joined_data_all_df = pd.merge(
    joined_data_df,
    joined_data_converted_df[fac_cols + ["model_response","normalized_prompt","original_output"]],
    left_on="model_response_conv",
    right_on="model_response",
    how="inner"
)

joined_data_all_df.to_csv("/Users/sancheznicolas/Documents/Research/GreenTeam/green_shield_eval/green-shielding/webpage_local/annotation_manager/ak_review_round0/normalization/normalization_judge_disagreements_joined.csv", index=False)

import matplotlib.pyplot as plt
import numpy as np
import os

# Directory to save publication-quality figures
fig_save_dir = "/Users/sancheznicolas/Documents/Research/GreenTeam/green_shield_eval/green-shielding/figs/normalization"
os.makedirs(fig_save_dir, exist_ok=True)

# Prepare lists to hold mean accuracy for each factor/setting
original_accuracies = []
converted_accuracies = []

for fac in fac_cols:
    # Only consider rows where this factor is not NaN (shouldn't be, but just in case)
    mask = ~joined_data_all_df[fac].isna()
    # Indices where factor is True
    true_indices = joined_data_all_df[mask][joined_data_all_df[fac] == True].index

    if len(true_indices) == 0:
        # If no examples with this factor, record as nan
        original_accuracies.append(np.nan)
        converted_accuracies.append(np.nan)
        continue

    # For original: judge_label_og; for converted: judge_label_conv
    # We'll treat "CORRECT" as correct, else as not correct
    correct_og = (joined_data_all_df.loc[true_indices, 'judge_label_og'] == "CORRECT").mean()
    correct_conv = (joined_data_all_df.loc[true_indices, 'judge_label_conv'] == "CORRECT").mean()
    original_accuracies.append(correct_og)
    converted_accuracies.append(correct_conv)

# Bar graph setup
x = np.arange(len(fac_cols))
bar_width = 0.35

fig, ax = plt.subplots(figsize=(14, 7), dpi=300, constrained_layout=True)

rects1 = ax.bar(x - bar_width/2, original_accuracies, bar_width, label='Original Prompt', color='#3B82F6', edgecolor='black')
rects2 = ax.bar(x + bar_width/2, converted_accuracies, bar_width, label='Converted Prompt', color='#EC4899', edgecolor='black')

# Axes and labels
ax.set_xlabel('Factor', fontsize=16, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=16, fontweight='bold')
ax.set_title('Model Accuracy by Factor: Original vs Converted Prompts', fontsize=18, fontweight='bold', pad=16)
ax.set_xticks(x)
ax.set_xticklabels([f.replace('_', '\n') for f in fac_cols], rotation=0, fontsize=12, ha='center')
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

# Add data labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        if not np.isnan(height):
            ax.annotate(f'{height:.1%}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0,3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='semibold')

autolabel(rects1)
autolabel(rects2)

# Grid and legend
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend(fontsize=14, frameon=True, loc='best')

plt.tight_layout()
figpath = os.path.join(fig_save_dir, 'accuracy_by_factor_original_vs_converted.png')
fig.savefig(figpath, bbox_inches='tight')
plt.close(fig)




import pandas as pd

# Filtering for cases where: 
# - judge_label_og == "CORRECT"
# - judge_label_conv == "WRONG"
criteria = (joined_data_all_df["judge_label_og"] == "CORRECT") & (joined_data_all_df["judge_label_conv"] == "WRONG")

# Collect all the desired columns in the output
columns_to_show = [
    "question_og",
    "normalized_prompt",
    "judge_label_og",
    "judge_label_conv",
    "model_response_og",
    "model_response_conv",
    "reference_diagnosis_og",
    "original_output"
]

examples_to_show = joined_data_all_df.loc[criteria, columns_to_show]

# Optionally limit to top 50
examples_to_show = examples_to_show.head(50)

# Create a more detailed HTML page
html_parts = [
    "<html><head><meta charset='UTF-8'>",
    "<title>Prompt Examples: Correct → Wrong</title>",
    "<style>",
    "body { font-family: Arial, sans-serif; margin: 2rem; }",
    "h2 { color: #3B82F6; }",
    "table { border-collapse: collapse; width: 100%; }",
    "th, td { border: 1px solid #ccc; padding: 10px; text-align: left; vertical-align: top; }",
    "th { background: #f6f6ff; }",
    ".qog { background: #F0F9FF; }",
    ".conv { background: #FEF2F8; }",
    ".respog { background: #ECFDF5; }",
    ".respconv { background: #FEF9C3; }",
    ".refog { background: #FDE68A; }",
    ".refconv { background: #FECACA; }",
    "</style>",
    "</head><body>",
    "<h1>Examples Where Judge Label Changes from <span style='color:#3B82F6'>Correct</span> to <span style='color:#EC4899'>Wrong</span></h1>",
    "<table>",
    "<tr><th>#</th>"
    "<th>Original Question</th>"
    "<th>Converted Prompt</th>"
    "<th>Original Model Response</th>"
    "<th>Converted Model Response</th>"
    "<th>Original Reference Diagnosis</th>"
    "<th>Original Output</th>"
    "</tr>"
]

for idx, row in examples_to_show.iterrows():
    html_parts.append(
        f"<tr>"
        f"<td>{idx}</td>"
        f"<td class='qog'><strong>judge_label_og:</strong> {row['judge_label_og']}<br><div>{row['question_og']}</div></td>"
        f"<td class='conv'><strong>judge_label_conv:</strong> {row['judge_label_conv']}<br><div>{row['normalized_prompt']}</div></td>"
        f"<td class='respog'><div>{row.get('model_response_og', '')}</div></td>"
        f"<td class='respconv'><div>{row.get('model_response_conv', '')}</div></td>"
        f"<td class='refog'><div>{row.get('reference_diagnosis_og', '')}</div></td>"
        f"<td class='refconv'><div>{row.get('original_output', '')}</div></td>"
        f"</tr>"
    )
html_parts.extend([
    "</table>",
    "<p>Showing up to 50 examples where the model's accuracy dropped after normalization.<br>",
    "Columns included: Original/Converted question & prompt, model responses, and original output.</p>",
    "</body></html>"
])
html_text = "\n".join(html_parts)

# Write to file
output_html_path = os.path.join(fig_save_dir, "examples_correct_to_wrong.html")
with open(output_html_path, "w", encoding="utf-8") as f:
    f.write(html_text)
print(f"HTML summary written to {output_html_path}")


print("Done")


python judge_triplet.py 
  --input_path ../webpage_local/annotation_manager/ak_review_round0/normalization/normalization_judge_disagreements_joined.csv \
  --output_path ../results/HCM_ref-9k/gpt-4.1-mini_full_triplet_judged.json \
  --judge_model gpt-4.1-mini \
  --original_user_input_field question_og \
  --original_output_field original_output \
  --original_model_output_field model_response_og \
  --converted_model_output_field model_response_conv




python judge_triplet.py --input_path ../webpage_local/annotation_manager/ak_review_round0/normalization/normalization_judge_disagreements_joined.csv --output_path ../results/HCM_ref-9k/gpt-4.1-mini_full_triplet_judged.json --judge_model gpt-4.1-mini --original_user_input_field question_og --original_output_field original_output --original_model_output_field model_response_og --converted_model_output_field model_response_conv