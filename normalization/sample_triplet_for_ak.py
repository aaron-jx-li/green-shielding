"""
This script is used to sample a triplet of data from the converted and original data for annotation.
"""
import os
import pandas as pd
import json
import random
# ===== Paths =====
green_shielding_dir = "/Users/sancheznicolas/Documents/Research/GreenTeam/green_shield_eval/green-shielding"
convert_data_dir = os.path.join(green_shielding_dir, "results/HCM_ref-9k")
converted_data_path = os.path.join(convert_data_dir, "converted_gpt-4.1-mini.json")
original_data_path = os.path.join(convert_data_dir, "gpt-4.1-mini_full.json")
judged_converted_data_path = os.path.join(convert_data_dir, "converted_gpt-4.1-mini_judged.json")
judged_original_data_path = os.path.join(convert_data_dir, "gpt-4.1-mini_full_judged.json")

dest_path = os.path.join(green_shielding_dir, "webpage_local/annotation_manager/ak_review_round0/normalization/rand_sample_triplet.csv")

# ===== Load Data =====
converted_data = json.load(open(converted_data_path))
original_data = json.load(open(original_data_path))
judged_converted_data = json.load(open(judged_converted_data_path))['per_sample']
judged_original_data = json.load(open(judged_original_data_path))['per_sample']

# get converted_data except for the factors column into a pandas dataframe
converted_data_df = []
factors = []
for item in converted_data:
    factors.append(item["factors"]) # factors is a dictionary so need to store and remove
    item.pop("factors") # remove the factors column from the dictionary
    converted_data_df.append(item) # append the dictionary to the list without the factors column

# convert the list of dictionaries to pandas dataframe
converted_data_df = pd.DataFrame(converted_data_df)
factors_df = pd.DataFrame(factors)

# concat the dataframe with the factors dataframe to have all the converted data in one dataframe
joined_data_converted_df = pd.concat([converted_data_df, factors_df], axis=1) # concat the dataframe with the factors dataframe
fac_cols = factors_df.columns.tolist()

# get the dataframe for the judged converted data
judged_converted_data_df = []
for item in judged_converted_data:
    judged_converted_data_df.append(item)
judged_converted_data_df = pd.DataFrame(judged_converted_data_df)
judged_converted_data_df.set_index("index", inplace=True)

# get the dataframe for the judged original data
judged_original_data_df = []
for item in judged_original_data:
    judged_original_data_df.append(item)
judged_original_data_df = pd.DataFrame(judged_original_data_df)
judged_original_data_df.set_index("index", inplace=True)


# Use pd.merge to join on DataFrame indices and add suffixes for column differentiation
joined_judged_data_df = pd.merge(
    judged_original_data_df,
    judged_converted_data_df,
    left_index=True,
    right_index=True,
    how="inner",
    suffixes=("_og", "_conv")
)

# Correctly join the DataFrames on model_response_conv == model_response
joined_data_all_df = pd.merge(
    joined_judged_data_df,
    joined_data_converted_df[fac_cols + ["model_response","normalized_prompt","original_output"]],
    left_on="model_response_conv",
    right_on="model_response",
    how="inner"
)

# prep for webpage display and annotation
joined_data_all_df.rename(columns={"original_output": "doctor_output"}, inplace=True)
joined_data_all_df = joined_data_all_df.sample(200, random_state=42)
columns_for_webpage_display = ["question_og", "model_response_og",\
                            "model_response_conv", "doctor_output", \
                            "normalized_prompt"]

columns_for_webpage_randomization = ["Response1_label", "Response2_label", "Response3_label", "Response1_value", "Response2_value", "Response3_value"]
candidate_responses = ["model_response_og", "model_response_conv", "doctor_output"]
for col in columns_for_webpage_randomization:
    joined_data_all_df[col] = ""

for row, row_data in joined_data_all_df.iterrows():
    random_shuffle = candidate_responses.copy()
    random.shuffle(random_shuffle)
    joined_data_all_df.loc[row, "Response1_label"] = random_shuffle[0]
    joined_data_all_df.loc[row, "Response1_value"] = row_data[random_shuffle[0]]
    joined_data_all_df.loc[row, "Response2_label"] = random_shuffle[1]
    joined_data_all_df.loc[row, "Response2_value"] = row_data[random_shuffle[1]]
    joined_data_all_df.loc[row, "Response3_label"] = random_shuffle[2]
    joined_data_all_df.loc[row, "Response3_value"] = row_data[random_shuffle[2]]


joined_data_all_df.loc[row, columns_for_webpage_randomization] = row_data[columns_for_webpage_randomization]
columns_for_webpage_annotation = ["all_resp_match", "none_resp_match",\
         "resp1_outlier", "resp2_outlier", "resp3_outlier", \
            "resp1_correct", "resp2_correct", "resp3_correct", "user_annotated"]

for col in columns_for_webpage_annotation:
    joined_data_all_df[col] = -1 # initialize all columns to -1

import pdb; pdb.set_trace()
joined_data_all_df[columns_for_webpage_display + \
    columns_for_webpage_randomization + \
    columns_for_webpage_annotation].to_csv(dest_path, index=False)
print(f"Saved to {dest_path}")