#!/usr/bin/env python3
"""
Convert CSV data to JSON format for the annotation webpage.
"""

import pandas as pd
import json
import sys
import os
annotation_filename = "./ak_review_round0/normalization/rand_sample_triplet.csv"
annotation_df = pd.read_csv(annotation_filename)

print("NUM TIMES USER ANNOTATED: ", sum(annotation_df['user_annotated'] == 1))
print("NUM TIMES RESP CORRECT: ", sum(annotation_df['resp1_correct'] == 1), sum(annotation_df['resp2_correct'] == 1), sum(annotation_df['resp3_correct'] == 1) )
print("NUM TIMES RESP INCORRECT: ", sum(annotation_df['resp1_correct'] == 0), sum(annotation_df['resp2_correct'] == 0), sum(annotation_df['resp3_correct'] == 0))
print("NUM TIMES ALL RESP MATCH: ", sum(annotation_df['all_resp_match'] == 1))
print("NUM TIMES NONE RESP MATCH: ", sum(annotation_df['none_resp_match'] == 1))
print("NUM TIMES RESP OUTLIER: ", sum(annotation_df['resp1_outlier'] == 1), sum(annotation_df['resp2_outlier'] == 1), sum(annotation_df['resp3_outlier'] == 1))
import pdb; pdb.set_trace()

annotation_df['user_annotated'] = -1
annotation_columls = ["all_resp_match", "none_resp_match",\
         "resp1_outlier", "resp2_outlier", "resp3_outlier", \
            "resp1_correct", "resp2_correct", "resp3_correct"]
for col in annotation_columls:
    annotation_df[col] = -1
print(f"Reset annotation for {annotation_filename}")
annotation_df.to_csv(annotation_filename, index=False)