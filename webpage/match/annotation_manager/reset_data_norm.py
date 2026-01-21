import pandas as pd
import json
import sys
import os
import numpy as np
import pdb; pdb.set_trace()

normalization_data_path = "/Users/sancheznicolas/Documents/Research/GreenTeam/green_shielding3/green-shielding/webpage_match/annotation_manager/ak_review_round0/normalization/normalization_judge_disagreements_joined.csv"
normalization_data = pd.read_csv(normalization_data_path)
import pdb; pdb.set_trace()
normalization_data_disagreements = normalization_data[normalization_data['judge_label_og'] != normalization_data['judge_label_conv']]
# set seed
np.random.seed(42)
normalization_data_disagreements = normalization_data_disagreements.sample(100)
# add a column for user annotation
normalization_data_disagreements['user_annotation'] = -1
normalization_data_disagreements.to_csv(normalization_data_path, index=False)