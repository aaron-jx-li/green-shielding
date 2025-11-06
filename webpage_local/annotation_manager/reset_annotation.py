#!/usr/bin/env python3
"""
Convert CSV data to JSON format for the annotation webpage.
"""

import pandas as pd
import json
import sys
import os
annotation_filename = "./ak_review_round0/all_questions_metadata.csv"
annotation_df = pd.read_csv(annotation_filename)
annotation_df['expert_dec'] = -1
annotation_df['to_be_seen'] = True
annotation_df.to_csv(annotation_filename, index=False)