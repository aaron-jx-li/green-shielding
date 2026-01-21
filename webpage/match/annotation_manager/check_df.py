#!/usr/bin/env python3
"""
This script is just used to check dataframes easily, no functionality
"""

import pandas as pd
import json
import sys
import os
import csv

import pdb; pdb.set_trace()
trial2 = pd.read_csv("/Users/sancheznicolas/Documents/Research/GreenTeam/green_shielding3/green-shielding/webpage_match/annotation_manager/ak_review_round0/normalization/rand_sample_triplet.csv", quoting=csv.QUOTE_ALL)

trial = pd.read_csv("/Users/sancheznicolas/Documents/Research/GreenTeam/green_shielding2/green-shielding/webpage/annotation_manager/ak_review_round0/all_questions.csv", quoting=csv.QUOTE_ALL)


# "s1",s2,"s3","True","1","-1","True","","36"


# str1,str2,"N-acetylcysteine",False,4,-1,True,False,135
