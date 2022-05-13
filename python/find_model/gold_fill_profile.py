# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 11:41:19 2021

@author: Datong
"""

import pandas as pd

source = 'NTHU_pilot_median' # GRJH, NTHU_pilot, YCSH_post, NTHU_pilot_median
path_gold = f'../data/{source}/{source}_ft.csv'
path_gold_save = f'../data/{source}/{source}_gold.csv'
path_profile = f'../data/{source}/{source}_profile.csv'

# read file
gold = pd.read_csv(path_gold)
profile = pd.read_csv(path_profile)

# fill profile
gold['cmj'] = 9.8 * ((gold['ft'] / 1000) ** 2) / 8 * 100
for i in range(len(gold)):
    sub = int(gold.iloc[i]['subNum'])
    
    gold.at[i, 'gender'] = profile[profile.subNum == sub]['gender'].item()
    gold.at[i, 'grade'] = profile[profile.subNum == sub]['grade'].item()
    gold.at[i, 'height'] = profile[profile.subNum == sub]['height'].item()
    gold.at[i, 'slj'] = profile[profile.subNum == sub]['slj'].item()
    gold.at[i, 'sprint_50'] = profile[profile.subNum == sub]['sprint_50'].item()
    gold.at[i, 'source'] = source
    
gold = gold[['subNum', 'testNum', 'gender', 'grade', 'cmj',
             'slj', 'sprint_50', 'height', 'source']]
gold.to_csv(path_gold_save, index=False)

