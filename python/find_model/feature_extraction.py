# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 16:30:44 2021

@author: Datong
"""

from ReadData import getNaxsen
import pandas as pd
import os

from AccFeatureExtraction import get_fn, accFeatureExtraction

dataPath = '../dataset_youth'
goldPath = '../dataset_youth/gold_record.csv'   

samp_rate = 1000
down_spRate = 100
gold = pd.read_csv(goldPath)
gold_save = gold.copy()

is_plot_feature = 0
for i in range(0, len(gold)):
    if is_plot_feature == 0:
        print(f'\r{(i+1)/len(gold)*100:.2f} %', end='\r')
    # get data
    sub = gold.iloc[i]["subNum"]
    testNum = gold.iloc[i]["testNum"]
    source = gold.iloc[i]["source"]
    fn = get_fn(dataPath, source, sub, testNum)
    getData = getNaxsen(fn, samp_rate)
    data = getData()
    if len(data) < 1:
        continue
    
    # feature extraction
    afe = accFeatureExtraction(samp_rate, down_spRate, is_plot_feature)
    feats = afe(data)
    if len(feats) < 40:
        print(f'sub:{sub} test:{testNum} source:{source} can not find feature!')
        continue
    gold_idx = gold.index[(gold.subNum == sub) & (gold.testNum == testNum) &
                          (gold.source == source)].item()
    
    # append features
    for f_idx in range(len(feats)):
        gold_save.at[i, f"feat_{f_idx+1}"] = feats[f_idx]
        
# clear nan and save features
gold_save = gold_save[gold_save["feat_1"].notnull()]
# gold_save.to_csv(os.path.join(dataPath, 'youth_model_new.csv'), index=False)
    
    
    