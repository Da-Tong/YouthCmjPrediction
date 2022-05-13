# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 20:36:33 2021

@author: Datong
"""

from YouthJumpPredict import youthJumpPredict
from ReadData import getNaxsen 
from find_model.AccFeatureExtraction import get_fn
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os

def downsample(acc_raw, fz_raw, fz_new): # only pick, no interpolate
    acc = acc_raw[::(fz_raw // fz_new)] 
    return acc[:, 0]

dataPath = 'dataset_youth'
goldPath = 'dataset_youth/gold_record.csv'
cmj_model_path = 'find_model/cmj_model.txt'
slj_model_path = 'find_model/slj_model.txt'
sprint_50_model_path = 'find_model/sprint_50_model.txt'

samp_rate = 1000
down_spRate = 100
is_print_detail = 1

cmj_model = np.loadtxt(cmj_model_path, dtype=str, delimiter=',').astype(float).round(6)
slj_model = np.loadtxt(slj_model_path, dtype=str, delimiter=',').astype(float).round(6)
sprint_50_model = np.loadtxt(sprint_50_model_path, dtype=str, delimiter=',').astype(float).round(6)

gold = pd.read_csv(goldPath)
sub_prev = -1
source_prev = []
for i in range(0, len(gold)):
    i=99
    
    # get data
    sub = gold.iloc[i]["subNum"]
    testNum = gold.iloc[i]["testNum"]
    source = gold.iloc[i]["source"]
    height  = gold.iloc[i]["height"]
    gender = gold.iloc[i]["gender"]
    grade = gold.iloc[i]["grade"]
    cmj = gold.iloc[i]["cmj"]
    slj = gold.iloc[i]["slj"]
    sprint_50 = gold.iloc[i]["sprint_50"]
    history_slj = 200
    history_sprint_50 = 8
    
    print(f"\n==== sub:{sub} test:{testNum} source:{source} ====")
    print("height:", height)
    print("gender:", gender)
    print("grade:", grade)
    print("history_slj:", history_slj)
    print("history_sprint_50:", history_sprint_50)
    
    fn = get_fn(dataPath, source, sub, testNum)
    getData = getNaxsen(fn, samp_rate)
    data = getData()
    acc_x = downsample(data, samp_rate, down_spRate).astype(np.float32)
    
    yjp = youthJumpPredict(acc_x, height, gender, grade,
                           cmj_model, slj_model, sprint_50_model,
                           history_slj, history_sprint_50)
    error_code = yjp.excuteCmjPredict()
        
    #  ==== print algo result ====
    if error_code == 0:
        gold.at[i, "cmj_pred"] = yjp.cmj_pred
        gold.at[i, "slj_pred"] = yjp.slj_pred
        gold.at[i, "sprint_50_pred"] = yjp.sprint_50_pred
        
        if is_print_detail:            
            print("\nfeats: ", end="")
            print("\n", yjp.feat_8.astype(np.float32),
                  "\n", yjp.feat_9.astype(np.float32),
                  "\n", yjp.feat_16,
                  "\n", yjp.feat_33,
                  "\n", yjp.feat_41)
        
            print(f"\ncmj: {cmj:.6f} cm")
            print(f"cmj_pred: {yjp.cmj_pred:.6f} cm")
            
            print(f"\nslj: {slj:.6f} cm")
            print(f"slj_pred: {yjp.slj_pred:.6f} cm")
            
            print(f"\nsprint_50: {sprint_50:.6f} s")
            print(f"sprint_50_pred: {yjp.sprint_50_pred:.6f} s")
    break

fn_out = f"C_code/sub{sub}_test{testNum}_{source}.txt"
print(fn_out)
if len(acc_x) == 1000:
    with open(fn_out, "w") as f:
        for i in range(len(acc_x)):
            f.write(str(acc_x[i]))
            if i == len(acc_x) - 1:
                break
            f.write("\n")        