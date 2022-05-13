# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 14:42:02 2021

@author: Datong
"""

from ReadData import getNaxsen
from scipy import signal
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from AccFeatureExtraction import get_fn


def get_res_acc(acc, remove_grav):
    if remove_grav:
        return np.sqrt((acc **2).sum(axis=1)) - 9.8
    else:
        return np.sqrt((acc **2).sum(axis=1))

dataPath = '../dataset_youth'
goldPath = '../dataset_youth/gold_record.csv'   
samp_rate = 1000
gold = pd.read_csv(goldPath)
isplot = 1
is_touch_bound_counts = 0
for i in range(0, len(gold)):
    sub = gold.iloc[i]["subNum"]
    testNum = gold.iloc[i]["testNum"]
    source = gold.iloc[i]["source"]
    fn = get_fn(dataPath, source, sub, testNum)
    getData = getNaxsen(fn, samp_rate)
    data = getData()
    if len(data) < 1:
        continue
    
    data = data[::samp_rate // 100]
    samp_rate_filt = 100
    
    acc_x = data[:, 0]
    acc_y = data[:, 1]
    acc_z = data[:, 2]
    acc_res = get_res_acc(data, 0)
    
    sos = signal.butter(3, 2, 'hp',
                            fs=samp_rate_filt, output='sos')
    
    fig, ax = plt.subplots()
    ax.plot(acc_x, label="acc_x")
    ax.plot(acc_res, label="acc_res")
    ax.set_title(f"sub_{sub} test_{testNum}")
    ax.legend()
    
    # acc_x = signal.sosfilt(sos, acc_x)
    # acc_res = signal.sosfilt(sos, acc_res)
    
    # fig, ax = plt.subplots(figsize=(12, 9))
    # ax.plot(acc_x, label="acc_x")
    # ax.plot(acc_res, label="acc_res")
    # ax.legend()
    break
    
    
    
    