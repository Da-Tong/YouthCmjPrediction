# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 17:40:19 2022

@author: Datong
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt

def Bland_Altman(name_a, name_b, data):
    m1 = data[name_a]
    m2 = data[name_b]
    
    fig, ax = plt.subplots()
    sm.graphics.mean_diff_plot(m1, m2, ax = ax)
    ax.set_title(name_a, fontsize=20)
    
# ==== read data ====
dataPath = "verify_table.csv"
data = pd.read_csv(dataPath)
data = data.groupby(["subNum", "source"]).last()
# data = data.drop(index="YCSH", level=1)
data = data.drop(index="NTHU_pilot", level=1)
# data = data.drop(index="NTHU_pilot_median", level=1)

# ==== Bland-Altman plot ====
Bland_Altman("slj_pred", "slj", data)
Bland_Altman("sprint_50_pred", "sprint_50", data)

# ==== SLJ error ====
data = data[data.slj_pred.notnull()]
slj_mae = round((data["slj"] - data["slj_pred"]).abs().mean(), 2)
slj_mape = round(((data["slj"] - data["slj_pred"]).abs() / data["slj"]).mean() * 100, 2)
data["slj_ape"] = (data["slj_pred"] - data["slj"]).abs() / data["slj"] * 100
data["sprint_50_ape"] = (data["sprint_50_pred"] - data["sprint_50"]).abs() / data["sprint_50"] * 100
slj_ap_10 = data["slj_pred"][data["slj_ape"] < 10].count() / data.slj_pred.count() * 100
slj_ap_30 = data["slj_pred"][data["slj_ape"] < 30].count() / data.slj_pred.count() * 100

print(data["slj"].describe().round(2))
print(data["slj_pred"].describe().round(2))
print(f"slj mae: {slj_mae} cm")
print(f"slj mape: {slj_mape} %")
print(f"slj_ap_10: {slj_ap_10:.2f} %")
print(f"slj_ap_30: {slj_ap_30:.2f} %")


# ==== Sprint-50 error ====
data = data[data.sprint_50.notnull()]
sprint_50_mae = round((data["sprint_50"] - data["sprint_50_pred"]).abs().mean(), 2)
sprint_50_mape = round(((data["sprint_50"] - data["sprint_50_pred"]).abs() / data["sprint_50"]).mean() * 100, 2)
sprint_50_ap_10 = data["sprint_50_pred"][data["sprint_50_ape"] < 10].count() / data.sprint_50_pred.count() * 100
sprint_50_ap_30 = data["sprint_50_pred"][data["sprint_50_ape"] < 30].count() / data.sprint_50_pred.count() * 100

print(data["sprint_50"].describe().round(2))
print(data["sprint_50_pred"].describe().round(2))
print(f"sprint_50 mae {sprint_50_mae} s")
print(f"sprint_50 mape {sprint_50_mape} %")
print(f"sprint_50_ap_10: {sprint_50_ap_10:.2f} %")
print(f"sprint_50_ap_30: {sprint_50_ap_30:.2f} %")




