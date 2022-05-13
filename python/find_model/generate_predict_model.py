# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 16:26:31 2021

@author: Datong
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import joblib
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

gold_path = '../dataset_youth/youth_model_new.csv'
gold = pd.read_csv(gold_path)
gold["feat_41"] = gold["feat_33"] ** 2

# ==== cmj model ====
X = gold[[
          "height",
          "feat_8",
          "feat_9",
          "feat_16",
          "feat_33",
          "feat_41",
        ]]
y = gold["cmj"]

X = np.array(X)
y = np.array(y).reshape(-1, )
cmj_model = LinearRegression()
cmj_model.fit(X, y)
y_preds = cmj_model.predict(X)

cmj_a = cmj_model.coef_
cmj_b = cmj_model.intercept_

y_test = (np.sum(X * cmj_a, axis=1) + cmj_b).reshape(-1, )
print(np.mean(np.abs(y_preds - y_test)))


fig, ax = plt.subplots()
ax.plot(y_preds, y, 'o')
ax.set_title(pearsonr(y_test, y))
ax.set_xlabel("cmj")


# ==== slj model ====
gold_max = gold
X = gold[[
          "cmj",
        ]]
y = gold["slj"]

X = np.array(X)
y = np.array(y).reshape(-1, )
slj_model = HuberRegressor()
slj_model.fit(X, y)
y_preds = slj_model.predict(X)

slj_a = slj_model.coef_
slj_b = slj_model.intercept_

y_test = (X * slj_a + slj_b).reshape(-1, )
print(np.mean(np.abs(y_preds - y_test)))


fig, ax = plt.subplots()
ax.plot(y_preds, y, 'o')
ax.set_title(pearsonr(y_test, y))
ax.set_xlabel("slj")

# ==== sprint_50 model ====
gold = gold[gold["sprint_50"].notnull()]
X = gold[[
          "cmj",
          "slj"
        ]]
y = gold["sprint_50"]

X = np.array(X)
y = np.array(y)
sprint_50_model = HuberRegressor()
sprint_50_model.fit(X, y)
y_preds = sprint_50_model.predict(X)

sprint_50_a = sprint_50_model.coef_
sprint_50_b = sprint_50_model.intercept_

y_test = (np.sum(X * sprint_50_a, axis=1) + sprint_50_b).reshape(-1, )
print(np.mean(np.abs(y_preds - y_test)))

fig, ax = plt.subplots()
ax.plot(y_test, y, 'o')
ax.set_title(pearsonr(y_preds, y))
ax.set_xlabel("sprint")

with open("cmj_model.txt", 'w') as f:
    for i in cmj_a:
        f.write(str(i))
        f.write(',')
    f.write(str(cmj_b))

with open("slj_model.txt", 'w') as f:
    for i in slj_a:
        f.write(str(i))
        f.write(',')
    f.write(str(slj_b))
    
with open("sprint_50_model.txt", 'w') as f:
    for i in sprint_50_a:
        f.write(str(i))
        f.write(',')
    f.write(str(sprint_50_b))
    