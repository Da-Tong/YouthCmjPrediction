# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 10:14:48 2021

@author: Datong
"""

import numpy as np
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR



def MAPE(pred, true):
    return np.mean(np.abs(pred - true) / np.abs(true)) * 100

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def run_model(X, y):
    X = np.array(X)
    y = np.array(y)
    
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    
    names = [
        "LinearRegression",
              "Poly_2 LinearRegression", 
              "Poly_3 LinearRegression",
               # "linear SVR",
              # "rbf",
              # "rbf",
              # "sigmoid",
              # "precomputed",
                # "linear standar SVR",
               # "rbf standar",
               # "sigmoid standar",
              # "RandomForestRegressor",
              # "RandomForestRegressor",
                "GradientBoostingRegressor 20, 4",
                "GradientBoostingRegressor 30, 3",
                "GradientBoostingRegressor 100, 2",
               # "GradientBoostingRegressor 50",
               # "GradientBoostingRegressor 60",
               # "GradientBoostingRegressor 70",
               # "GradientBoostingRegressor 80",
               # "GradientBoostingRegressor 90",
             ]
    
    # iter_num = 1000
    regressors = [
        LinearRegression(),
        make_pipeline(PolynomialFeatures(2),
                                LinearRegression()),
        make_pipeline(PolynomialFeatures(3),
                                LinearRegression()),
        # SVR(kernel='linear', epsilon=0.1, C=2),
        # SVR(kernel='rbf', epsilon=0.001, C=2),
        # SVR(kernel='rbf', epsilon=0.005, C=0.5),
        # SVR(kernel='sigmoid', epsilon=0.01, C=0.4),
        # SVR(kernel='precomputed', epsilon=0.2),
        
        # make_pipeline(StandardScaler(),
        #               SVR(kernel='linear', epsilon=0.1, C=2)),
        # make_pipeline(StandardScaler(),
        #               SVR(kernel='rbf', epsilon=0.001, C=2)),
        # make_pipeline(StandardScaler(),
        #               SVR(kernel='sigmoid', epsilon=0.001, C=2)),
        # make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2)),
        # make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2)),
        # make_pipeline(PolynomialFeatures(5),
        #                         HuberRegressor(max_iter=iter_num)),
        # make_pipeline(PolynomialFeatures(6),
        #                         HuberRegressor(max_iter=iter_num)),
        # RandomForestRegressor(n_estimators=30,
        #                       max_depth=None,
        #                       random_state=None),
        GradientBoostingRegressor(n_estimators=20,  max_depth=4),
        GradientBoostingRegressor(n_estimators=30,  max_depth=3),
        GradientBoostingRegressor(n_estimators=100,  max_depth=2),
        # GradientBoostingRegressor(n_estimators=50,  max_depth=2),
        # GradientBoostingRegressor(n_estimators=60,  max_depth=2),
        # GradientBoostingRegressor(n_estimators=70,  max_depth=2),
        # GradientBoostingRegressor(n_estimators=80,  max_depth=2),
        # GradientBoostingRegressor(n_estimators=90,  max_depth=2),
        ]

    regression_mape = {}
    regression_mae = {}
    regression_std = {}
    regression_rval = {}
    
    fig, axs = plt.subplots(4, 2)
    plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=1)
    for idx, clf in enumerate(regressors):
        print("\n====regressors:", regressors[idx], "=====\n")
        temp_acc = []
        temp_acc_mae = []
        test_x = []
        test_y = []

        loo = LeaveOneOut()
        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # model training
            if idx < 0:
                scale = StandardScaler()
                scale_X = scale.fit_transform(X_train)
                clf.fit(scale_X, y_train)
                y_pred = clf.predict(scale.transform(X_test))
            else:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

            temp_acc.append(MAPE(y_pred, y_test))
            temp_acc_mae.append(MAE(y_pred, y_test))
            test_x.append(y_pred)
            test_y.append(y_test)
        
        regression_mape[names[idx]] = round(np.mean(temp_acc) * 100, 2)
        regression_mae[names[idx]] = round(np.mean(temp_acc_mae) * 100, 2)
        regression_std[names[idx]] = np.std(temp_acc)
        
        # correlation of model
        test_x = np.array(test_x).reshape(-1, )
        test_y = np.array(test_y).reshape(-1, )
        y_pred = clf.predict(X)
        col = 0
        row = idx
        if row >= len(regressors) // 2:
            row -= len(regressors) // 2
            col = 1

        axs[row, col].plot(test_x, test_y, 'o')
        axs[row, col].set_title(f'''
                                {names[idx]} test
                                r: {pearsonr(test_x, test_y)[0]:.2f}
                                mape: {MAPE(test_x, test_y):.2f}
                                MAE: {MAE(test_x, test_y):.4f}''')
        regression_rval[names[idx]] = round(pearsonr(test_x, test_y)[0], 2)
        
    return regression_mape, regression_mae, regression_rval

def create_score_lower_upper(min_score, max_score, range_fix=15):
    '''
    input predicted score from model(slj/50m) then gave predict score range

    Parameters
    ----------
    min_score : minimal model predicted score (int)
        
    max_score : maximal model predicted score (int)
        
    range_fix : specific score range, default 15 points (int)

    Returns
    -------
    lower_bound: predicted score lower bound (int)
        
    upper_bound: predicted score upper bound (int)

    '''
    
    if max_score > (100 - range_fix):
        return 100 - range_fix, 100
    
    if min_score < range_fix:
        return 0, range_fix
    
    range_real = max_score - min_score
    if range_fix == range_real:
        return min_score, max_score
    elif range_fix < range_real:
        lower_bound = min_score + int(((range_real - range_fix) + 1) / 2)
        upper_bound = lower_bound + 15
    elif range_fix > range_real:
        lower_bound = min_score - int(((range_real - range_fix) - 1) / 2)
        upper_bound = lower_bound + 15
    
    return lower_bound, upper_bound

if __name__ == '__main__':
    import pandas as pd
    from PlotTool import Correlaiton
    from ChinaScoreEstimation import chinaScoreEstimation
    
    gold_path = '../dataset_youth/youth_model_new.csv'
    gold_path = '../gold_last.csv'
    gold = pd.read_csv(gold_path)
    # gold["feat_41"] = gold["feat_33"] ** 2
    # gold_test = gold
    
    
#%% get predicted cmj
    # gold_test = gold[gold["cmj"] < 50]
    # gold_test = gold[gold["slj"] < 270]
    gold_test = gold
    X = gold_test[[
                "height",
                # "feat_1",
                # "feat_4",
                # "feat_6",
                # "feat_8",
                # "feat_9",
                "feat_16",
                # "feat_19",
                "feat_33",
                "feat_41",
              ]]
    y = gold_test["cmj"]
    X = np.array(X)
    y = np.array(y)
    loo = LeaveOneOut()
    rlf = LinearRegression()
    # rlf = GradientBoostingRegressor(n_estimators=100, max_depth=2)
    # rlf = SVR(kernel='linear', epsilon=0.01, C=0.1)
    # rlf = make_pipeline(StandardScaler(),
    #                     LinearRegression())
    y_preds = []
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        rlf.fit(X_train, y_train)
        y_pred = rlf.predict(X_test)
        y_preds.append(y_pred[0])
    
    y_preds = np.array(y_preds).reshape(-1, 1)
    
    gold_test = gold_test.assign(cmj_pred = y_preds)
    # gold_test = gold_test[gold_test["cmj"] < 50]
    plt_corr = Correlaiton()
    plt_corr(gold_test["cmj_pred"], gold_test["cmj"])
    
    gold_mean = gold_test.groupby(['subNum', 'source']).mean()
    gold_max =  gold_test.groupby(['subNum', 'source']).max()
    plt_corr(gold_max["cmj"], gold_max["slj"])
#%% find feature of cmj
    corr = gold_test.corr()
    corr_performance = corr[corr["cmj"].abs() > 0.37][["cmj"]]
    
    fig, ax1 = plt.subplots()
    sns.heatmap(corr_performance, annot=True, cmap="YlGnBu")
    
    # plt_corr = Correlaiton()
    # plt_corr(gold_test["feat_33"] ** 2 / 100, gold_test["cmj"])
    
#%% find feature of slj
    # corr_mean = gold_mean.corr()
    # corr_max = gold_max.corr()
    # corr_performance_max = corr_max[corr_max["slj"].abs() > 0.5][["slj", "sprint_50"]]
    # corr_performance_mean = corr_mean[corr_mean["slj"].abs() > 0.5][["slj", "sprint_50"]]
    
    # fig, ax1 = plt.subplots()
    # sns.heatmap(corr_performance_max, annot=True, cmap="YlGnBu")
    # plt.title("max")
    # fig, ax1 = plt.subplots()
    # sns.heatmap(corr_performance_mean, annot=True, cmap="YlGnBu")
    # plt.title("mean")
    
    # plt_corr = Correlaiton()
    # # plt_corr(gold_max["cmj_pred"], gold_max["slj"])
    # # plt_corr(gold_max["slj_pred"], gold_max["slj"])
    
    #%% find optimize model of cmj prediction
    gold_test = gold.groupby(["subNum", "source"]).last().reset_index()
    gold_test = gold_test[gold_test["cmj_pred"].notnull()]
    gold_test = gold_test[gold_test["cmj"].notnull()]
    gold_test = gold_test[gold_test["sprint_50"].notnull()]
    
    X = gold_test[[
                "cmj_pred",
                "slj_pred",
                # "feat_4",
                # "feat_6",
                # "feat_8",
                # "feat_9",
                # "feat_16",
                # "feat_19",
                # "feat_33",
                # "feat_41",
              ]]
    y = gold_test["sprint_50"]
    (regression_mape,
      regression_mae,
      regression_rval) = run_model(X, y)
    
    plt_corr = Correlaiton()
    # plt_corr(gold_test["cmj"], gold_test["cmj_pred"])
    # plt_corr(gold_test["cmj_pred"], gold_test["slj"])
    plt_corr(gold_test["slj_pred"], gold_test["slj"])
    plt_corr(gold_test["cmj_pred"], gold_test["sprint_50"])
    plt_corr(gold_test["sprint_50_pred"], gold_test["sprint_50"])
    
# %% use two model predict slj
    sep_bound = 32
    X = gold_max[gold_max.cmj_pred < sep_bound][["cmj"]] 
    y = gold_max[gold_max.cmj_pred < sep_bound]["slj"]
    X = np.array(X)
    y = np.array(y)
    rlf_short = make_pipeline(PolynomialFeatures(1),
                                HuberRegressor(max_iter=1000))
    # rlf_short = HuberRegressor()
    rlf_short.fit(X, y)
    X_test = gold_test[gold_test.cmj_pred < sep_bound][["cmj_pred"]]
    X_test = np.array(X_test)
    gold_test.at[gold_test.cmj_pred < sep_bound, "slj_pred"] = rlf_short.predict(X_test)    
    
    X = gold_max[gold_max.cmj_pred >= sep_bound][["cmj"]]
    y = gold_max[gold_max.cmj_pred >= sep_bound]["slj"]
    X = np.array(X)
    y = np.array(y)
    rlf_high = make_pipeline(PolynomialFeatures(1),
                                HuberRegressor(max_iter=1000))
    y_high = rlf_high.fit(X, y)
    X_test = gold_test[gold_test.cmj_pred >= sep_bound][["cmj_pred"]]
    X_test = np.array(X_test)
    gold_test.at[gold_test.cmj_pred >= sep_bound, "slj_pred"] = rlf_high.predict(X_test)
    gold_max = gold_test.groupby(['subNum', 'source']).max().reset_index()
    
    # plt_corr(gold_max[gold_max.cmj < sep_bound]["cmj"], gold_max[gold_max.cmj < sep_bound]["slj"])
    # plt_corr(gold_max[gold_max.cmj >= sep_bound]["cmj"], gold_max[gold_max.cmj >= sep_bound]["slj"])
    
    #%% use one max jump to get slj score bound
    cse = chinaScoreEstimation()
    half_score_range = 8
    for i in range(len(gold_max)):
        gender = gold_max.iloc[i]['gender']
        grade = int(gold_max.iloc[i]['grade'])
        
        slj = gold_max.iloc[i]['slj']
        gold_max.at[i, "slj_score"] = cse(gender, grade, 'slj', slj)
        
        slj_pred = gold_max.iloc[i]['slj_pred']
        gold_max.at[i, "slj_score_pred"] = cse(gender, grade, 'slj', slj_pred)
        
        gold_max.at[i, "slj_upper_bound"] = cse(gender, grade,
                                                    'slj', slj_pred) + half_score_range
        gold_max.at[i, "slj_lower_bound"] = cse(gender, grade,
                                                    'slj', slj_pred) - half_score_range
        
    fig, ax = plt.subplots(figsize=(13, 10))
    correct_num = 0
    higher_num = 0
    lower_num = 0
    for i in range(len(gold_max)):
        sub = i + 1
        score = gold_max.iloc[i]["slj_score"]
        lb = gold_max.iloc[i]["slj_lower_bound"].item()
        ub = gold_max.iloc[i]["slj_upper_bound"].item()
        half_score_range = (ub - lb) / 2
        
        color_index = score - (lb + ub) / 2
        if color_index <= half_score_range and color_index >= -half_score_range:
            color = 'green'
            correct_num += 1
        
        elif color_index > half_score_range:
            color = 'orange'
            lower_num += 1
            
        elif color_index < -half_score_range:
            color = 'crimson'
            higher_num += 1
            
        # plot predict score range vs. real score
        ln1 = ax.hlines(y=sub, xmin=lb, xmax=ub,
                  linewidth=10, color=color, alpha = 1/4, label="predict")
        ln2 = ax.hlines(y=sub, xmin=score-0.5, xmax=score+0.5,
                  linewidth=10, color=color, alpha = 0.7, label="real")
        ax.hlines(y=sub, xmin=score, xmax=(lb + ub) / 2, linestyles='dashed',
                  linewidth=1, color=color, alpha = 0.7)
        
    ax.set_xlabel("SLJ Score", fontsize=22)
    ax.set_ylabel("Subject", fontsize=22)
    print("==== SLJ ====")
    print("-- max once --")
    print(f"correct: {correct_num}")
    print(f"lower_num: {lower_num}")
    print(f"higher_num: {higher_num}")
    
    #%% use different jump to iter slj score bound
    a = gold_test[["subNum", "source", "grade", "gender", "cmj", "cmj_pred",
                  "slj", "slj_pred"]]
    a = a.reset_index()
    
    for i in range(len(a)):
        gender = a.iloc[i]['gender']
        grade = int(a.iloc[i]['grade'])
        
        slj = a.iloc[i]['slj']
        a.at[i, "slj_score"] = cse(gender, grade, 'slj', slj)
        
        slj_pred = a.iloc[i]['slj_pred']
        a.at[i, "slj_score_pred"] = cse(gender, grade, 'slj', slj_pred)
        
    a_max =  a.groupby(['subNum', 'source']).max().reset_index()
    a_min =  a.groupby(['subNum', 'source']).min().reset_index()
    for i in range(len(a_max)):
        min_score = a_min.iloc[i]["slj_score_pred"].item()
        max_score = a_max.iloc[i]["slj_score_pred"].item()
        lb, ub = create_score_lower_upper(min_score, max_score)
        a_max.at[i, "slj_lower_bound"] = lb
        a_max.at[i, "slj_upper_bound"] = ub
    
    fig, ax = plt.subplots(figsize=(13, 10))
    correct_num = 0
    higher_num = 0
    lower_num = 0
    for i in range(len(a_max)):
        sub = i + 1
        score = a_max.iloc[i]["slj_score"]
        lb = a_max.iloc[i]["slj_lower_bound"].item()
        ub = a_max.iloc[i]["slj_upper_bound"].item()
        
        color_index = score - (lb + ub) / 2
        if color_index < 8 and color_index > -8:
            color = 'green'
            correct_num += 1
        
        elif color_index > 8:
            color = 'orange'
            lower_num += 1
            
        elif color_index < -8:
            color = 'crimson'
            higher_num += 1
            
        # plot predict score range vs. real score
        ln1 = ax.hlines(y=sub, xmin=lb, xmax=ub,
                  linewidth=10, color=color, alpha = 1/4, label="predict")
        ln2 = ax.hlines(y=sub, xmin=score-0.5, xmax=score+0.5,
                  linewidth=10, color=color, alpha = 0.7, label="real")
        ax.hlines(y=sub, xmin=score, xmax=(lb + ub) / 2, linestyles='dashed',
                  linewidth=1, color=color, alpha = 0.7)
        
    ax.set_xlabel("SLJ Score", fontsize=22)
    ax.set_ylabel("Subject", fontsize=22)
    
    print("-- iter --")
    print(f"correct: {correct_num}")
    print(f"lower_num: {lower_num}")
    print(f"higher_num: {higher_num}")
    
    # print score error
    score_slj_mae_total = MAE(a_max["slj_score_pred"], a_max["slj_score"])
    score_slj_mae = MAE(a_max[a_max.slj_score > 0]["slj_score_pred"], a_max[a_max.slj_score > 0]["slj_score"])
    score_slj_mape = MAPE(a_max[a_max.slj_score > 0]["slj_score_pred"], a_max[a_max.slj_score > 0]["slj_score"])
    slj_mae = MAE(a_max["slj_pred"], a_max["slj"])
    slj_mape = MAPE(a_max["slj_pred"], a_max["slj"])
    
    print()
    print(f"score mae: {score_slj_mae_total:.2f} points")
    print(f"score mape: {score_slj_mape:.2f} %")
    print(f"cm mae: {slj_mae:.2f} cm\ncm mape: {slj_mape:.2f} %")
    
    plt_corr(a_max["slj_score_pred"], a_max["slj_score"])
    
    #%%  find feature of 50m
    # gold_mean = gold_mean[gold_mean["sprint_50"].notnull()]
    # gold_max = gold_max[gold_max["sprint_50"].notnull()]
    # corr_mean = gold_mean.corr()
    # corr_max = gold_max.corr()
    # corr_performance_max = corr_max[corr_max["sprint_50"].abs() > 0.5][["slj", "sprint_50"]]
    # corr_performance_mean = corr_mean[corr_mean["sprint_50"].abs() > 0.5][["slj", "sprint_50"]]
    
    # fig, ax1 = plt.subplots()
    # sns.heatmap(corr_performance_max, annot=True, cmap="YlGnBu")
    # plt.title("max")
    # fig, ax1 = plt.subplots()
    # sns.heatmap(corr_performance_mean, annot=True, cmap="YlGnBu")
    # plt.title("mean")
    
    # plt_corr = Correlaiton()
    # plt_corr(gold_max["cmj"], gold_max["slj"])
    # plt_corr(gold_max["slj_pred"], gold_max["slj"])
    #%% find best model for sprint_50
    
    # X = gold_max[[
    #             "cmj",
    #             # "slj_pred",
    #             "slj",
    #           ]]
    # y = gold_max["sprint_50"]
    # (regression_mape,
    #   regression_mae,
    #   regression_rval) = run_model(X, y)
    #%% predict sprint_50
    gold_test = gold_test[gold_test["sprint_50"].notnull()]
    gold_max = gold_test.groupby(['subNum', 'source']).max().reset_index()
    
    plt_corr(gold_max["cmj"], gold_max["sprint_50"])
    plt_corr(gold_max["slj"], gold_max["sprint_50"])
    
    X = gold_max[[
                "cmj",
                "slj",
              ]]
    y = gold_max["sprint_50"]
    X = np.array(X)
    y = np.array(y)
    rlf = HuberRegressor()
    rlf.fit(X, y)
    
    x_test = gold_test[["cmj_pred", "slj_pred"]]
    x_test = np.array(x_test)
    gold_test.at[:, "sprint_50_pred"] = rlf.predict(x_test)
    gold_max = gold_test.groupby(['subNum', 'source']).max().reset_index()
    gold_min = gold_test.groupby(['subNum', 'source']).min().reset_index()
    # gold_max["sprint_50"] = gold_test.groupby(['subNum', 'source']).mean().reset_index()["sprint_50"]
    
    #%% use one max jump to get sprint_50 score bound
    cse = chinaScoreEstimation()
    half_score_range = 8
    for i in range(len(gold_min)):
        gender = gold_min.iloc[i]['gender']
        grade = int(gold_min.iloc[i]['grade'])
        
        sprint_50 = gold_min.iloc[i]['sprint_50']
        gold_min.at[i, "sprint_50_score"] = cse(gender, grade,
                                                'sprint', sprint_50)
        
        sprint_50_pred = gold_min.iloc[i]['sprint_50_pred']
        gold_min.at[i, "sprint_50_score_pred"] = cse(gender, grade,
                                                     'sprint', sprint_50_pred)
        
        gold_min.at[i, "sprint_50_upper_bound"] = (cse(gender, grade,
                                                       'sprint', sprint_50_pred) +
                                                   half_score_range)
        gold_min.at[i, "sprint_50_lower_bound"] = (cse(gender, grade,
                                                       'sprint', sprint_50_pred) -
                                                   half_score_range)
        
    fig, ax = plt.subplots(figsize=(13, 10))
    correct_num = 0
    higher_num = 0
    lower_num = 0
    for i in range(len(gold_min)):
        sub = i + 1
        score = gold_min.iloc[i]["sprint_50_score"]
        lb = gold_min.iloc[i]["sprint_50_lower_bound"].item()
        ub = gold_min.iloc[i]["sprint_50_upper_bound"].item()
        half_score_range = (ub - lb) / 2
        
        color_index = score - (lb + ub) / 2
        if color_index <= half_score_range and color_index >= -half_score_range:
            color = 'green'
            correct_num += 1
        
        elif color_index > half_score_range:
            color = 'orange'
            lower_num += 1
            
        elif color_index < -half_score_range:
            color = 'crimson'
            higher_num += 1
            
        # plot predict score range vs. real score
        ln1 = ax.hlines(y=sub, xmin=lb, xmax=ub,
                  linewidth=10, color=color, alpha = 1/4, label="predict")
        ln2 = ax.hlines(y=sub, xmin=score-0.5, xmax=score+0.5,
                  linewidth=10, color=color, alpha = 0.7, label="real")
        ax.hlines(y=sub, xmin=score, xmax=(lb + ub) / 2, linestyles='dashed',
                  linewidth=1, color=color, alpha = 0.7)
        
    ax.set_xlabel("Sprint_50 Score", fontsize=22)
    ax.set_ylabel("Subject", fontsize=22)
    print("\n==== Spinrt_50 ====")
    print("-- max once --")
    print(f"correct: {correct_num}")
    print(f"lower_num: {lower_num}")
    print(f"higher_num: {higher_num}")
    
    #%% use different jump to iter sprint_50 score bound
    a = gold_test[["subNum", "source", "grade", "gender", "cmj", "cmj_pred",
                  "sprint_50", "sprint_50_pred"]]
    a = a.reset_index()
    
    for i in range(len(a)):
        gender = a.iloc[i]['gender']
        grade = int(a.iloc[i]['grade'])
        
        sprint_50 = a.iloc[i]['sprint_50']
        a.at[i, "sprint_50_score"] = cse(gender, grade, 'sprint', sprint_50)
        
        sprint_50_pred = a.iloc[i]['sprint_50_pred']
        a.at[i, "sprint_50_score_pred"] = cse(gender, grade, 'sprint', sprint_50_pred)
        
    a_max =  a.groupby(['subNum', 'source']).max().reset_index()
    a_min =  a.groupby(['subNum', 'source']).min().reset_index()
    
    # for i in range(len(a_max)):
    #     min_score = a_min.iloc[i]["sprint_50_score_pred"].item()
    #     max_score = a_max.iloc[i]["sprint_50_score_pred"].item()
    #     lb, ub = create_score_lower_upper(min_score, max_score)
        
    #     a_max.at[i, "sprint_50_lower_bound"] = lb
    #     a_max.at[i, "sprint_50_upper_bound"] = ub
    
    # fig, ax = plt.subplots(figsize=(13, 10))
    # correct_num = 0
    # higher_num = 0
    # lower_num = 0
    # for i in range(len(a_max)):
    #     sub = i + 1
    #     score = a_max.iloc[i]["sprint_50_score"]
    #     lb = a_max.iloc[i]["sprint_50_lower_bound"].item()
    #     ub = a_max.iloc[i]["sprint_50_upper_bound"].item()
        
    #     color_index = score - (lb + ub) / 2
    #     if color_index < 8 and color_index > -8:
    #         color = 'green'
    #         correct_num += 1
        
    #     elif color_index > 8:
    #         color = 'orange'
    #         lower_num += 1
            
    #     elif color_index < -8:
    #         color = 'crimson'
    #         higher_num += 1
            
    #     # plot predict score range vs. real score
    #     ln1 = ax.hlines(y=sub, xmin=lb, xmax=ub,
    #               linewidth=10, color=color, alpha = 1/4, label="predict")
    #     ln2 = ax.hlines(y=sub, xmin=score-0.5, xmax=score+0.5,
    #               linewidth=10, color=color, alpha = 0.7, label="real")
    #     ax.hlines(y=sub, xmin=score, xmax=(lb + ub) / 2, linestyles='dashed',
    #               linewidth=1, color=color, alpha = 0.7)
        
    # ax.set_xlabel("Sprint_50 Score", fontsize=22)
    # ax.set_ylabel("Subject", fontsize=22)
    
    # print("-- iter --")
    # print(f"correct: {correct_num}")
    # print(f"lower_num: {lower_num}")
    # print(f"higher_num: {higher_num}")
    
    # # print score error
    score_sprint_50_mae_total = MAE(a_max["sprint_50_score_pred"], a_max["sprint_50_score"])
    score_sprint_50_mae = MAE(a_max[a_max.sprint_50_score > 0]["sprint_50_score_pred"], a_max[a_max.sprint_50_score > 0]["sprint_50_score"])
    score_sprint_50_mape = MAPE(a_max[a_max.sprint_50_score > 0]["sprint_50_score_pred"], a_max[a_max.sprint_50_score > 0]["sprint_50_score"])
    sprint_50_mae = MAE(a_max["sprint_50_pred"], a_max["sprint_50"])
    sprint_50_mape = MAPE(a_max["sprint_50_pred"], a_max["sprint_50"])
    
    print()
    print(f"score mae: {score_sprint_50_mae_total:.2f} points")
    print(f"score mape: {score_sprint_50_mape:.2f} %")
    print(f"s mae: {sprint_50_mae:.2f} s\ns mape: {sprint_50_mape:.2f} %")
    
    # plt_corr(a_max["sprint_50_score_pred"], a_max["sprint_50_score"])