# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 15:26:35 2021

@author: Datong
"""
import numpy as np
from scipy import signal
from ChinaScoreEstimation import chinaScoreEstimation

def trap_intergral_1d(data, fz):
    data_trap = np.zeros(len(data))
    h = 1 / fz
    for i in range(1, len(data)):
        f1 = data[i-1]
        f2 = data[i]
        data_trap[i] = data_trap[i-1] + (f1 + f2) * h / 2
    return data_trap

def linear_function(model_a, model_b, feats):
    assert len(model_a) == len(feats), "feature length not equal to parameter length!"
    pred_value = 0
    for i in range(len(model_a)):
        pred_value += model_a[i] * feats[i]
    pred_value += model_b
    return pred_value

def is_local_min(now, prev_1, prev_2, next_1, next_2):
    if now < prev_1 and now < next_1:
        if now < prev_2 and now < next_2:
            return 1
    return 0
    
def is_local_max(now, prev_1, prev_2, prev_3,
                 next_1, next_2, next_3):
    if now > prev_1 and now > next_1:
        if now > prev_2 and now > next_2:
            if now > prev_3 and now > next_3:
                return 1
    return 0

class youthJumpPredict():
    
    def __init__(self, acc_x, height, gender, grade,
                 cmj_model, slj_model, sprint_50_model,
                 history_slj=-1, history_sprint_50=-1):
        self.acc_x = acc_x                          # 100 * 10 float
        self.height = height                        # 1 int
        self.gender = gender                        # 1 int
        self.grade = grade                          # 1 int
        self.history_slj = history_slj              # float
        self.history_sprint_50 = history_sprint_50  # float
        
        self.cmj_model = cmj_model                  # 7 float
        self.slj_model = slj_model                  # 2 float
        self.sprint_50_model = sprint_50_model      # 3 float
        
        self.fz = 100                               # 1 int
        self.acc_size = 1000                        # 1 int
        self.sos = signal.butter(2, 2, 'hp', fs=self.fz, output='sos')
        self.cse = chinaScoreEstimation()
        
        self.feat_8 = np.float32(-1.0)                          # 1 float
        self.feat_9 = np.float32(-1.0)                          # 1 float
        self.feat_16 = -1                           # 1 int
        self.feat_33 = -1                           # 1 int
        self.feat_41 = -1                           # 1 int
         
        self.cmj_pred = np.float32(-1.0)                        # 1 float
        self.slj_pred = np.float32(-1.0)                        # 1 float
        self.sprint_50_pred = np.float32(-1.0)                  # 1 float
        
        self.slj_lb = -1                            # 1 int
        self.slj_ub = -1                            # 1 int
        self.sprint_50_lb = -1                      # 1 int
        self.sprint_50_ub = -1                      # 1 int
    
    def cmj_check(self):
        '''
        cut acc_x into 4s data, and  check if cmj with specific potrue
        is successful by 2 condition.
        
        1. is data length enough before and after cmj. (2s seperately)
        
        2. is arm steady state in first 0.5s before cmj start .
          a. std < 0.1 g  (arm need keep static)
          b. mean < 0.5 g (keep arm straight)
             
        Returns
        -------
        0: sucess
        1: cmj failed
        r
        '''
        min_idx = np.argmin(self.acc_x)
        if min_idx < 2 * self.fz or min_idx > (len(self.acc_x) - 2 * self.fz):
            print("not enough data!")
            return 1
        # print("min_idx: ", min_idx)
        # print("min_value: ", np.min(self.acc_x))
        
        self.acc_x = self.acc_x[min_idx - 2 * self.fz:min_idx + 2 * self.fz]
        # print(f"std: {np.std(self.acc_x[:int(0.5 * self.fz)], ddof=1)}")
        # print(f"mean: {np.mean(self.acc_x[:int(0.5 * self.fz)])}")
        if np.std(
            self.acc_x[:int(0.5 * self.fz)], ddof=1) > 0.98:
            print("not static start (std)!")
            return 1
        
        if np.mean(
            self.acc_x[:int(0.5 * self.fz)]) > 4.9:
            print("not static start (mean)!")
            return 1
        
        return 0
    
    def get_intergral_twice(self, acc):
        '''
        get filted acc, velocity, displacement of input acc

        Parameters
        ----------
        acc : raw acc, float array, len:400

        Returns
        -------
        acc_filt : filted acc, float array
        velocity : float array
        displacement : float array
            
        '''
        acc_filt = signal.sosfilt(self.sos, acc)
        velocity = trap_intergral_1d(acc_filt, self.fz)
        displacement = trap_intergral_1d(velocity, self.fz)
        return (acc_filt.astype(np.float32),
                velocity.astype(np.float32),
                displacement.astype(np.float32))
    
    def get_vel_feature_idx(self, x_vel):
        '''
        find feature of x velocity with 6 events

        Parameters
        ----------
        x_vel : np.array, float, x velocity 
            
        Returns
        -------
        index list of feature, len=6

        '''
        p1 = -1
        p2 = -1
        p3 = -1
        p4 = -1
        p5 = -1
        p6 = -1
        
        for i in range(110, len(x_vel)-50):
            if (p1 < 0) and (x_vel[i] < -0.4):
                if is_local_min(x_vel[i], x_vel[i-1], x_vel[i-2],
                                x_vel[i+1], x_vel[i+2]):
                    p1 = i
                    continue
                    
            if (p1 > 0) and (p2 < 0) and (x_vel[i] > 0.5):
                if is_local_max(x_vel[i],
                    x_vel[i-1], x_vel[i-2], x_vel[i-3],
                    x_vel[i+1], x_vel[i+2], x_vel[i+3]):
                    p2 = i
                    continue
                
            if (p2 > 0) and (p3 < 0) and (x_vel[i] < -1):
                if is_local_min(x_vel[i], x_vel[i-1], x_vel[i-2],
                                x_vel[i+1], x_vel[i+2]):
                    p3 = i
                    continue
                
            if (p3 > 0) and (p4 < 0) and (x_vel[i] > 1.5):
                if is_local_max(x_vel[i],
                    x_vel[i-1], x_vel[i-2], x_vel[i-6],
                    x_vel[i+1], x_vel[i+2], x_vel[i+6]):
                    p4 = i
                    continue
                
            if (p4 > 0) and (p5 < 0) and (x_vel[i] < 0.25):
                if is_local_min(x_vel[i], x_vel[i-1], x_vel[i-2],
                                x_vel[i+1], x_vel[i+2]):
                    p5 = i
                    continue
                
            if (p5 > 0) and (p6 < 0) and (x_vel[i] > 0.25):
                if is_local_max(x_vel[i],
                    x_vel[i-1], x_vel[i-2], x_vel[i-3],
                    x_vel[i+1], x_vel[i+2], x_vel[i+3]):
                    p6 = i
                    break
        
        return [p1, p2, p3, p4, p5, p6]
    
    def get_disp_feature_idx(self, x_disp):
        '''
        find feature of x displacement with 6 events

        Parameters
        ----------
        x_disp : np.array, float, x displacement 
            
        Returns
        -------
        index list of feature, len=6

        '''
        p1 = -1
        p2 = -1
        p3 = -1
        p4 = -1
        p5 = -1
        p6 = -1
        
        for i in range(110, len(x_disp)-25):
            if (p1 < 0) and (x_disp[i] < -0.1):
                if is_local_min(x_disp[i], x_disp[i-1], x_disp[i-2],
                                x_disp[i+1], x_disp[i+2]):
                    p1 = i
                    continue
                    
            if (p1 > 0) and (p2 < 0) and (x_disp[i] > x_disp[p1]):
                if is_local_max(x_disp[i],
                    x_disp[i-1], x_disp[i-2], x_disp[i-3],
                    x_disp[i+1], x_disp[i+2], x_disp[i+3]):
                    p2 = i
                    continue
                
            if (p2 > 0) and (p3 < 0) and (x_disp[i] < x_disp[p1]):
                if is_local_min(x_disp[i], x_disp[i-1], x_disp[i-2],
                                x_disp[i+1], x_disp[i+2]):
                    p3 = i
                    continue
                
            if (p3 > 0) and (p4 < 0) and (x_disp[i] > x_disp[p1]):
                if is_local_max(x_disp[i],
                    x_disp[i-1], x_disp[i-2], x_disp[i-6],
                    x_disp[i+1], x_disp[i+2], x_disp[i+6]):
                    p4 = i
                    continue
                
            if (p4 > 0) and (p5 < 0) and (x_disp[i] > x_disp[p4]):
                if is_local_max(x_disp[i],
                    x_disp[i-1], x_disp[i-2], x_disp[i-6],
                    x_disp[i+1], x_disp[i+2], x_disp[i+6]):
                    p5 = i
                    continue
                
            if (p5 > 0) and (p6 < 0) and (x_disp[i] < x_disp[p5]):
                if is_local_min(x_disp[i], x_disp[i-1], x_disp[i-2],
                                x_disp[i+1], x_disp[i+2]):
                    p6 = i
                    break
        
        return [p1, p2, p3, p4, p5, p6]
    
    def acc_feature_extraction(self):
        '''
        get feature of acc x, if can not detect sepecific event that is unqualified data

        Returns
        -------
        error_code : int
            0: success
            2: can not find velocity feature
            3: can not find displacemnet feature

        '''
        error_code = 0
        x_filt, x_vel, x_disp = self.get_intergral_twice(self.acc_x)
        
        x_vel_idxs = self.get_vel_feature_idx(x_vel)
        if -1 in x_vel_idxs:
            error_code = 2
            
        x_disp_idxs = self.get_disp_feature_idx(x_disp)
        if -1 in x_disp_idxs:
            error_code = 3
            
        self.feat_8 = np.float32(x_vel[x_vel_idxs[2]] - x_vel[x_vel_idxs[1]])
        self.feat_9 = np.float32(x_vel[x_vel_idxs[3]] - x_vel[x_vel_idxs[2]])
        self.feat_16 =  x_vel_idxs[5] - x_vel_idxs[4]
        
        self.feat_33 = x_disp_idxs[4] - x_disp_idxs[3]
        self.feat_41 = self.feat_33 ** 2
        
        return error_code
    
    def predict_cmj(self):
        '''
        use trained model to predict cmj
        
        model generate by find_model/generate_predict_model.py

        Returns
        -------
        error_code : int
            4: cmj range error

        '''
        error_code = 0
        feats = np.array([self.height, self.feat_8, self.feat_9, self.feat_16,
                          self.feat_33, self.feat_41]).astype(np.float32)
        a = self.cmj_model[:-1].astype(np.float32)
        b = self.cmj_model[-1].astype(np.float32)
        
        self.cmj_pred = linear_function(a, b, feats).astype(np.float32)
        if self.cmj_pred < 10 or self.cmj_pred > 70:
            error_code = 4
            
        return error_code
    
    def predict_slj(self):
        '''
        use trained model to predict slj
        
        model generate by find_model/generate_predict_model.py

        Returns
        -------
        error_code : int
            4: cmj range error
            5: slj range error

        '''
        error_code = 0
        if self.cmj_pred > 0:
            feats = np.array([self.cmj_pred]).astype(np.float32)
            
            a = self.slj_model[:-1].astype(np.float32)
            b = self.slj_model[-1].astype(np.float32)
            
            self.slj_pred = linear_function(a, b, feats).astype(np.float32)
            if self.slj_pred < 50 or self.slj_pred > 330:
                error_code = 5
        else:
            error_code = 4
        return error_code
    
    def predict_sprint_50(self):
        '''
        use trained model to predict sprint_50
        
        model generate by find_model/generate_predict_model.py

        Returns
        -------
        error_code : int
            4: cmj range error
            5: slj range error
            6: sprint_50 range error

        '''
        error_code = 0
        if self.cmj_pred > 0:
            if self.slj_pred > 0:
                feats = np.array([self.cmj_pred, self.slj_pred]).astype(np.float32)
                
                a = self.sprint_50_model[:-1].astype(np.float32)
                b = self.sprint_50_model[-1].astype(np.float32)
                
                self.sprint_50_pred = linear_function(a, b, feats).astype(np.float32)
                if self.sprint_50_pred < 5 or self.sprint_50_pred > 15:
                    error_code = 6
            else:
                error_code = 5
        else:
            error_code = 4
        return error_code
    
    def update_slj_score(self):
        '''
        use current slj predict value and history slj predict value
        to update slj score.
        
        1. if slj is longer, then learning rate set to 0.7.

        2. if slj is shorter, then learning rate set to 0.3.
        
        Because best performance need to consider as higher weight,
        but still have to restrain outlier by prediction value.
        
        Returns
        -------
        error_code : int
            7: slj score score error

        '''
        error_code = 0
        if self.history_slj > 0:
            if self.slj_pred > self.history_slj:
                self.slj_pred = (np.float32(0.7) * self.slj_pred +
                                 np.float32(0.3) * self.history_slj)
            else:
                self.slj_pred = (np.float32(0.3) * self.slj_pred +
                                 np.float32(0.7) * self.history_slj)
        
        slj_score_pred = self.cse(
            self.gender, self.grade, 'slj', self.slj_pred)
        self.slj_lb = max(slj_score_pred - 7, 0)
        self.slj_ub = min(self.slj_lb + 15, 100)
        
        if self.slj_ub == 100:
            self.slj_lb = 85
        
        if slj_score_pred < 0 or slj_score_pred > 100:
            error_code = 7
        
        return error_code
    
    def update_sprint_50_score(self):
        '''
        use current sprint_50 predict value and history sprint_50 predict value
        to update sprint_50 score.
        
        1. if sprint_50 is faster, then learning rate set to 0.7.

        2. if sprint_50 is slower, then learning rate set to 0.3.
        
        Because best performance need to consider as higher weight,
        but still have to restrain outlier by prediction value.
        
        Returns
        -------
        error_code : int
            8: sprint_50 score error

        '''
        error_code = 0
        if self.history_sprint_50 > 0:
            if self.sprint_50_pred < self.history_sprint_50:
                self.sprint_50_pred = (np.float32(0.5) * self.sprint_50_pred +
                                       np.float32(0.5) * self.history_sprint_50)
            else:
                self.sprint_50_pred = (np.float32(0.3) * self.sprint_50_pred +
                                       np.float32(0.7) * self.history_sprint_50)
        
        sprint_50_score_pred = self.cse(
                self.gender, self.grade, 'sprint', self.sprint_50_pred)
        self.sprint_50_lb = max(sprint_50_score_pred - 7, 0)
        self.sprint_50_ub = min(self.sprint_50_lb + 15, 100)
        
        if self.sprint_50_ub == 100:
            self.sprint_50_lb = 85
            
        if sprint_50_score_pred < 0 or sprint_50_score_pred > 100:
            error_code = 8
            
        return error_code
    
    def preProcess(self):
        error_code = self.cmj_check()
        
        if error_code == 0:
            error_code = self.acc_feature_extraction()
        else:
            return error_code
        return error_code
    
    def prediction(self):
        error_code = self.predict_cmj()
        
        if error_code == 0:
            error_code = self.predict_slj()
        else:
            return error_code
        
        if error_code == 0:
            error_code = self.predict_sprint_50()
        else:
            return error_code
        
        return error_code
    
    def postProcess(self):
        error_code = self.update_slj_score()
        
        if error_code == 0:
            error_code = self.update_sprint_50_score()
        else:
            return error_code
        
        return error_code
    
    def print_error(self, error_code):
        error_table = {
             0:"ok",
             1:"cmj failed",
             2:"x_vel can't find feature",
             3:"x_disp can't find feature",
             4:"cmj range error",
             5:"slj range error",
             6:"sprint_50 range error",
             7:"slj score score error",
             8:"sprint_50 score error"
            }
        print(error_table[error_code])
    
    def excuteCmjPredict(self):
        '''
        handle error by error code

        Returns
        -------
        error_code : int

        '''
        error_code = self.preProcess()
        
        if error_code == 0:
            error_code = self.prediction()
        else:
            self.print_error(error_code)
            return error_code
            
        if error_code == 0:
            error_code = self.postProcess()
        else:
            self.print_error(error_code)
            return error_code
       
        return error_code
        
#%%
if __name__ == '__main__':
    from ReadData import getNaxsen 
    from find_model.AccFeatureExtraction import get_fn
    from find_model.PlotTool import Correlaiton
    from matplotlib import pyplot as plt
    import pandas as pd
    
    def downsample(acc_raw, fz_raw, fz_new): # only pick, no interpolate
        acc = acc_raw[::(fz_raw // fz_new)] 
        return acc[:, 0]

    dataPath = 'dataset_youth'
    goldPath = 'gold_record.csv'
    cmj_model_path = 'find_model/cmj_model.txt'
    slj_model_path = 'find_model/slj_model.txt'
    sprint_50_model_path = 'find_model/sprint_50_model.txt'
    
    samp_rate = 1000
    down_spRate = 100
    cse = chinaScoreEstimation()
    is_print_detail = 0
    
    cmj_model = np.loadtxt(cmj_model_path, dtype=str, delimiter=',').astype(float).round(6)
    slj_model = np.loadtxt(slj_model_path, dtype=str, delimiter=',').astype(float).round(6)
    sprint_50_model = np.loadtxt(sprint_50_model_path, dtype=str, delimiter=',').astype(float).round(6)
    
    gold = pd.read_csv(goldPath)
    sub_prev = -1
    source_prev = []
    history_slj = -1
    history_sprint_50 = -1
    
    tmp_mean = []
    tmp_std = []
    for i in range(0, len(gold)):
        # print(f"\r{round(i/len(gold), 2) * 100} %", end='\r')
        
        # get data
        sub = gold.iloc[i]["subNum"].astype(int)
        testNum = gold.iloc[i]["testNum"].astype(int)
        source = gold.iloc[i]["source"]
        height  = gold.iloc[i]["height"]
        gender = gold.iloc[i]["gender"]
        grade = gold.iloc[i]["grade"].astype(int)
        cmj = gold.iloc[i]["cmj"]
        slj = gold.iloc[i]["slj"]
        sprint_50 = gold.iloc[i]["sprint_50"]
        slj_score = cse(gender, grade, 'slj', slj)
        sprint_50_score = cse(gender, grade, 'sprint', sprint_50)     
        
        if source == "NTHU_pilot" or source == "NTHU_pilot_median":
            continue
        
        if source == "IBSH_SS" and (sub == 16 or sub == 8):
            continue
        
        fn = get_fn(dataPath, source, sub, testNum)
        getData = getNaxsen(fn, samp_rate)
        data = getData()
        acc_x = downsample(data, samp_rate, down_spRate).astype(np.float32)
        
        if sub == sub_prev and source == source_prev:
            pass
            # print(f"---- sub:{sub} test:{testNum} source:{source} ----")
            # print("Exist history data!")
        else:
            print(f"\n==== sub:{sub} test:{testNum} source:{source} ====")
            # print("first update!")
            history_slj = -1
            history_sprint_50 = -1
            
        yjp = youthJumpPredict(acc_x, height, gender, grade,
                               cmj_model, slj_model, sprint_50_model,
                               history_slj, history_sprint_50)
        error_code = yjp.excuteCmjPredict()
        
        #  ==== print algo result ====
        if error_code == 0:
            gold.at[i, "cmj_pred"] = yjp.cmj_pred
            gold.at[i, "slj_pred"] = yjp.slj_pred
            gold.at[i, "slj_score"] = slj_score
            gold.at[i, "slj_lb"] = yjp.slj_lb
            gold.at[i, "slj_ub"] = yjp.slj_ub
            gold.at[i, "sprint_50_pred"] = yjp.sprint_50_pred
            gold.at[i, "sprint_50_score"] = sprint_50_score
            gold.at[i, "sprint_50_lb"] = yjp.sprint_50_lb
            gold.at[i, "sprint_50_ub"] = yjp.sprint_50_ub            
            
            if is_print_detail:
                print(f"\ncmj: {cmj:.2f} cm")
                print(f"cmj_pred: {yjp.cmj_pred:.2f} cm")
                
                print(f"\nslj: {slj:.2f} cm")
                print(f"slj_pred: {yjp.slj_pred:.2f} cm")
                
                print(f"\nsprint_50: {sprint_50:.2f} s")
                print(f"sprint_50_pred: {yjp.sprint_50_pred:.2f} s")
                
                print(f"\nslj predict lower bound: {yjp.slj_lb}")
                print(f"slj predict upper bound: {yjp.slj_ub}")
                print(f"slj score: {slj_score}")
                
                print(f"\nsprint_50 predict lower bound: {yjp.sprint_50_lb}")
                print(f"sprint_50 predict upper bound: {yjp.sprint_50_ub}")
                print(f"sprint_50 score: {sprint_50_score}")
            
            # update user information
            sub_prev = sub
            source_prev = source
            history_slj = yjp.slj_pred
            history_sprint_50 = yjp.sprint_50_pred
            # print("---- finish ----")
        else:
            continue
    gold.loc[:, "cmj_pred":] = gold.loc[:, "cmj_pred":].astype(np.float32)
    gold.to_csv("verify_table.csv", index=False)
    
    #%% plot slj score range
    gold_last =  gold.groupby(['subNum', 'source']).last()
    gold_last = gold_last[gold_last["cmj_pred"].notnull()].reset_index()
    gold_last = gold_last[gold_last["slj"].notnull()].reset_index()
    correct_num = 0
    lower_num = 0
    higher_num = 0
    
    plt_corr = Correlaiton()
    plt_corr(gold_last["cmj_pred"], gold_last["cmj"])
    plt_corr(gold_last["slj_pred"], gold_last["slj"])
    plt_corr(gold_last["cmj"], gold_last["slj"])
    
    fig, ax = plt.subplots(figsize=(13, 10))
    for i in range(len(gold_last)):
        sub = i + 1
        score = gold_last.iloc[i]["slj_score"]
        lb = gold_last.iloc[i]["slj_lb"].item()
        ub = gold_last.iloc[i]["slj_ub"].item()
        half_score_range = (ub - lb) / 2
        
        # for text
        subNum = gold_last.iloc[i]["subNum"]
        source = gold_last.iloc[i]["source"]
        gender = gold_last.iloc[i]["gender"]
        
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
        ax.text(ub, sub, f"sub{subNum}_{source}_{gender}", ha="center")
        
    print("\n==== SLJ ====")
    print(f"correct: {correct_num}")
    print(f"lower_num: {lower_num}")
    print(f"higher_num: {higher_num}")
    
    #%% plot sprint_50 score range
    gold_last = gold_last[gold_last["sprint_50"].notnull()]
    gold_last = gold_last[gold_last["cmj_pred"].notnull()].reset_index()
    plt_corr(gold_last["sprint_50_pred"], gold_last["sprint_50"])
    plt_corr(gold_last["cmj"], gold_last["sprint_50"])
    plt_corr(gold_last["slj"], gold_last["sprint_50"])
    correct_num = 0
    higher_num = 0
    lower_num = 0
    fig, ax = plt.subplots(figsize=(13, 10))
    for i in range(len(gold_last)):
        sub = i + 1
        score = gold_last.iloc[i]["sprint_50_score"]
        lb = gold_last.iloc[i]["sprint_50_lb"].item()
        ub = gold_last.iloc[i]["sprint_50_ub"].item()
        half_score_range = (ub - lb) / 2
        
        # for text
        subNum = gold_last.iloc[i]["subNum"]
        source = gold_last.iloc[i]["source"]
        gender = gold_last.iloc[i]["gender"]
        
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
        ax.text(ub, sub, f"sub{subNum}_{source}_{gender}", ha="center")
        
    print("\n==== Spinrt_50 ====")
    print(f"correct: {correct_num}")
    print(f"lower_num: {lower_num}")
    print(f"higher_num: {higher_num}")
    
#%%
    gold_last = gold.groupby(["subNum", "source"]).last()
    cmj_mae = round((gold_last["cmj_pred"] - gold_last["cmj"]).abs().mean(), 2)
    cmj_mape = round(
        ((gold_last["cmj_pred"] - gold_last["cmj"]).abs() / gold_last["cmj"] * 100).mean(), 2)
    
    slj_mae = round((gold_last["slj_pred"] - gold_last["slj"]).abs().mean(), 2)
    slj_mape = round(
        ((gold_last["slj_pred"] - gold_last["slj"]).abs() / gold_last["slj"] * 100).mean(), 2)
    
    sprint_50_mae = round((gold_last["sprint_50_pred"] - gold_last["sprint_50"]).abs().mean(), 2)
    sprint_50_mape = round(
        ((gold_last["sprint_50_pred"] - gold_last["sprint_50"]).abs() / gold_last["sprint_50"] * 100).mean(), 2)
    
    print(f"\ncmj_mae: {cmj_mae:.2f} cm")
    print(f"cmj_mape: {cmj_mape:.2f} %")
    print(f"slj_mae: {slj_mae:.2f} cm")
    print(f"slj_mape: {slj_mape:.2f} %")
    print(f"sprint_50_mae: {sprint_50_mae:.2f} s")
    print(f"sprint_50_mape: {sprint_50_mape:.2f} %")
    
    