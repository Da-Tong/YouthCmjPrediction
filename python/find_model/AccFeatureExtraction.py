# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 23:20:41 2021

@author: Datong
"""

import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

def get_res_acc(acc, remove_grav):
    if remove_grav:
        return np.sqrt((acc **2).sum(axis=1)) - 9.8
    else:
        return np.sqrt((acc **2).sum(axis=1))

def trap_intergral_1d(data, fz):
    data_trap = np.zeros(len(data))
    h = 1 / fz
    for i in range(1, len(data)):
        f1 = data[i-1]
        f2 = data[i]
        data_trap[i] = data_trap[i-1] + (f1 + f2) * h / 2
        
    return data_trap

def get_fn(dataPath, source, sub, testNum):
    import os
    if source == 'IBSH':
        fn = os.path.join(dataPath, source, "NAXSEN",
                          f'sub{sub:03d}', f't{testNum:03d}.csv')
        
    elif source == 'IBSH_SS':
        fn = os.path.join(dataPath, source, "NAXSEN",
                          f'sub{sub}', f't{testNum:03d}.csv')
        
    elif source == 'YCSH':
        fn = os.path.join(dataPath, source,
                          f'sb{sub:02d}', f'cmj_{testNum:02d}.csv')
                
    elif source == 'GRJH':
        fn = os.path.join(dataPath, source,
                          f'GRJH_{sub:02d}',  f'test{testNum}.csv')
        
    elif source == "YCSH_post":
        fn = os.path.join(dataPath, source, 
                          f"sub_2{sub:02d}", f"test{testNum}.csv")
    
    elif source == "NTHU_phase1_pre":
        fn = os.path.join(dataPath, source, 
                          f'sub{sub:03d}', f't{testNum:03d}.csv')
        
    elif source == "NTHU_pilot_post":
        fn = os.path.join(dataPath, source, 
                          f'sub{sub:03d}', f't{testNum:03d}.csv')
    
    elif source == "NTHU_pilot_pre":
        fn = os.path.join(dataPath, source, 
                          f'{sub}', f't{testNum:03d}.csv')
    
    elif source == "NTSH_pre":
        fn = os.path.join(dataPath, source, 
                          f'sub{sub:03d}', f't{testNum:03d}.csv')
        
    elif source == "NTSU_round1":
        fn = os.path.join(dataPath, source, "jump_imu",
                          f'sub_{sub:03d}', f't{testNum:03d}.csv')
        
    return fn


class accFeatureExtraction():
    
    def __init__(self, fz_raw, fz_new, is_plot_feature=0):
        self.fz_raw = fz_raw # sampling rate of device
        self.fz_new = fz_new # sampling rate that want to downsample
        self.is_plot_feature = is_plot_feature # where to plot feature figure
        self.sos = signal.butter(2, 2, 'hp',
                                 fs=self.fz_new, output='sos')
    
    def __call__(self, acc_raw):
        '''
        input cmj acc raw data then output feature of this trial

        Parameters
        ----------
        acc_raw : n*3 np.array, float, unit: m/s^2

        Returns
        -------
        feature list: len=40, float

        '''
        acc_x, acc_y, acc_z, acc_raw = self.downsample(acc_raw)
        x_filt, x_vel, x_disp = self.get_intergral_twice(acc_x)
        
        raw_feats = self.get_acc_feature(acc_x, x_filt) # len:4
        
        x_vel_idxs = self.get_vel_feature_idx(x_vel)
        if -1 in x_vel_idxs:
            print("velocity features error!")
            return []
        
        x_disp_idxs = self.get_disp_feature_idx(x_disp)
        # print(f'x_disp_idxs = {x_disp_idxs} ')
        if -1 in x_disp_idxs:
            print("displacement features error!")
            return []
        
        x_vel_feats = self.get_feature_list(x_vel, x_vel_idxs) # len:18
        x_disp_feats = self.get_feature_list(x_disp, x_disp_idxs) # len:18
        
        if self.is_plot_feature:
            self.plot_vel_feature(x_vel, x_vel_idxs)
            self.plot_acc_feature(acc_x, x_filt)
            self.plot_disp_feature(x_disp, x_disp_idxs)
                    
        return raw_feats + x_vel_feats + x_disp_feats
        
    
    def downsample(self, acc_raw): # only pick, no interpolate
        acc = acc_raw[::(self.fz_raw // self.fz_new)] 
        return acc[:, 0], acc[:, 1], acc[:, 2], acc
    
    def get_intergral_twice(self, acc):
        '''
        get filted acc, velocity, displacement of input acc

        Parameters
        ----------
        acc : raw acc, float array(400, 1)

        Returns
        -------
        acc_filt : filted acc, float array
        velocity : float array
        displacement : float array
            
        '''
        acc_filt = signal.sosfilt(self.sos, acc)
        velocity = trap_intergral_1d(acc_filt, self.fz_new)
        displacement = trap_intergral_1d(velocity, self.fz_new)
        return acc_filt, velocity, displacement
    
    def get_acc_feature(self, acc_x, x_filt):
        '''
        find feature of raw data

        Parameters
        ----------
        acc_x : np.array, raw data of acc x
        
        x_filt : np.array, filted data of acc x

        Returns
        -------
        list: float, len=4

        '''
        feat_1 = np.min(acc_x)
        feat_2 = np.max(acc_x)
        
        feat_3 = np.min(x_filt)
        feat_4 = np.max(x_filt)
        
        return [feat_1, feat_2, feat_3, feat_4]
    
    def is_local_min(self, now, prev_1, prev_2, next_1, next_2):
        if now < prev_1 and now < next_1:
            if now < prev_2 and now < next_2:
                return 1
        return 0
    
    def is_local_max(self, now, prev_1, prev_2, prev_3,
                     next_1, next_2, next_3):
        if now > prev_1 and now > next_1:
            if now > prev_2 and now > next_2:
                if now > prev_3 and now > next_3:
                    return 1
        return 0
    
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
                if self.is_local_min(x_vel[i], x_vel[i-1], x_vel[i-2],
                                     x_vel[i+1], x_vel[i+2]):
                    p1 = i
                    continue
                    
            if (p1 > 0) and (p2 < 0) and (x_vel[i] > 0.5):
                if self.is_local_max(x_vel[i],
                        x_vel[i-1], x_vel[i-2], x_vel[i-3],
                        x_vel[i+1], x_vel[i+2], x_vel[i+3]):
                    p2 = i
                    continue
                
            if (p2 > 0) and (p3 < 0) and (x_vel[i] < -1):
                if self.is_local_min(x_vel[i], x_vel[i-1], x_vel[i-2],
                                     x_vel[i+1], x_vel[i+2]):
                    p3 = i
                    continue
                
            if (p3 > 0) and (p4 < 0) and (x_vel[i] > 1.5):
                if self.is_local_max(x_vel[i],
                        x_vel[i-1], x_vel[i-2], x_vel[i-6],
                        x_vel[i+1], x_vel[i+2], x_vel[i+6]):
                    p4 = i
                    continue
                
            if (p4 > 0) and (p5 < 0) and (x_vel[i] < 0.25):
                if self.is_local_min(x_vel[i], x_vel[i-1], x_vel[i-2],
                                     x_vel[i+1], x_vel[i+2]):
                    p5 = i
                    continue
                
            if (p5 > 0) and (p6 < 0) and (x_vel[i] > 0.25):
                if self.is_local_max(x_vel[i],
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
                if self.is_local_min(x_disp[i], x_disp[i-1], x_disp[i-2],
                                     x_disp[i+1], x_disp[i+2]):
                    p1 = i
                    continue
                    
            if (p1 > 0) and (p2 < 0) and (x_disp[i] > x_disp[p1]):
                if self.is_local_max(x_disp[i],
                        x_disp[i-1], x_disp[i-2], x_disp[i-3],
                        x_disp[i+1], x_disp[i+2], x_disp[i+3]):
                    p2 = i
                    continue
                
            if (p2 > 0) and (p3 < 0) and (x_disp[i] < x_disp[p1]):
                if self.is_local_min(x_disp[i], x_disp[i-1], x_disp[i-2],
                                     x_disp[i+1], x_disp[i+2]):
                    p3 = i
                    continue
                
            if (p3 > 0) and (p4 < 0) and (x_disp[i] > x_disp[p1]):
                if self.is_local_max(x_disp[i],
                        x_disp[i-1], x_disp[i-2], x_disp[i-6],
                        x_disp[i+1], x_disp[i+2], x_disp[i+6]):
                    p4 = i
                    continue
                
            if (p4 > 0) and (p5 < 0) and (x_disp[i] > x_disp[p4]):
                if self.is_local_max(x_disp[i],
                        x_disp[i-1], x_disp[i-2], x_disp[i-6],
                        x_disp[i+1], x_disp[i+2], x_disp[i+6]):
                    p5 = i
                    continue
                
            if (p5 > 0) and (p6 < 0) and (x_disp[i] < x_disp[p5]):
                if self.is_local_min(x_disp[i], x_disp[i-1], x_disp[i-2],
                                     x_disp[i+1], x_disp[i+2]):
                    p6 = i
                    break
        
        return [p1, p2, p3, p4, p5, p6]
        
    
    def get_feature_list(self, sig, sig_idxs):
        '''
        use input data and feature index to get feature value

        Parameters
        ----------
        sig: x velocity or x displacement, np.array, float, len: 400
        
        sig_idxs: index of feature, int array, len: 6

        Returns
        -------
        feature list, len: 18

        '''
        feat_1 = np.max(sig)
        feat_2 = np.min(sig)
        
        feat_3, feat_4, feat_5, feat_6, feat_7 = np.diff(sig[sig_idxs])
        feat_8, feat_9, feat_10, feat_11, feat_12 = np.diff(sig_idxs)
        feat_13, feat_14, feat_15, feat_16, feat_17, feat_18 = sig[sig_idxs]
        
        return [
            feat_1, feat_2, feat_3, feat_4, feat_5, feat_6, feat_7, feat_8, 
            feat_9, feat_10, feat_11, feat_12, feat_13, feat_14, feat_15,
            feat_16, feat_17, feat_18
            ]
    
    def plot_vel_feature(self, x_vel, x_vel_idxs):
        fig, ax = plt.subplots()
        ax.plot(x_vel)
        ax.plot(x_vel_idxs, x_vel[np.array(x_vel_idxs)], 'rx')
        ax.set_title('velocity')
        
    def plot_acc_feature(self, acc_x, x_filt):
        fig, ax = plt.subplots()
        ax.plot(acc_x, label='acc_x')
        ax.plot(np.argmin(acc_x), np.min(acc_x), 'rx')
        ax.plot(np.argmax(acc_x), np.max(acc_x), 'rx')
        
        # ax.plot(x_filt, label='x_filt')
        # ax.plot(np.argmin(x_filt), np.min(x_filt), 'rx')
        # ax.plot(np.argmax(x_filt), np.max(x_filt), 'rx')
        # ax.legend()
        ax.set_title('acc')
        
    def plot_disp_feature(self, x_disp, x_disp_idxs):
        fig, ax = plt.subplots()
        ax.plot(x_disp)
        ax.plot(x_disp_idxs, x_disp[x_disp_idxs], 'rx')
        ax.set_title('displacement')
        
        
        
if __name__ == '__main__':
    from ReadData import getNaxsen
    import pandas as pd
    
    dataPath = '../dataset_youth'
    goldPath = '../gold_record.csv'
    
    samp_rate = 1000
    down_spRate = 100
    gold = pd.read_csv(goldPath)
    gold_save = gold.copy()
    
    is_plot_feature = 1
    
    for i in range(0, 10):
        print(f'i:{i} ', end='')
        
        sub = gold.iloc[i]["subNum"].astype(int)
        testNum = gold.iloc[i]["testNum"].astype(int)
        source = gold.iloc[i]["source"]
        
        fn = get_fn(dataPath, source, sub, testNum)
        getData = getNaxsen(fn, samp_rate)
        data = getData()
        afe = accFeatureExtraction(samp_rate, down_spRate, is_plot_feature)
        feats = afe(data)
        break
    
    # sos = signal.butter(2, 2, 'hp', fs=100, output='sos')
    # accx = data[::10, 0]
    # # acc_filt = signal.sosfilt(sos, accx)
    
    # acc_filt = trap_intergral_1d(accx, 100)
    # fig, ax = plt.subplots()
    # ax.plot(acc_filt)
    
    # acc_x, acc_y, acc_z, acc_raw = afe.downsample(data)
    # x_filt, x_vel, x_disp = afe.get_intergral_twice(acc_x)
    # fig, ax = plt.subplots()
    # ax.plot(x_vel)