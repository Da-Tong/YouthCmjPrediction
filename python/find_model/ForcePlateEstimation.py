# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 13:40:19 2021

@author: Datong
"""

import os
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt

class getForce():
    
    def __init__(self, path, sub, testNum):
        self.path = path
        self.sub = sub
        self.testNum = testNum
        
    def __call__(self):
        fn = self.get_filename()
        if not os.path.isfile(fn):
            print(f"{fn} is not exist!")
            return [], []
        
        force, time = self.get_force(fn)
        force, time = self.cut_force(force, time)
        
        return np.array(force), np.array(time)
    
    def get_filename(self):
        fn = os.path.join(self.path,
                          f'sub{self.sub}',
                          f't{self.testNum:03d}.txt')
        return fn
        
    def get_force(self, fn):
        time = []
        data = []
        st_idx = -1
        with open(fn, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i in range(len(lines)):                
                line = lines[i]
                if len(line) > 1:
                    if 'kgf' in line:
                        time_idx = line.split(';').index('Time ')
                        force_idx = line.split(';').index('Force 公斤重 (kgf)')
                        
                    if line[0] == '0':
                        st_idx = i
                        strt = line.split('\t')[time_idx]
                        st_time = datetime.strptime(strt, '%H:%M:%S.%f')
                        time.append(0)
                        data.append(float(line.split('\t')[force_idx]))
                    
                    if i > st_idx and st_idx > 0:
                        strt = line.split('\t')[time_idx]
                        tmp_time = datetime.strptime(strt, '%H:%M:%S.%f')
                        time.append((tmp_time - st_time).total_seconds() * 1000)
                        data.append(float(line.split('\t')[force_idx]))
        return data, time
    
    def cut_force(self, force, time):
        cut_idx = np.argmax(force)
        return (force[cut_idx - 200:cut_idx + 200],
                time[cut_idx - 200:cut_idx + 200])
                

class forceplateEstimation():
    '''
        
    from forceplate data get vertical jump height and RSI, RFD
        
    '''
    
    def __init__(self, force, weight, jump_type, samp_rate, isPlot):
        self.force = force
        self.weight = weight
        self.jump_type = jump_type
        self.samp_rate = samp_rate
        self.isPlot = isPlot
        
        self.contact_idxs = []
        self.toeOff_idxs = []
        self.toeoff_fpeak_idx = []
        self.contact_fpeak_idx = []
        self.jump_toeoff_idx = []
        self.jump_contact_idx = []
        
    def __call__(self):
        fpeak_idxs = self.find_events_and_peaks()
        self.find_jump_peak(fpeak_idxs)
        
        if self.jump_type == 'sj':
            jh = self.get_jump_height()
            return jh
        
        elif self.jump_type == 'cmj':
            jh = self.get_jump_height()
            rfd_mean, rfd_max = self.get_rfd(fpeak_idxs)
            return jh, rfd_mean, rfd_max
            
        elif self.jump_type == 'dj':
            rsi = self.get_rsi()
            return rsi
        
    def find_events_and_peaks(self):
        '''
        detect contact and toe-off point
        
        detect force local maximun with value bigger than weight

        Returns
        -------
        None.

        '''
        fpeak_idxs = []
        for i in range(1, len(self.force)-1):
            if self.force[i-1] > 6 and self.force[i+1] < 1:
                if self.force[i] < 1 and i :
                    toeOff = i
                else:
                    toeOff = i + 1
                
                if toeOff not in self.toeOff_idxs:
                    self.toeOff_idxs.append(toeOff)
            
            if self.force[i-1] < 1 and self.force[i+1] > 5:
                if self.force[i] < 1:
                    contact = i
                else:
                    contact = i - 1
                
                if contact not in self.contact_idxs:
                    self.contact_idxs.append(contact)
            
            if (self.force[i] > self.force[i-1] and
                self.force[i] > self.force[i+1] and
                self.force[i] > 1.2 * self.weight): 
                fpeak_idxs.append(i)
                
        return np.array(fpeak_idxs)
                
    def find_jump_peak(self, fpeak_idxs):
        for i in range(len(fpeak_idxs)-1):
            if (len(np.where(self.force[fpeak_idxs[i]:fpeak_idxs[i+1]] < 1)[0])
                > 0.05 * self.samp_rate):
                self.toeoff_fpeak_idx = fpeak_idxs[i]
                self.contact_fpeak_idx = fpeak_idxs[i+1]
                break
            
    def get_jump_height(self):
        '''
        get jump height between two force peak

        Returns
        -------
        jh : jump height (cm), float

        '''
        # for use get_jump_height() seperately
        if len(self.contact_idxs) < 1:
            fpeak_idxs = self.find_events_and_peaks()
            self.find_jump_peak(fpeak_idxs)
            
        toe_con_list = np.array(sorted(self.contact_idxs + self.toeOff_idxs))
        toe_con_list = toe_con_list[np.logical_and(
            toe_con_list > self.toeoff_fpeak_idx,
            toe_con_list < self.contact_fpeak_idx)]
        if len(toe_con_list) == 2:
            ft = (toe_con_list[1] - toe_con_list[0]) / self.samp_rate
            jh = 9.8 * (ft ** 2) / 8 * 100
            self.jump_toeoff_idx = toe_con_list[0]
            self.jump_contact_idx = toe_con_list[1]
        
        if self.isPlot and self.jump_type == 'sj':
            fig, ax = plt.subplots()
            ax.plot(self.force, 'x-', label='force')
            ax.plot(self.jump_toeoff_idx, self.force[self.jump_toeoff_idx],
                    'ro', label='toeoff')
            ax.plot(self.jump_contact_idx, self.force[self.jump_contact_idx],
                    'ro', label='contact')
            ax.legend()
        
        return jh
    
    def get_rfd(self, fpeak_idxs):
        '''
        reate of force develope, mean and max, unit (N / sec)

        Parameters
        ----------
        fpeak_idxs : np array
            position of force peaks from find_events_and_peaks()

        Returns
        -------
        rfd_mean, rfd_max

        '''
        for i in range(self.toeoff_fpeak_idx, 0, -1):
            if np.mean(self.force[i-5:i]) - self.weight < 3:
                if np.std(self.force[i-5:i]) < 2:
                    steady_idx = i
                    break
        
        # start develope forece point : point where force > weight 10N
        rfd_start_idx = (np.where(
            self.force[steady_idx:] > (self.weight + 1.02))[0][0] + steady_idx)
        rfd_peak_idx = (np.argmax(
            self.force[steady_idx:self.jump_toeoff_idx]) + steady_idx)
        rfd_mean = 9.8 * ((self.force[rfd_peak_idx] - self.force[rfd_start_idx]) /
                   ((rfd_peak_idx - rfd_start_idx) / self.samp_rate))
        rfd_max = (9.8 * np.max(np.diff(self.force[rfd_start_idx:rfd_peak_idx])) /
                  (1 / self.samp_rate))
        
        if self.isPlot:
            fig, ax = plt.subplots()
            ax.plot(self.force, 'x-', label='force')
            ax.plot(self.jump_toeoff_idx, self.force[self.jump_toeoff_idx],
                    'ro', label='toeoff')
            ax.plot(self.jump_contact_idx, self.force[self.jump_contact_idx],
                    'ro', label='contact')
            ax.plot(rfd_start_idx, self.force[rfd_start_idx],
                    'y*', label='rfd_start_idx', ms=10)
            ax.plot(rfd_peak_idx, self.force[rfd_peak_idx],
                    'r*', label='rfd_peak_idx', ms=10)
            ax.legend()
        
        return rfd_mean, rfd_max
    
    def get_rsi(self):
        '''
        Reactive Strength Index: jump height / contact time, unit (m / sec)

        Returns
        -------
        rsi: float

        '''
        self.contact_idxs = np.array(self.contact_idxs)
        self.toeOff_idxs = np.array(self.toeOff_idxs)
        rsi_1 = self.contact_idxs[self.contact_idxs < self.toeoff_fpeak_idx][-1]
        rsi_2 = self.toeOff_idxs[self.toeOff_idxs > self.toeoff_fpeak_idx][0]
        rsi_3 = self.contact_idxs[self.contact_idxs > rsi_2][0]
        
        jh = 9.8 * ((rsi_3 - rsi_2) / self.samp_rate) ** 2 / 8
        rsi = jh / ((rsi_2 - rsi_1) / self.samp_rate)
        
        if self.isPlot:
            fig, ax = plt.subplots()
            ax.plot(self.force, 'x-', label='force')
            ax.plot(rsi_1, self.force[rsi_1], 'y*', label='rsi_1', ms=10)
            ax.plot(rsi_2, self.force[rsi_2], 'g*', label='rsi_2', ms=10)
            ax.plot(rsi_3, self.force[rsi_3], 'b*', label='rsi_3', ms=10)
            ax.legend()
            
        return rsi

def get_profile(profile, sub):
    if len(list(profile[profile.subNum == sub].index)) > 0:
        gender = profile[profile.subNum == sub]["gender"].item()
        grade = profile[profile.subNum == sub]["grade"].item()
        height = profile[profile.subNum == sub]["height"].item()
        weight = profile[profile.subNum == sub]["weight"].item()
        return gender, grade, height, weight
    else:
        return -1, -1, -1, -1
    

def get_IBSH_report(path_force, path_profile, isPlot):
    profile = pd.read_csv(path_profile)
    report = {}
    for sub in range(1, 14):
        gender, grade, height, weight = get_profile(profile, sub)
        report[f'sub{sub:03d}'] = {}
        sjs = []
        cmjs = []
        rfd_means = []
        rfd_maxs = []
        rsis = []
        for testNum in range(1, 13):
            # define type
            jumpType = []
            if testNum < 4:
                jumpType = 'sj'
            elif testNum < 10:
                jumpType = 'cmj'
            else:
                jumpType = 'dj'
            
            get_force = getForce(path_force, sub, testNum)
            force, time = get_force()
            
            if len(force) > 1:
                fe = forceplateEstimation(force, weight, jumpType,
                                          samp_rate, isPlot)
                if jumpType == 'sj':
                    sj = fe()
                    sjs.append(sj)
                elif jumpType == 'cmj':
                    cmj, rfd_mean, rfd_max = fe()
                    rfd_means.append(rfd_mean)
                    rfd_maxs.append(rfd_max)
                    cmjs.append(cmj)
                elif jumpType == 'dj':
                    rsi = fe()
                    rsis.append(rsi)
                plt.title(f'sub{sub:03d}')
        
        if sjs:
            report[f'sub{sub:03d}']["sj_max"] = max(sjs)
            report[f'sub{sub:03d}']["cmj_max"] = max(cmjs)
            report[f'sub{sub:03d}']["rfd_mean_max"] = max(rfd_means)
            report[f'sub{sub:03d}']["rfd_max_max"] = max(rfd_maxs)
            report[f'sub{sub:03d}']["rsi_max"] = max(rsis)
        else:
            report.pop(f'sub{sub:03d}')
        
    return pd.DataFrame.from_dict(report).T
    
def get_gold_table(path_force, path_profile, isPlot):
    profile = pd.read_csv(path_profile)
    gold_table = pd.DataFrame()
    gold_idx = 0
    for sub in range(1, 24):
        gender, grade, height, weight = get_profile(profile, sub)
        if grade < 0: # if sub doesn't in profile
            continue
        slj = profile[profile.subNum == sub]["slj"].item()
        sprint = profile[profile.subNum == sub]["50m"].item()
        
        for testNum in range(7, 10): # sport-science (1, 4), basketball : (7, 10)
            get_force = getForce(path_force, sub, testNum)
            force, time = get_force()
            
            if len(force) > 1:
                fe = forceplateEstimation(force, weight, 'sj',
                                          samp_rate, isPlot)
                cmj = fe.get_jump_height()
                
                gold_table.at[gold_idx, "subNum"] = sub
                gold_table.at[gold_idx, "testNum"] = testNum
                gold_table.at[gold_idx, "gender"] = gender
                gold_table.at[gold_idx, "grade"] = grade
                gold_table.at[gold_idx, "cmj"] = cmj
                gold_table.at[gold_idx, "slj"] = slj
                gold_table.at[gold_idx, "sprint_50"] = sprint
                gold_table.at[gold_idx, "height"] = height
                gold_table.at[gold_idx, "source"] = "IBSH"
                
                gold_idx += 1
    return gold_table
    
    
if __name__ == '__main__':
    import pandas as pd
    from PlotTool import Correlaiton
    
    path_force = '../data/IBSH/Force'
    path_profile = '../data/IBSH/IBSH_profile.csv'
    samp_rate = 100
    
    # report = get_IBSH_report(path_force, path_profile, 1)
    # report = report.round(2)
    gold_table = get_gold_table(path_force, path_profile, 1)
    
    # report.to_csv("data/IBSH/ibsh_report.csv")
    gold_table.to_csv(path_profile.split('_')[0] + '_gold.csv', index=False)
    
    gold_table = gold_table[gold_table["sprint_50"].notnull()]
    gold_last = gold_table.groupby(["subNum"]).last()
    plt_corr = Correlaiton()
    plt_corr(gold_last['cmj'], gold_last['sprint_50'])
    
    