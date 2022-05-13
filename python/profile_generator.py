# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 11:00:53 2022

@author: newgr
"""

import pandas as pd
import numpy as np
import os

from find_model.ForcePlateEstimation import getForce, forceplateEstimation

def fill_ft_profile(data, profile, source):
     # fill profile
    data['cmj'] = 9.8 * ((data['ft'] / 1000) ** 2) / 8 * 100
    for i in range(len(data)):
        sub = int(data.iloc[i]['subNum'])
        
        data.at[i, 'age'] = profile[profile.subNum == sub]['age'].item()
        data.at[i, 'gender'] = profile[profile.subNum == sub]['gender'].item()
        data.at[i, 'height'] = profile[profile.subNum == sub]['height'].item()
        data.at[i, 'weight'] = profile[profile.subNum == sub]['weight'].item()
        data.at[i, 'grade'] = profile[profile.subNum == sub]['grade'].item()
        data.at[i, 'slj'] = profile[profile.subNum == sub]['slj'].item()
        data.at[i, 'sprint_50'] = profile[profile.subNum == sub]['sprint_50'].item()
        data.at[i, 'source'] = source
        
    data = data[['subNum', 'testNum', 'age', 'gender', 'height', 'weight', 'grade',
             'cmj', 'slj', 'sprint_50', 'source']]
    return data

def get_jh_table_by_force(dataPath, source):
    subs = os.listdir(os.path.join(dataPath, source, 'Force'))
    testNums = [1, 2, 3]
    if source == 'IBSH':
        testNums = [7, 8, 9]
    profile = pd.read_csv(
        os.path.join(dataPath, 'ALL_Profile', source, f'{source}_profile.csv'))
    
    jh_table = {
        'subNum':[],
        'testNum':[],
        'age':[],
        'gender':[],
        'height':[],
        'weight':[],
        'grade':[],
        'cmj':[],
        'slj':[],
        'sprint_50':[],
        'source':[],
        }
    
    for sub in subs:
        subNum = int(sub[3:])
        age = profile[profile['subNum'] == subNum]['age'].item()
        gender = profile[profile['subNum'] == subNum]['gender'].item()
        height = profile[profile['subNum'] == subNum]['height'].item()
        weight = profile[profile['subNum'] == subNum]['weight'].item()
        grade = profile[profile['subNum'] == subNum]['grade'].item()
        slj = profile[profile['subNum'] == subNum]['slj'].item()
        sprint_50 = profile[profile['subNum'] == subNum]['50m'].item()
        
        for testNum in testNums:
            read_force = getForce(dataPath, subNum, testNum) # init no use
            force, time = read_force.get_force(
                os.path.join(dataPath, source, 'Force', sub, f't{testNum:03d}.txt'))
            force, time = read_force.cut_force(force, time)
            
            
            if len(force) > 1:
                fe = forceplateEstimation(np.array(force), weight, 'sj', 100, 0)
                cmj = fe()
            
            jh_table['subNum'].append(subNum)
            jh_table['testNum'].append(testNum)
            jh_table['age'].append(age)
            jh_table['gender'].append(gender)
            jh_table['height'].append(height)
            jh_table['weight'].append(weight)
            jh_table['grade'].append(grade)
            jh_table['cmj'].append(cmj)
            jh_table['slj'].append(slj)
            jh_table['sprint_50'].append(sprint_50)
            jh_table['source'].append(source)
            
    return pd.DataFrame.from_dict(jh_table)

if __name__ == '__main__':        
    dataPath = 'dataset_youth'
    proPath = 'dataset_youth/ALL_Profile'
    
    source_ft_list = ['GRJH', 'NTSH_pre', 'YCSH_post', 'NTHU_pilot_post']
    source_force_list = ['IBSH', 'IBSH_SS']
    
    profile = {
        'subNum':[],
        'testNum':[],
        'age':[],
        'gender':[],
        'height':[],
        'weight':[],
        'grade':[],
        'cmj':[],
        'slj':[],
        'sprint_50':[],
        'source':[],
        }
    profile = pd.DataFrame.from_dict(profile)
    
    for source_ft in source_ft_list:
        profile_ft = pd.read_csv(
            os.path.join(proPath, source_ft, f'{source_ft}_profile.csv'))
        data_ft = pd.read_csv(
            os.path.join(proPath, source_ft, f'{source_ft}_ft.csv'))
        profile_tmp = []
        
        profile_tmp = fill_ft_profile(data_ft, profile_ft, source_ft)
        profile = pd.concat([profile, profile_tmp])
        
    for source_force in source_force_list:
        profile_tmp = []
        profile_tmp = get_jh_table_by_force(dataPath, source_force)
        profile = pd.concat([profile, profile_tmp])
    
    print(f'SLJ statistic\n{profile["slj"].describe()}\n')
    print(f'sprint_50 statistic\n{profile["sprint_50"].describe()}')
    
    profile.to_csv('gold_record.csv', index=False)