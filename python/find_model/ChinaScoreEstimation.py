# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 13:33:03 2021

@author: Datong
"""
import pandas as pd

class chinaScoreEstimation():
    '''
    get score from china criteria.
    '''
    
    def __init__(self):
        self.slj_Female_criteria = pd.read_csv('../score_table/slj_Female_scoreSheet.csv')
        self.slj_Male_criteria = pd.read_csv('../score_table/slj_Male_scoreSheet.csv')
        self.sprint_Female_criteria = pd.read_csv('../score_table/sprint_Female_scoreSheet.csv')
        self.sprint_Male_criteria = pd.read_csv('../score_table/sprint_Male_scoreSheet.csv')
        
    def __call__(self, gender, grade, EXtype, EX_performance):
        scores_criteria = self.get_criteria(gender, grade, EXtype)
        score = self.get_score(scores_criteria, EX_performance)
        return score
    
    def get_criteria(self, gender, grade, EXtype):
        '''
        use profile to get criteria of sepecific exam

        Parameters
        ----------
        gender : int
           F: female
           M: male
           
        grade : int, 1-12
            
        EXtype : string
            slj, sprint

        Returns
        -------
        scores_criteria : pd.series

        '''
        if gender == 'F' and EXtype == 'slj': 
            scores = self.slj_Female_criteria['score']
            scores_criteria = self.slj_Female_criteria[str(grade)]
        elif gender == 'M' and EXtype == 'slj': 
            scores = self.slj_Male_criteria['score']
            scores_criteria = self.slj_Male_criteria[str(grade)]
        elif gender == 'F' and EXtype == 'sprint':  
            scores = self.sprint_Female_criteria['score']
            scores_criteria = self.sprint_Female_criteria[str(grade)]
        elif gender == 'M' and EXtype == 'sprint': 
            scores = self.sprint_Male_criteria['score']
            scores_criteria = self.sprint_Male_criteria[str(grade)]
        else:
            print('no such exam type!')
            
        scores_criteria.index = scores
        return scores_criteria
            
    def get_score(self, scores_criteria, EX_performance):
        '''
        use scores_criteria from get_criteria() to get exam scores of performance

        Parameters
        ----------
        scores_criteria : pd.series
            
        EX_performance : float, slj or 50m performance

        Returns
        -------
        score_output : int, exam scores

        '''
        Threshold = scores_criteria.iloc[0]
        if scores_criteria.iloc[0] > scores_criteria.iloc[-1]:
            i = -1       
            while EX_performance < Threshold:
                i += 1
                Threshold = scores_criteria.iloc[0+i]
                if EX_performance >= Threshold:
                    score_output = scores_criteria.index[i]
                    break
                if EX_performance < scores_criteria.iloc[-1]:
                    score_output = 0
                    break
            else:  score_output = 100 
        else:
            i = 0       
            while EX_performance > Threshold:
                i += 1
                Threshold = scores_criteria.iloc[0+i]
                if EX_performance <= Threshold:
                    score_output = scores_criteria.index[i]
                    break
                if EX_performance > scores_criteria.iloc[-1]:
                    score_output = 0
                    break
            else:  score_output = 100
            
        return score_output
    #%%
    
if __name__ == '__main__':
    gold_table = pd.read_csv('../dataset_youth/gold_record.csv')
    cse = chinaScoreEstimation()
    
    for i in range(len(gold_table)):
        sub = gold_table.iloc[i]['subNum']
        gender = gold_table.iloc[i]['gender']
        grade = gold_table.iloc[i]['grade']
        slj = gold_table.iloc[i]['slj']
        sprint_50 = gold_table.iloc[i]['sprint_50']
        
        gold_table.at[i, "slj_score"] = cse(gender, grade, 'slj', slj)
        gold_table.at[i, "sprint_score"] = cse(gender, grade, 'sprint', sprint_50)
        