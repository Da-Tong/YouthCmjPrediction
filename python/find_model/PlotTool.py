import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

class BlankAltman():
    
    def __init__(self):
        return 
        
        
    def __call__(self, xData, yData):
        '''
        need pd series to get column name

        Parameters
        ----------
        xData : pd series
            
        yData : pd series

        Returns
        -------
        None.

        '''
        self.xData = xData
        self.yData = yData
        
        ax = self.setFigure()
        self.x = np.mean([self.xData, self.yData], axis=0)
        self.y = self.yData - self.xData
        self.get_confidence_interval()
        ax.plot(self.x, self.y, 'ko', ms=8)
        ax.hlines(self.upper_bound, np.min(self.x), np.max(self.x),
                  linestyles='dashed', color='black', lw=2)
        ax.hlines(self.lower_bound, np.min(self.x), np.max(self.x),
                  linestyles='dashed', color='black', lw=2)
        
        ax.text(self.text_x, self.text_y_lower,
                str(self.lower_bound) + " (-1.96SD)", fontsize=15)
        ax.text(self.text_x, self.text_y_upper,
                str(self.upper_bound) + " (+1.96SD)", fontsize=15)
        
        
    def setFigure(self):
        '''
        set figure detail

        Returns
        -------
        ax : pyplot handle

        '''
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        
        ax.spines["bottom"].set_linewidth(2.5)
        ax.spines["left"].set_linewidth(2.5)
        
        ax.set_xlabel("Average", size=28)
        ax.set_ylabel(self.yData.name + ' - ' + self.xData.name, size=28)
        
        figName = self.xData.name + " vs. " + self.yData.name
        ax.tick_params(direction='out', width=2, labelsize=20)
        ax.set_title(figName, size=32, weight='bold')
        return ax 
    
    def get_confidence_interval(self):
        self.text_x = np.max(self.x)
        
        self.median_line = np.mean(self.y)
        self.upper_bound = self.median_line + 1.96 * np.std(self.y)
        self.lower_bound = self.median_line - 1.96 * np.std(self.y)
        
        self.text_media =  np.mean(self.y) + 0.01 * np.std(self.y)
        self.text_y_upper = self.median_line + 2.1 * np.std(self.y)
        self.text_y_lower = self.median_line - 2.1 * np.std(self.y)
        

class Correlaiton():
    
    def __init__(self):
        return
        
    def __call__(self, xData, yData, isSave=0):
        '''
        plot scatter figure and regression line with r square value
        
        need pd series to get column name
        
        Parameters
        ----------
        xData : pd series
            
        yData : pd series

        Returns
        -------
        None.

        '''
        self.xData = xData
        self.yData = yData
        
        self.get_regression_information()
        ax = self.setFigure()
        self.plot_data(ax)
        
        if isSave:
            plt.savefig(self.figName + '.png')
        
    def setFigure(self):
        '''
        set figure detail

        Returns
        -------
        ax : pyplot handle

        '''
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        
        ax.spines["bottom"].set_linewidth(2.5)
        ax.spines["left"].set_linewidth(2.5)
        
        ax.set_xlabel(self.xData.name, size=28)
        ax.set_ylabel(self.yData.name, size=28)
        
        self.figName = self.xData.name + " vs. " + self.yData.name
        ax.tick_params(direction='in', width=2, labelsize=20)
        ax.set_title(self.figName, size=32, weight='bold')
        return ax
        
    def get_regression_information(self):
        '''
        get slope and intersection of linear regression
        
        get text position

        Returns
        -------
        None.

        '''
        self.a, self.b = np.polyfit(self.xData, self.yData, 1)
        self.r_square = pearsonr(self.xData, self.yData)[0] ** 2
        x_min = np.min(self.xData)
        x_max = np.max(self.xData)
        y_min = np.min(self.yData)
        y_max = np.max(self.yData)
        self.x = np.linspace(x_min, x_max, 50)
        self.text_x = x_min + (x_max - x_min) / 5
        self.text_y = y_min + (y_max - y_min) *0.9
        
    def plot_data(self, ax):
        '''
        plot scatter with setted plot handle

        Parameters
        ----------
        ax : handle set by self.setFigure()

        Returns
        -------
        None.

        '''
        ax.plot(self.xData, self.yData, 'ko', ms=10)
        ax.plot(self.x, self.a * self.x + self.b, 'r--', lw=2)
        ax.text(self.text_x, self.text_y, f"R square: {round(self.r_square, 3)}",
                fontsize=24, color='r')
        
class BarChart():
    
    def __init__(self):
        return 
        
        
    def __call__(self, xData, yData):
        '''
        plot scatter figure and regression line with r square value
        
        need pd series to get column name
        
        Parameters
        ----------
        xData : pd series
            
        yData : pd series

        Returns
        -------
        None.

        '''
        self.xData = xData
        self.yData = yData
        
        self.get_bar_information()
        ax = self.setFigure()
        self.plot_data(ax)
        
    def setFigure(self):
        '''
        set figure detail

        Returns
        -------
        ax : pyplot handle

        '''
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        
        ax.spines["bottom"].set_linewidth(2.5)
        ax.spines["left"].set_linewidth(2.5)
        
        ax.set_xlabel(self.xData.name, size=28)
        ax.set_ylabel(self.yData.name, size=28)
        
        figName = self.xData.name + " vs. " + self.yData.name
        ax.tick_params(direction='in', width=2, labelsize=20)
        ax.set_title(figName, size=32, weight='bold')
        return ax
        
    def get_bar_information(self):
        '''
        get bar information

        Returns
        -------
        None.

        '''
        pass
        
    def plot_data(self, ax):
        '''
        plot scatter with setted plot handle

        Parameters
        ----------
        ax : handle set by self.setFigure()

        Returns
        -------
        None.

        '''
        pass
        

if __name__ == "__main__":
    path = 'data/model.csv'
    data = pd.read_csv(path)
    
    x_labels = ["jump_height"]
    y_labels = ["slj"]
    
    x_label = 0
    y_label = 0
    
    xData = data[x_labels[x_label]]
    yData = data[y_labels[y_label]]
    
    xData = xData[yData.notnull()]
    yData = yData[yData.notnull()]
    
    plt_corr = Correlaiton()
    plt_corr(xData, yData, 1)
    
    # blan_altman = BlankAltman()
    # blan_altman(xData, yData)


