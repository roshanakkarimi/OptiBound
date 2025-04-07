import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np 

class StatUtils:
    def __init__(self, sample:pd.DataFrame):
        self.sample = sample

    def nd_hist_plot(self,  color):
        sns.histplot(self.sample, kde=True, stat="count", color=color, alpha=0.7)
        plt.title("Distribution of Feature")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.show()

    def param_stat(self, param:pd.Series):
        return f'mean:{np.mean(param)}, std: {np.std(param)}'
