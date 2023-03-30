'''
Author: Adam Forestier
Date: March 29, 2023
Notes:
    - seaborn grids are anagolous to Matplotlib subplots() to automatically create a grid based off of a categorical column
    - Instead of passing a specific number of columns or rows, one simply passes in the name of the column and seaborn with automatically map the subplots grid!
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('StudentsPerformance.csv')

# Example
sns.catplot(data=df, x='gender', y='math score', kind='box', row='lunch', col='test preparation course')
plt.show()

# PairGrid Example - customize pairplot
g = sns.PairGrid(df, hue='gender')
g = g.map_upper(sns.scatterplot)
g = g.map_diag(sns.histplot)
g = g.map_lower(sns.kdeplot)
g = g.add_legend()
plt.show()