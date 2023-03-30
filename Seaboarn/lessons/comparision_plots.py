'''
Author: Adam Forestier
Date: March 29, 2023
Notes:
    - 2 dimensional versions of other plots
    Types
        jointplot - histogram and scatterplot 2d plot. Challenging to read
        pairplot - quick way to compare all numerical columns in a datframe. Very useful - I like
                 - CPU and RAM intensive. Filter down to columns you want
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('StudentsPerformance.csv')

# jointplot
sns.jointplot(data=df, x='math score', y='reading score', kind='hex')
plt.show()

# Pairplot - good stuff
sns.pairplot(data=df)
plt.show()

# pairplot with hue
sns.pairplot(data=df, hue='lunch')
plt.show()