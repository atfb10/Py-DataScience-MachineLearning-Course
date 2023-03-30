'''
Author: Adam Forestier
Date: March 29, 2023
Notes:
    - Example: Distribution of salaries per level of level of education
    - Types of figures:
        * Boxplot
            > 25% of plots in first quartile (Q1)
            > 50% in Inter Quartile Range (IQR)
            > 25% in  third quartile (Q3)
            > Whiskers are definited by IQR x 1.5
            > Anything outside of thse are outliers
            > Line in middle is median
            > A boxplot can be made per category
        * Violinplot
             > display probabiity density across the data using KDE
             > can be made per category
        * Swarmplot
             > Shows all data points in the distribution
             > Can be made per category
             > I find this highly useful - it actually shows the number of data points per category - unlike the violin and boxplots
        * Boxenplot (Letter-Value Plot)
             > Developed recently: 2011
             > Using a system of letter-value quantiiles to display against a standard boxplot. Kind of blends KDE and box plot
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('StudentsPerformance.csv')
print(df.head())

# Box plot
plt.figure(figsize=(8, 8))
sns.boxplot(data=df, y='math score', x='test preparation course')
plt.show()

# Box plot
sns.boxplot(data=df, x='parental level of education', y='reading score', palette='Pastel1', hue='lunch')
plt.legend(bbox_to_anchor=(1.05,0.5))
plt.show()

# Violin plot
sns.violinplot(data=df, y='writing score', x='gender')
plt.show()

# Swarm plot
sns.swarmplot(data=df, x='race/ethnicity', y='math score', hue='lunch', size=3, dodge=True) # I really like the swarmplot
plt.show()