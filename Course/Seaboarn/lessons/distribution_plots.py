'''
Author: Adam Forestier
Date: March 28, 2023
Notes:
    - Distribtuion Plots - display a single continuous feature and help visualize properties such as deviation and average values
    - 3 types: Rug Plot, Histogram, KDE (kernal density estimation) plot
        Rug: single tick per value along the x-axis. No y-axis
        Histogram: count number of ticks for each value or value range. y is the count or percent
        Kernal Density Estimation: a method of estimating a probabilty density function of a random variable: it is a way of estimating a continuous probability curve for a finite data sample
            - most commonly use Gaussian distribution; normal distribtuion curve to display continuous probability
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('dm_office_sales.csv')
print(df.head())

# Rugplot example
plt.figure(figsize=(8, 4))
sns.rugplot(x='salary', data=df, height=.5)
plt.title('Salary Rug')
plt.show()

# Histogram example
sns.set(style='darkgrid')
sns.displot(x='salary', data=df, bins=25, color='#34d2eb', edgecolor='red', linewidth=2) # bins is number of columns. **kwargs in seaborn is all of the arguments available in matplotlib
plt.title('Salary Distribution')
plt.savefig('D:\\coding\\udemy\\python_datascience_machine_learning\\Seaboarn\\img\\dist1.jpg')
plt.show()

# histogram with kernal density estimation drawn over using normal distribution
sns.set(style='whitegrid')
sns.displot(x='sales', data=df, color='orange', bins=20, kde=True)
plt.title('Sales with KDE')
plt.savefig('D:\\coding\\udemy\\python_datascience_machine_learning\\Seaboarn\\img\\dist2.jpg')
plt.show()

# KDE example
np.random.seed(42)
sample_ages = np.random.randint(0, 100, 200)
df = pd.DataFrame(data=sample_ages, columns=['ages'])
sns.kdeplot(x='ages', data=df, clip=[0, 100], fill=True) # clip makes the range 0 to 100
plt.savefig('D:\\coding\\udemy\\python_datascience_machine_learning\\Seaboarn\\img\\dist3.jpg')
plt.show()