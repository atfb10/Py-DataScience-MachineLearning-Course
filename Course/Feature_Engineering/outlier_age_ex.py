'''
Adam Forestier
April 10, 2023
Notes:
'''

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns

def create_ages(mean=50, standard_deviation=13, n_samples=100, seed=42):
    '''
    create a random distribution of ages
    '''
    np.random.seed(seed)
    return np.round(np.random.normal(loc=mean, scale=standard_deviation, size=n_samples), decimals=0)

# Deal w/ outliers

# Plot distribution and boxplot
sample = create_ages()
sns.displot(sample, bins=20, kde=True)
plt.show()
sns.boxplot(data=sample)
plt.show()

# Turn into a series
ser = pd.Series(sample)
print(ser.describe())
upper_quartile, lower_quartile = np.percentile(q=[75, 25], a=ser)
iqr = upper_quartile - lower_quartile
