'''
Author: Adam Forestier
Date: March 28, 2023
Notes:
    - categorical plots - display statistical metrics per category (i.e. mean value per category or a count of the number of rows per category)
    - 2 main types: 
        1. countplot - counts number of rows per category in the y-axis
        2. barplot - general form of displaying any chosen metric per category in the y-axis
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('dm_office_sales.csv')
print(df.head())

# Countplot example
plt.figure(figsize=(10,4))
sns.countplot(x='division', data=df, palette='Set2')
plt.show()

# Countplot example 2
sns.countplot(x='level of education', data=df, hue='training level', palette='Set3') # hue allows for another layer of data to be displayed 
plt.show()

# bar plot example
sns.barplot(data='df', x='level of education', y='salary', estimator=np.mean, ci='sd') # estimator is function to graph on the y axis. ci is the confidence interval. Standard deviation often makes the most sense
plt.show() # NOTE: Worth noting it is often better to show metric per category in a simple table. I.E. it is easier to read the average salary as text, than to view a filled in bar