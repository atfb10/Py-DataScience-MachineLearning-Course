'''
Adam Forestier
April 10, 2023
Notes:
'''

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns

df = pd.read_csv('Ames_Housing_Data.csv')

# label I want to predict
y = df['SalePrice']

# See correlation
sale_price_corr = df.corr()['SalePrice'].sort_values(ascending=False) # Overall qual & Gr liv Area are 1 and 2 most correlated
# print(sale_price_corr)
sns.scatterplot(data=df, x='Overall Qual', y='SalePrice')
# plt.show()
sns.scatterplot(data=df, x='Gr Liv Area', y='SalePrice', color='orange')
# plt.show()

# Get outliers
# It is ok to have expensive houses w/ great quality because it follows the trend
# It does not follow the trend to have homes with great quality and low selling points, this will screw up the model.
drop_index = df[(df['SalePrice'] < 300000) & (df['Gr Liv Area'] > 4000)].index
df = df.drop(drop_index, axis=0)
drop_index = df[(df['SalePrice'] < 200000) & (df['Overall Qual'] > 8)].index
df = df.drop(drop_index, axis=0)

# Re-plot
sns.scatterplot(data=df, x='Overall Qual', y='SalePrice')
# plt.show()
sns.scatterplot(data=df, x='Gr Liv Area', y='SalePrice', color='orange')
# plt.show()

# Save with outliers removed
df.to_csv('Ames_Housing_NoOutliers.csv')