'''
Adam Forestier
April 11, 2023
Notes:
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('Ames_Housing_Outliers_Missing_Handled.csv')

# MS SubClass - is numeric, but is ACTUALLY categorical. It has numbers assigned, but there is no ranking
df['MS SubClass'] = np.vectorize(str)(df['MS SubClass']) # Turn to a string so that is can be encoded

# Get object dataframe (object is equivalent to string in pandas) and numeric dataframes
object_df = df.select_dtypes(include='object')
numeric_df = df.select_dtypes(exclude='object')
object_dummies_df = pd.get_dummies(object_df, drop_first=True)
df = pd.concat([numeric_df, object_dummies_df], axis=1)
print(df.corr()['SalePrice'].sort_values())