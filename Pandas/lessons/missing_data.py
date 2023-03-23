'''
author: Adam Forestier
dates: March 23, 2023
notes:
    - Pandas display missing values as NaN
    - pd.NaT imply the missing time should be a timestamp
'''

import numpy as np
import pandas as pd

df = pd.read_csv('movie_scores.csv')
reset_df = df

df.isnull()

# Null filtering
df[df['pre_movie_score'].notnull()] # Gets not null values for the pre_movie_score column
df[(df['pre_movie_score'].isnull()) & (df['first_name'].notnull())] # Multiple null conditions.

# Delete null data
df.dropna() # Drop all rows that has any null value
df.dropna(thresh=3) # Drop all rows that has less than 3 actual values
df.dropna(axis=1, thresh=2) # drop all columns that has than 2 actual values
df.dropna(subset=['last_name']) # Drop any row that does not have a value for last name

# Replace null data
df.fillna("New Value") # fill na values with New Value
df['pre_movie_score'].fillna(0) # Fill only specfic columns with data
df['pre_movie_score'].fillna(df['pre_movie_score'].mean()) # Fill only specfic columns with data by using a method/function

ser = pd.Series({
    'first': 100,
    'business': np.nan,
    'economy-plus': 50,
    'economy': 30
})

ser = ser.interpolate() # Fill with linear interpolation - based on value above and value below