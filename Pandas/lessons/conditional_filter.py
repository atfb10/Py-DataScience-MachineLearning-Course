'''
author: Adam Forestier
dates: March 22, 2023
notes:
    -
'''
import numpy as np 
import pandas as pd

# Read in
df = pd.read_csv('tips.csv')
reset_df = df

# Single condition Conditional filtering
is_lunch = df['time'] == 'Lunch' # Two line method. More readable, but another variable which is inefficient. Is lunch is a Pandas Series of booleans of whether or not the time column is "Lunch"
df = df[is_lunch] # Two line method. More readable, but another variable which is inefficient
df = df[df['tip'] > 4] # Get tips > $4.00 single line
df = df[df['smoker'] == 'Yes'] # Another single line example

# Multiple condition filter
df = reset_df
is_expensive = df['total_bill'] >= 30 # Multiple lines
is_male = df['sex'] == 'Male' # multiple lines
df = df[(is_expensive) & (is_male)] # multiple lines. AND

df = reset_df
df = df[(df['time'] == 'Dinner') | (df['size'] > 2)] # single line. OR

df = reset_df
options = ['Sat', 'Sun', 'Fri'] # awesome. return df if value in column "isin" list of columns
df =df[df['day'].isin(options)] # Single line option
options = [2, 1, 3] # isin multiple lines
is_small_table = df['size'].isin(options) # isin multiple lines
df = df[is_small_table] # isin multiple lines