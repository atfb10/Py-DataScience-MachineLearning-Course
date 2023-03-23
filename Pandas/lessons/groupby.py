'''
author: Adam Forestier
dates: March 23, 2023
notes:
    - groupby() operation allows us to examine data on a per category
    - Categorical columns are non-continous
    - steps
        * seperate out categorical columns and their associated data value
        * perform aggregate function
'''

import numpy as np
import pandas as pd

df = pd.read_csv('mpg.csv')
reset_df = df

# Categorical columns: cylinders, model_year
df['model_year'].nunique() # See how many unique model years
df['cylinders'].nunique() # See how many unique cylinders

# Groupy
# single group
df.groupby('model_year').mean() # Change model year to index. Then takes the average of each numerical column and returns only the numerical columns
df.groupby('model_year').describe().transpose() # basic stats for groupby
df.groupby('model_year').mean()['mpg'] # for 1 column
model_year_stats_df = df.groupby('model_year').describe()
model_year_stats_df.loc[70] # Get only for model year 70
model_year_stats_df.loc[[70, 80]] # Get only for model year 70 and 80

# Multiple group
df.groupby(['model_year', 'cylinders']).mean() # groupby by 2 categoriacal columns - it creates a multilevel index. check out below how to use multilevel index
yearly_cyl = df.groupby(['model_year', 'cylinders']).describe()
yearly_cyl.index.names # show index names
yearly_cyl.index.levels # Show index values
yearly_cyl.loc[82] # Returns only for index of model_year = 82
 # pass a tuple in .loc() if you wish to return only by a multilevel index
yearly_cyl.loc[(80, 4)] # This will return only cars from 1980 with 4 cylinders as a series
yearly_cyl.xs(key=75, level='model_year') # Returns a cross section of data. Here - cars from 75
yearly_cyl.xs(key=4, level='cylinders') # another example. 4 cylinder vehicles. this is great because you can select which level of index you would like to filter by
yearly_cyl.swaplevel() # swap indexes
yearly_cyl.sort_index(level='model_year', ascending=False) # sort by index (usually wish to sort by outmost index level)


# Always do filter first, then do aggregate function on groupby
# Group by model_year and cylinders; with only 6 and 8 cylinder vehicles
df = reset_df
model_year_6_8_cylinder_df = df[df['cylinders'].isin([6, 8])].groupby(['model_year', 'cylinders']).describe()

# Choose aggregate function using agg. very powerful
df.agg('std') # arguments: string of function columns. here is an example of standard deviation
df.agg(['std', 'mean']) # can pass a list of string names for function calls
df.agg(['std', 'mean'])['mpg'] # function calls for only mpg

# Holy cow this is powerful! You can pass in a dictionary into agg() where the key is the column, and the value is a list of function calls you wish to perform on that column: the key
# Of note: for the functions called on one column, but not on another, it will will that cell with NaN. View the print of the_magic variable in the terminal
mydict = {
    'mpg': ['max', 'mean'],
    'weight': ['mean', 'std']
}
the_magic = df.agg(mydict)
print(the_magic)