'''
Author: Adam Forestier
Date: March 22, 2023
Notes
    - Dataframe: group of Pandas Series objects that share the same index
                 a table of columns & rows that can be restructured and refiltered
'''

import numpy as np
import pandas as pd

# create dataframe
np.random.seed(101)
data = np.random.randint(0, 101, (4, 3)) # range of possible values is 0 - 100. 4 rows by 3 columns
index = ['CA', 'NY', 'AZ', 'TX']
cols = ['Jan', 'Feb', 'Mar']
df = pd.DataFrame(data=data, index=index, columns=cols)

# Read in csv
df = pd.read_csv('tips.csv')

# high level method calls
df.columns # column names
df.index # index
df.head() # Shows first 5 rows by default. Can pass in specific amount
df.tail() # Shows last 5 rows by default. can pass in specific amount
df.info() # General information about df
df.describe() # Calculates generic statistical information on the numeric data columns. Awesome!
df.describe().transpose() # Flips the columns and rows (aka will show the statistics as columns and the table headers as rows)

# Working with columns
df['total_bill'] # Get just total bill column
df['total_bill'] + 5 # Perform operation on bill column
df[['total_bill', 'tip']] # Get 2 columns (pass in list of column names)
df['tip_percentage'] = round(100 * df['tip'] / df['total_bill'], 2) # Create new column.
df['total_cost'] = df['tip'] + df['total_bill'] # Create new column 2.
df['dumb'] = 0
df = df.drop('dumb', axis=1) # Remove a column. If you do not reassign OR set inplace parameter = True, it does not actually remove the column

# Working with rows
df = df.set_index('Payment ID') # Set index to a column. Must reassign variable
df = df.reset_index() # Reset index to increment of 1
df = df.set_index('Payment ID') # Set index to a column. Must reassign variable
df.iloc[102] # Get a single row by location
df.loc['Sun2959']# Get a single row by index
df.iloc[23:44] # get elements 23-43
df.loc[['Sun2959', 'Sun5260']] # get multiple elements by index
df = df.drop('Sun2959', axis=0) # Drop a row. Must reassign to make this permanent
df = df.iloc[1:] # Get rid of the first row by selecting the second row onwards
one_row = df.iloc[0] 
df = df.append(one_row) # Insert row. Inserting a dictionary, where the key is equal to the column names, and the value is equal to whatever values associated with the key