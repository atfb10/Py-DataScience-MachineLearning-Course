'''
Author: Adam Forestier
Date: March 23, 2023
Notes
    - concatenation is pasting the two data sources together by column (they need the same index) or by rows by appending them
    - merge is analogous to a JOIN command in SQL
        * merge method takes in 1. tables you want merge 2. how - merge method. 3. on - column present in both dataframes to perform merge on
        * inner merge: the set of records that match in both tables
        * left merge: the set of records containing all records of the column the two tables are joined on. Pandas fills the missing values for left table rows on the columns merged from the right table with NaN
        * right merge: the set of records containing all records of the column the two tables are joined on. Pandas fills the missing values for right table rows on the columns merged from the left table with NaN
        * outer merge: the complete set of records containing all records of what the two tables are joined on. Pandas fills missing values with NaN
'''

import numpy as np
import pandas as pd 

# concatanation
d1 = {'A': ['A0', 'A1', 'A2', 'A3', 'A4'], 'B': ['B0', 'B1', 'B2', 'B3', 'B4']}
d2 = {'C': ['C0', 'C1', 'C2', 'C3', 'C4'], 'D': ['D0', 'D1', 'D2', 'D3', 'D4']}
d3 = {'A': ['A5', 'A6', 'A7', 'A8', 'A9'], 'B': ['B5', 'B6', 'B7', 'B8', 'B9']}


df1 = pd.DataFrame(d1)
df2 = pd.DataFrame(d2)
df3 = pd.DataFrame(d3)

# concatenation
pd.concat([df1, df2], axis=1) # join along columns
# print(df)
pd.concat([df1, df3], axis=0) # join by appending rows. need the same column names for it to be a useful operation
# print(df)

# Inner merge
reg = pd.DataFrame({'reg_id': [1,2,3,4], 'name': ['Andrew', 'Bob', 'Claire', 'Keith']})
login = pd.DataFrame({'login_id': [1,2,3,4], 'name': ['Jeff', 'Claire', 'Reginold', 'Bob']})
df = pd.merge(reg, login, how='inner', on='name')
print('inner merge')
print(df)
print('-------------')

# left merge
df = pd.merge(reg, login, how='left', on='name')
print('left merge')
print(df)
print('-------------')

# left merge
df = pd.merge(reg, login, how='right', on='name')
print('right merge')
print(df)
print('-------------')

# outer merge
df = pd.merge(reg, login, how='outer', on='name')
print('outer merge')
print(df)
print('-------------')

# Merge on index
reg = reg.set_index('name')
df = pd.merge(reg, login, how='inner', left_index=True, right_on='name') # join on left index and right column 'name'
login = login.set_index('name')
df = pd.merge(reg, login, how='inner', left_index=True, right_index=True) # join on both indexes
reg = reg.reset_index()

# merge on if column names are different but represent the same data 
reg.columns = ['reg_name', 'reg_id']
df = pd.merge(reg, login, on='inner', left_on='reg_name', right_on='name')