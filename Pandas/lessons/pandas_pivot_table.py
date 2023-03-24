'''
Author: Adam Forestier
Date: March 24, 2023
Notes
    - a dataframe with repeated values can be pivoted for reorginzation and clarity
    - choose columns to define the index, columns and values: df.pivot(index='foo', columns='bar', values='baz'): the pivoted index AND columns should have repeated values
    - no new information is shown, it is merely reorganized
'''

import numpy as np
import pandas as pd

df = pd.read_csv('Sales_Funnel_CRM.csv')
print(df.head())
print('-----------------')

# Pivot example 
licenses = df[['Company', 'Licenses', 'Product']]
licenses_pivoted = pd.pivot(data=licenses, index='Company', columns='Product', values='Licenses') # This creates a DataFrame where companies are index, the columns are the product typs and values are the quantity of licenses sold to the company by license type 
print(licenses_pivoted.head())
print('-----------------------------------------------------------------------')

# Pivot table example: Similar to pivot, but it performs an aggregate function in addtition
sales = pd.pivot_table(df, index='Company', aggfunc='sum', values=['Licenses', 'Sale Price'])
print(sales.head())
print('-----------------------------------------------------------------------')

# Pivot table example 2 - group by acc manager and contact. get total sale price for each account manager by their contact. replace NaN values with 0
acc_mng_pivot_table = pd.pivot_table(data=df, index=['Account Manager', 'Contact'], values='Sale Price', aggfunc='sum', fill_value=0)
print(acc_mng_pivot_table)