'''
author: Adam Forestier
date: march 22, 2023
notes:
    series is a data structure for Pandas that holds an array of information ALONG with a named index
    Pandas series adds on a labeled index - data is still numerically organized. Have access to numeric and named index
    Pandas series are build off of NumPy arrays - so operations can be broadcast
'''

import numpy as np
import pandas as pd

index = ['USA', 'Canada', 'Mexico']
data = [1776, 1867, 1821]
my_series = pd.Series(data, index)

# Access data by both named and location index
my_series[0] # 1776
my_series['USA'] # 1776

# Create Panda series from dictionary
d = {
    'Sam': 5,
    'Fred': 3, 
    'Jeff': 32
}

# Operations
q1 = pd.Series({'Japan': 80, 'China': 450, 'India': 200, 'USA': 250})
q1_with_brazil = pd.Series({'Brazil': 80, 'China': 450, 'India': 200, 'USA': 250})
q2 = pd.Series({'Brazil': 100, 'China': 500, 'India': 210, 'USA': 260})

# Show keys 
q1.keys()

# broadcasting
print(q1/100)
print(q1)

# If series are identical
print(q1_with_brazil + q2)

# if sales are not identical
print(q1 + q2) # by default Pandas puts NaN if element is not present in both series

print(q1.add(q2, fill_value=0)) # Handle when element is not in both lists. just set the fill value!
