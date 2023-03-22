
'''
author: Adam Forestier
date: March 22, 2023
notes:
    - weird... with divide by 0 in NumPy - does not error out, just provides an err
'''

import numpy as np

arr = np.arange(0, 10)

# This will perform on an element by element basic. Multiple 2 to each element
arr * 2

# Math functions on element by element basis
arr = np.arange(1, 10)
np.sin(arr)
np.sqrt(arr)
np.log(arr)

# summary stats
arr.sum()
arr.mean()
arr.max()
arr.var() # variance
arr.std() # standard deviation

# 2d array 
two_d = np.arange(0, 25).reshape(5, 5)

# perform sum across rows - AKA, sum of the columns
print(two_d.sum(axis=0))

# perform sum across columns - AKA, sum of the rows
print(two_d.sum(axis=1))