'''
Author: Adam Forestier
Date: March 22, 2023
Notes:

'''

import numpy as np

# Create numpy array
my_list = [1,2,3]
my_array = np.array(my_list)

# Matrix with numpy array
my_matrix = [[1,2,3], [4,5,6], [7, 8, 9]]

# array from 0-9
my_array = np.arange(0, 10)

# create 2 rows, of 5 cols of zeros
np.zeros((2,5))

# Evenly spaced numbers over a range for a specified number of numbers
np.linspace(1, 100, 10)
np.linspace(0, 5, 21)

# Identity matrix
np.eye(5)

# Random uniform distribution between 0 and 1
np.random.rand(1)

# Matrix
np.random.rand(2, 4)

# standard normal distribition matrix. standard normal distribution. #'s closer to 0 is more likely than farther away
print(np.random.randn(10, 2))

# Random int
np.random.randint(0, 101, 10) # 0 to 100 inclusive, select 10 numbers

# Set seed. allows you to get a particular set of random numbers
np.random.seed(42) # arbitrary random seed selection in Python is 42 (this is a funny reference to "Hitchhiker's guide to the Galaxy"). Numb
np.random.rand(4) # You will get the same 4 numbers back in the same program - until setting a new seed with a new arbitrary number

# Turn an array into a matrix. Matrix must have same element count as array
arr = np.arange(0, 25)
arr.reshape(5,5)
arr.shape # Show shape of matrix

# Find max and min
ranarray = np.random.randint(0, 101, 10)
ranarray.min()
ranarray.max()
ranarray.argmax() # location of max int in array
ranarray.argmin() # location of min int in array

# Show type of elements in aray
ranarray.dtype