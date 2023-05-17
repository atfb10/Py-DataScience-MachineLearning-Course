'''
author: Adam Forestier
date: March 22, 2023
notes

'''
import numpy as np 

myarr = np.arange(0, 11)

# Get element 9
myarr[9]

# get elements the second to 4th element. [4] mpt included
myarr[1:4]

# Get elements 5 and beyond
myarr[5:]

# Get elements up to but not including 4th element
myarr[:3]

# Powerful. Cannot do this in normal python lists. broadcast change across the list
myarr[:4] = 100 # This will set elements 0, 1, 2, 3, 4 to 100 AND it applies the changes the myarr variable without having to reassin. This is epic!

# Dangerous though... changes to arrays formed from original array, will effect original array
my_slice = myarr[:4]
my_slice[:] = 99 # This will change every element in my_slice to be 99. It WILL ALSO CHANGE ELEMENTS 0-4 TO 99 IN THE ORIGINAL ARRAY. This is due to pointers
print(myarr)

# How to fix... use copy function
myarr = np.arange(0, 9)
myarr_copy = myarr.copy()
my_slice = myarr[:4]
my_slice[:] = 99 # This will now no longet effect the variable the slice was taken from
print(myarr)

arr_2d = np.array([[2, 4, 1], [432, 2, 1], [99, 100, 2]])
# print(arr_2d)

# grab a single row
arr_2d[0] # returns first row

# Grab 432
arr_2d[1,0] # Could also do this using two bracket [1][0] notation, but this looks cleaner

# Get slice of matrix. I want 432, 2, 99, 100
print(arr_2d[1:,:2])

# Conditional selection
arr = np.arange(0, 11)

# This is absolutely! Show which elements are greater than 4
print(arr > 4)

# Filter using conditionals. This is magic
boolean_arr = arr > 4
new_arr = arr[boolean_arr]
print(new_arr)

# in a single line
newest_arr = arr[arr < 2]
print(newest_arr)