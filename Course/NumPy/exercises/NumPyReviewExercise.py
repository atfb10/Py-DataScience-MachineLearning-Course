# 1. Import NumPy
import numpy as np

# 2. Create array of 10 zeros
zeros = np.zeros(10)

# 3. Create array of 10 ones
ones = np.ones(10)

# 4. Create an array of 10 fives
fives = np.full((10), 5)

# 5. Create an array of integers from 10 to 50
arr = np.arange(10, 51)

# 6. Create an array of even integers from 10 to 50
arr = np.arange(10, 51, 2)

# 7. Create a three by three matrix from 0 - 8
three_x_three = np.arange(0, 9).reshape(3, 3)

# 8. Create a 3 x 3 identity matrix
eye_matrix = np.eye(3).reshape(3, 3)

# 9. generate a random number between 0 to 1
n = np.random.rand()

# 10. array of 25 random numbers sampled from a standard normal distrubution
rand_std_dist = np.random.randn(25)

# 11. Matrix from 0.01 to 1
increment_by_tenth_matrix = np.arange(0, 1, .01).reshape(10, 10) + .01

# 12. array of 20 linearly spaced points between 0, 1
lin_space = np.linspace(0, 1, 20)

# Given this matrix... mat = np.arange(1,26).reshape(5,5)
mat = np.arange(1,26).reshape(5,5)

# 13. Get specified part of the matrix
sliced = mat[2:,1:]

# 14. Get 20
twenty = mat[3,4]

# 15. get specified part of the matrix
sliced = mat[0:3,1]

# 16. get specified part of the matrix
sliced = mat[4,:]

# 17. get specified part of the matrix
sliced = mat[3:]


# 18. sum of matrix
matrix_sum = mat.sum()

# 19. standard deviation of matrix
matrix_std_dev = mat.std()

# 20 sum of all columns in matrix
result = mat.sum(axis=0)
print(result)