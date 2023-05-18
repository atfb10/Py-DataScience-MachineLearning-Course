'''
Adam Forestier
May 17, 2023
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

'''
PCA Steps:
    1. Get original data
    2. Calculate Covariance Matrix
    3. Calculate EigenVectors
    4. Sort EigenVectors by EigenValues
    5. Choose N largest EigenValues
    6. Project original data onto EigenVectors
'''

# Read in data
df = pd.read_csv('cancer_tumor_data_features.csv')
# print(df.info())

# Too many features. Shrink dimensions w/ principal component analysis...

# Scale Data
scaler = StandardScaler()
scaled_X = scaler.fit_transform(df)

# Make covariance matrix
covariance_mat = np.cov(scaled_X, rowvar=False)

# Calculate Eigen Vectors & Values
eigen_values, eigen_vectors = np.linalg.eig(covariance_mat)

# Determine number of principal components to reduce to (here we choose 2. began with 30 features)
num_components = 2

# Sort by eigen values
sorted_key = np.argsort(eigen_values)[::-1][:num_components] # Sort by eigen values. reverse to have largest to smallest. grab only the first two largest eigen values

# Select N largest EigenValues
eigen_values, eigen_vectors = eigen_values[sorted_key], eigen_vectors[:, sorted_key]

# Project original data onto EigenVectors
principal_compenents = np.dot(scaled_X, eigen_vectors)
pca_df = pd.DataFrame(data=principal_compenents, columns=['x1', 'x2'])

# Visualize the two principal components
sns.scatterplot(data=pca_df, x='x1', y='x2')
plt.show()

# Now, let's see if the 2 components can seperate breast cancer from non, using only 2 features. Load the sklearn breast cancer data set
cancer_dict = load_breast_cancer()
sns.scatterplot(data=pca_df, x='x1', y='x2', hue=cancer_dict['target'])
plt.show()

# NOTE: WOW! We have reduced data from 30 features to 2 components AND the data is still highly seperable