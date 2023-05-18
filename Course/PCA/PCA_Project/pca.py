'''
Adam Forestier
May 16, 2023
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.offline as pyo
import seaborn as sns

from sklearn.decomposition import PCA
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
df = pd.read_csv('digits.csv')
# print(df.info())

# TASK: Create a new DataFrame called pixels that consists only of the pixel feature values by dropping the number_label column.
pixels = df.drop('number_label', axis=1)

# TASK: Grab a single image row representation by getting the first row of the pixels DataFrame.
# TASK: Convert this single row Series into a numpy array.
# TASK: Reshape this numpy array into an (8,8) array.
image_row1 = pixels.iloc[0].to_numpy().reshape(8, 8)
print(image_row1)

# TASK: Use Matplotlib or Seaborn to display the array as an image representation of the number drawn. Remember your palette or cmap choice would change the colors, but not the actual pixel values.
sns.heatmap(image_row1, annot=True, cmap='gray')
plt.show()

# TASK: Use Scikit-Learn to scale the pixel feature dataframe.
scaler = StandardScaler()
scaled_X = scaler.fit_transform(pixels)

# TASK: Perform PCA on the scaled pixel data set with 2 components.
pca_mod = PCA(n_components=2)
pca_results = pca_mod.fit_transform(scaled_X)

# TASK: How much variance is explained by 2 principal components.
explained_variance_ratio = np.sum(pca_mod.explained_variance_ratio_)

# TASK: Create a scatterplot of the digits in the 2 dimensional PCA space, color/label based on the original number_label column in the original dataset.
pca_df = pd.DataFrame(data=pca_results, columns=['pc1', 'pc2'])
sns.scatterplot(data=pca_df, x='pc1', y='pc2', hue=df['number_label'], palette='Dark2')
plt.show()

# TASK: Which numbers are the most "distinct"?
# NOTE: Answer -> 4. It is the most "distinct number label"

# TASK: Create an "interactive" 3D plot of the result of PCA with 3 principal components.
pca_mod = PCA(n_components=3)
pca_results = pca_mod.fit_transform(scaled_X)

# TASK: Create a scatterplot of the digits in the 2 dimensional PCA space, color/label based on the original number_label column in the original dataset.
pca_df = pd.DataFrame(data=pca_results, columns=['pc1', 'pc2', 'pc3'])
fig = px.scatter_3d(pca_df, x='pc1', y='pc2', z='pc3', color=df['number_label'])
filename =  '3d_scatter.html'
pyo.plot(fig, filename=filename)