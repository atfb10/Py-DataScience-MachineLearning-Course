'''
Adam Forestier
May 16, 2023
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# Read in data
df = pd.read_csv('cluster_two_blobs.csv')
df_outliers = pd.read_csv('cluster_two_blobs_outliers.csv')

# Visualize Results of models w/ different hyperparameters
def assign_visualize_clusters(model, data: pd.DataFrame) -> None:
    cluster_labels = model.fit_predict(data)
    sns.scatterplot(data=data, x='X1', y='X2', hue=cluster_labels, palette='Set1')
    plt.show()
    return

# Count number of outliers
def count_outliers(model) -> int:
    return np.sum(model.labels_ == -1)

# Percent of points classified as outliers
def outlier_percentage(model) -> float:
    return 100 * np.sum(model.labels_ == -1) / len(model.labels_)

# Create models
base = DBSCAN()
small_eps = DBSCAN(eps=.01)
one_eps = DBSCAN(eps=1)


# Test models
# assign_visualize_clusters(base, df)
# assign_visualize_clusters(base, df_outliers)
# print(f'outlier count {count_outliers(base)}')
# assign_visualize_clusters(small_eps, df_outliers)
# print(f'outlier count {count_outliers(small_eps)}')
# assign_visualize_clusters(one_eps, df_outliers)
# print(f'outlier count {count_outliers(one_eps)}')

# Elbow method. NOTE: Do exact same thing for minimum number of points
num_outliers = {}
outlier_percent = {}

for i in np.linspace(.001, 2, 100):
    model = DBSCAN(eps=i)
    model.fit(df_outliers)
    num_outliers[i] = count_outliers(model=model)
    outlier_percent[i] = outlier_percentage(model=model)

sns.lineplot(x=list(num_outliers.keys()), y=list(num_outliers.values()))
plt.show()
sns.lineplot(x=list(outlier_percent.keys()), y=list(outlier_percent.values()), color='orange')
plt.ylabel('Percentage of Outliers')
plt.xlabel('Epsilon')
plt.show()

# Zoom in to find epsilon to have 3 outliers!
sns.lineplot(x=list(outlier_percent.keys()), y=list(outlier_percent.values()), color='orange')
plt.ylabel('Percentage of Outliers')
plt.xlabel('Epsilon')
plt.ylim(0, 10)
plt.hlines(y=3, xmin=0, xmax=2, colors='black')
plt.show()