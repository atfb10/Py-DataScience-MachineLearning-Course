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
blobs = pd.read_csv('cluster_blobs.csv')
moons = pd.read_csv('cluster_moons.csv')
circles = pd.read_csv('cluster_circles.csv')

# Visualize
sns.scatterplot(data=blobs, x='X1', y='X2', color='orange')
plt.title('Blobs Scatter')
plt.show()
sns.scatterplot(data=moons, x='X1', y='X2', color='blue')
plt.title('Moon Scatter')
plt.show()
sns.scatterplot(data=circles, x='X1', y='X2', color='green')
plt.title('Circles Scatter')
plt.show()


# Visualize Predictions
def display_categories(model, data, title):
    labels = model.fit_predict(data)
    sns.scatterplot(data=data, x='X1', y='X2', hue=labels)
    plt.title(title)
    plt.show()
    return

# Create models
'''
class DBSCAN(
    eps: Float = 0.5,
    *,
    min_samples: Int = 5,
    metric: str | ((...) -> Any) = "euclidean",
    metric_params: dict | None = None,
    algorithm: Literal['auto', 'ball_tree', 'kd_tree', 'brute'] = "auto",
    leaf_size: Int = 30,
    p: Float | None = None,
    n_jobs: Int | None = None
)
'''
kmeans_blob = KMeans(n_clusters=3)
dbscan_blob = DBSCAN()
kmeans_moon = KMeans(n_clusters=2)
dbscan_moon = DBSCAN(eps=.15)
kmeans_circle = KMeans(n_clusters=2)
dbscan_circle = DBSCAN(eps=.15)

# Display the models
display_categories(kmeans_blob, blobs, 'blobs kmeans clustering')
display_categories(dbscan_blob, blobs, 'blobs dbscan clustering')
display_categories(kmeans_moon, moons, 'moons kmeans clustering')
display_categories(dbscan_moon, moons, 'moons dbscan clustering')
display_categories(kmeans_circle, circles, 'circles kmeans clustering')
display_categories(dbscan_circle, circles, 'circles dbscan clustering')