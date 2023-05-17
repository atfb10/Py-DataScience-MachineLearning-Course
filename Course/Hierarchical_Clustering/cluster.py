'''
Adam Forestier
May 16, 2023
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler

# Data investigation
df = pd.read_csv('cluster_mpg.csv')
df = pd.get_dummies(df.drop('name', axis=1))

# Scale 
scaler = MinMaxScaler()
scaled_array = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_array, columns=df.columns)

# Visualize

# Correlations of columns
sns.heatmap(scaled_df.corr(), cmap='viridis')
plt.show()

# Clustering rows together (NOTE: this is a dendogram)
sns.clustermap(scaled_df, cmap='viridis', col_cluster=False)
plt.show()

'''
Model
    n_clusters: int | None = 2, NOTE: Must be non if distance threshold is not None!
    *,
    affinity: str | ((...) -> Any) = "deprecated", NOTE: distance metric - just use default of Euclidean
    metric: str | ((...) -> Any) | None = None,
    memory: Any | str | None = None,
    connectivity: ArrayLike | ((...) -> Any) | None = None,
    compute_full_tree: bool | Literal['auto'] = "auto",
    linkage: Literal['ward', 'complete', 'average', 'single'] = "ward",
    distance_threshold: Float | None = None, NOTE: Linkage distance above which clusters will not be merged. compute_full_tree must be True & n_clusters must be false
    compute_distances: bool = False

'''
model = AgglomerativeClustering(n_clusters=4)
cluster_labels = model.fit_predict(scaled_df)

# Visualize the model
sns.scatterplot(data=df, x='mpg', y='weight', hue=cluster_labels, palette='viridis')
plt.show()

# What is the max theoretical distance between 2 points in the dataset? Using Euclidean
n_features = len(scaled_df.columns)
max_distance = np.sqrt(n_features)

'''
model
NOTE: using distance_threshold!
'''
model = AgglomerativeClustering(n_clusters=None, distance_threshold=0, compute_full_tree=True)
cluster_labels  = model.fit_predict(scaled_df)

# Create dendogram using scipy! Can help determine n_clusters
linkage_matrix = hierarchy.linkage(model.children_) # matrix. col1 = cluster #, col2 = cluster #, col3 = distance between the two, col4 = total amount of points it is connecting

'''
ndarray
truncate_mode - condenses dendrogram. 'None' -> No trunctation. 'lastp', 'level'
p - number of linkages allowed if truncate_mode is not none
'''
dendro = hierarchy.dendrogram(linkage_matrix, truncate_mode='lastp', p=10) # create a dendogram of all points
plt.show()

# MPG is a large seperator of clusters. what is the max distance between mpg?
max_mpg_id = scaled_df['mpg'].idxmax()
min_mpg_id = scaled_df['mpg'].idxmin()
max_mpg_car = scaled_df.iloc[max_mpg_id]
min_mpg_car = scaled_df.iloc[min_mpg_id]

distance = np.linalg.norm(max_mpg_car - min_mpg_car)

# Model using calculated distance
model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance, compute_full_tree=True)
cluster_labels = model.fit_predict(scaled_df)

# Visualize the model
sns.scatterplot(data=df, x='mpg', y='weight', hue=cluster_labels, palette='viridis')
plt.show()