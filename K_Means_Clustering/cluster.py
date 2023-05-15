'''
Adam Forestier
May 15, 2023
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Read in data
df = pd.read_csv('bank-full.csv')
# print('-------------------------------------------------')
# print(df.info())
# print('-------------------------------------------------')
# print(df.describe())

'''
EDA - Exploratory Data Analysis
NOTE: For unsupervised learning, always have domain knowledge of features or ask someone that does
NOTE: For unsupervised learning, always do lots of visualization to explore the data fully
'''
sns.histplot(data=df, x='age', bins=40, kde=True)
plt.show()
sns.histplot(data=df[df['pdays']!=999], x='pdays', bins=10, kde=True)
plt.show()
sns.histplot(data=df[df['pdays']!=999], x='duration', hue='contact', kde=True)
plt.show()
sns.countplot(data=df, x='contact')
plt.show()
sns.countplot(data=df, x='job')
plt.xticks(rotation=90)
plt.show()
sns.countplot(data=df[df['loan'] == 'yes'], x='education', hue='default', order=df['education'].value_counts().index)
plt.xticks(rotation=90)
plt.show()

# NOTE: Do not need to drop first for clustering!
X = pd.get_dummies(df)

# NOTE: IMPORTANT!!!! MUST SCALE for unsupervised learning. Distance metric is what determines
s = StandardScaler()
scaled_X = s.fit_transform(X=X)

'''
Make model
n_clusters
max_iter NOTE: Good to set! If point falls in between two spots, it will go back and forth and be stuck in infinite loop
use 'auto' for precompute_distance
random_state - default is None, usually not very important
'''
model = KMeans(n_clusters=2, max_iter=250)
cluster_labels = model.fit_predict(scaled_X) # Fit data points to cluster (fit). Assign cluster labels to each point (predict). Returns a numpy ndarray
X['cluster'] = cluster_labels # Create new column that has predicted cluster label

# Now let's see correlation to cluster label
cluster_corr = X.corr()['cluster'].iloc[:-1].sort_values().plot(kind='bar')
plt.show()

# NOTE: Let's find the best K that we can
ssd = {}
for i in range(2, 9):
    model = KMeans(n_clusters=i, max_iter=250)
    model.fit(scaled_X)
    ssd[i] = model.inertia_ # SSD point --> Cluster Center

plt.plot(list(ssd.keys()), list(ssd.values()), 'o--') # NOTE: k=6 seems like a good idea! The drop of sum of squared distances is signficantly less at this value K
plt.show()

# NOTE: How to show difference between each value K
k_diff = pd.Series(list(ssd.values()))
print(k_diff)