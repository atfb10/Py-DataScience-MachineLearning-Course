'''
Adam Forestier
May 16, 2023
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Read in data
df = pd.read_csv('wholesome_customers_data.csv')

# TASK: Create a scatterplot showing the relation between MILK and GROCERY spending, colored by Channel column.
sns.scatterplot(data=df, x='Milk', y='Grocery', hue='Channel')
plt.show()

# TASK: Use seaborn to create a histogram of MILK spending, colored by Channel. Can you figure out how to use seaborn to "stack" the channels, instead of have them overlap?
sns.histplot(data=df, x='Milk', hue='Channel', multiple='stack')
plt.show()

# TASK: Create an annotated clustermap of the correlations between spending on different cateogires.
spend_features = ['Milk', 'Grocery', 'Detergents_Paper', 'Delicassen', 'Fresh', 'Frozen']
temp = df[spend_features]
sns.clustermap(temp.corr(), annot=True)
plt.show()

# TASK: Create a PairPlot of the dataframe, colored by Region.
sns.pairplot(df, hue='Region')
plt.show()

# TASK: Since the values of the features are in different orders of magnitude, let's scale the data. Use StandardScaler to scale the data.
scaler = StandardScaler()
X = scaler.fit_transform(df)
X = pd.DataFrame(data=X, columns=df.columns)

# ASK: Use DBSCAN and a for loop to create a variety of models testing different epsilon values. Set min_samples equal to 2 times the number of features.
# During the loop, keep track of and log the percentage of points that are outliers.
def outlier_percentage(model) -> float:
    '''
    calculate outlier percentage
    '''
    return 100 * np.sum(model.labels_==-1) / len(model.labels_)

def dbscan_outlier_by_eps(eps_range: np.ndarray, data: pd.DataFrame) -> dict:
    '''
    find outlier by percentage by eps
    ''' 
    eps_outlier_percent = {}
    min_samples = len(data.columns) * 2
    for eps in eps_range:
        model = DBSCAN(eps=eps, min_samples=min_samples)
        model.fit(data)
        eps_outlier_percent[eps] = outlier_percentage(model=model)
    return eps_outlier_percent

# TASK: Create a line plot of the percentage of outlier points versus the epsilon value choice.
eps = np.linspace(0.001, 3, 50)
eps_outlier_percent = dbscan_outlier_by_eps(eps_range=eps, data=X)
sns.lineplot(x=list(eps_outlier_percent.keys()), y=list(eps_outlier_percent.values()))
plt.xlabel('Epsilon')
plt.ylabel('Percentage of Outliers')
plt.show()

# TASK: Based on the plot created in the previous task, retrain a DBSCAN model with a reasonable epsilon value. Note: For reference, the solutions use eps=2.
model = DBSCAN(eps=2, min_samples=len(X.columns)*2)
cluster_labels = model.fit_predict(X)

# TASK: Create a scatterplot of Milk vs Grocery, colored by the discovered labels of the DBSCAN model.
sns.scatterplot(data=df, x='Grocery', y='Milk', hue=cluster_labels)
plt.show()

# TASK: Create a scatterplot of Milk vs. Detergents Paper colored by the labels.
sns.scatterplot(data=df, x='Detergents_Paper', y='Milk', hue=cluster_labels)
plt.show()

# TASK: Create a new column on the original dataframe called "Labels" consisting of the DBSCAN labels.
df['Labels'] = cluster_labels

# TASK: Compare the statistical mean of the clusters and outliers for the spending amounts on the categories.
cluster_means = df.groupby('Labels').mean()[spend_features]
print(cluster_means)

# TASK: Normalize the dataframe from the previous task using MinMaxScaler so the spending means go from 0-1 and create a heatmap of the values
# spend_features.append('Labels')
# temp = df[spend_features]
# temp = temp.set_index
scaler = MinMaxScaler()
X = scaler.fit_transform(cluster_means)
X = pd.DataFrame(data=X, index=cluster_means.index, columns=cluster_means.columns)
sns.heatmap(X, annot=True)
plt.show()

# TASK: Create another heatmap similar to the one above, but with the outliers removed
sns.heatmap(X.loc[[0,1]], annot=True)
plt.show()