'''
Adam Forestier
May 17, 2023
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
df = pd.read_csv('cancer_tumor_data_features.csv')
# print(df.info())

# Too many features. Shrink dimensions w/ principal component analysis...

# Scale Data
scaler = StandardScaler()
scaled_X = scaler.fit_transform(df)

'''
NOTE: ALL SHOULD STAY DEFAULT EXCEPT n_components
class PCA(
    n_components: Int | float | str | None = None,
    *,
    copy: bool = True,
    whiten: bool = False,
    svd_solver: Literal['auto', 'full', 'arpack', 'randomized'] = "auto",
    tol: Float = 0,
    iterated_power: Int | Literal['auto'] = "auto",
    n_oversamples: Int = 10,
    power_iteration_normalizer: Literal['auto', 'QR', 'LU', 'none'] = "auto",
    random_state: Int | RandomState | None = None
)
'''
pca_mod = PCA(n_components=2)
pca_mod.fit(scaled_X) # Calculates Eigen Vectors & Values
pc_results = pca_mod.transform(scaled_X) # Projects data onto EigenVectos # NOTE: Can do fit_transform(scaled_X) as 1 line, just seperated here for note purposes
print(pc_results)
# Transform results into 2 feature df
pca_df = pd.DataFrame(data=pc_results, columns=['pc1', 'pc2'])

# Visualize the two principal components
sns.scatterplot(data=pca_df, x='pc1', y='pc2')
plt.show()

# Now, let's see if the 2 components can seperate breast cancer from non, using only 2 features. Load the sklearn breast cancer data set
cancer_dict = load_breast_cancer()
sns.scatterplot(data=pca_df, x='pc1', y='pc2', hue=cancer_dict['target'])
plt.show()

# NOTE: WOW! We have reduced data from 30 features to 2 components AND the data is still highly seperable

'''
NOTE: scikit learn has 2 helpful methods
1. The components themselves
2. The explained variance ratio
'''

# Components
components = pca_mod.components_ # Maximum variance data. sorted by explained variance
df_comp = pd.DataFrame(pca_mod.components_,index=['PC1','PC2'],columns=df.columns)
sns.heatmap(components, annot=True)
plt.show()

# Explained variance ratio
# NOTE: THIS IS HIGHLY IMPORTANT! It shows HOW MUCH OF THE VARIANCE IN DATA IS EXPLAINED BY "N" COMPENENTS
explained_variance_ratio = pca_mod.explained_variance_ratio_
print(explained_variance_ratio)

# Total explained variance ratio
total_explained_variance_ratio = 0
for ratio in explained_variance_ratio:
    total_explained_variance_ratio += ratio

print(total_explained_variance_ratio)


# Show explained variance increase as principal componenets increase
# NOTE: Increase componenets until variance starts to increase slowly!
explained_variance = []
for n in range(1,30):
    pca = PCA(n_components=n)
    pca.fit(scaled_X)
    
    explained_variance.append(np.sum(pca.explained_variance_ratio_))

plt.plot(range(1,30),explained_variance)
plt.xlabel("Number of Components")
plt.ylabel("Variance Explained");
plt.show()