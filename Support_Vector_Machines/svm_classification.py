'''
Adam Forestier
April 23, 2023
Notes:
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.svm import SVC 
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, RocCurveDisplay, auc, roc_curve
)

# margin plot function import 
from svm_margin_plot import plot_svm_boundary

# Investigate data
df = pd.read_csv('mouse_viral_study.csv')
virus_corr = df.corr()['Virus Present']
sns.set_style(style='darkgrid')
sns.pairplot(df, hue='Virus Present')
plt.show()
sns.heatmap(df.corr(), annot=True)
plt.show()
sns.countplot(x='Virus Present', data=df)
plt.show()  

# Investigate data, create a hyperplane
# Create hyperplane (2d, so just a line)
x = np.linspace(0, 10, 100)
m = -1
b = 11
y = m*x + b
sns.scatterplot(x='Med_1_mL', y='Med_2_mL', hue='Virus Present', data=df, palette='dark')
plt.plot(x, y, color='black')
plt.show()

# Seperate labels and feature
X = df.drop('Virus Present', axis=1)
y = df['Virus Present']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=101)

# Scale
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)
X_train = scaler.transform(X_train)

# Support Vector Classifier
'''
C - In SciKit-Learn, the strength of the regularization is INVERSELY PROPORTIONAL TO C in the mathmatical algorithm. By default it is 1.0. The penalty is a squared l2 penalty 
  - NOTE: What this is all means... If "C" is high, you allow few to zero points within the margin. If C is low, you allow many points within the margin
kernel - kernel (linear, poly, rbf, sigmoid, precomputed) default = rbf. rbf = radial basis function. To use pre-computed, must be a square matrix 
degree - degree of polynomial kernal function
gamma - 'scale', 'auto' OR manually provide floating point number.  scale = 1 / (n_features * x variants()). auto = 1 / n_features
      - gamma defines how much influence a single training example has. as gamma gets larger the closer the other examples have to be to be effected
      - NOTE: gamma getting larger, indicates each point having bigger influence. too high of gamma too high variance (overfitting). too low, too much bias and won't fit enough
'''

# Linear example
svc_model = SVC(kernel='linear', C=1.0)
svc_model.fit(X, y)
plot_svm_boundary(model=svc_model, X=X, y=y) # NOTE: Amazing@! This will show features you can effect. It shows the hyperplane, the margins and the support vectors. NOTE: Use whole dataset for the visualization

# Radial basis function - NOTE: Most often the best!
'''
Super rad! 
In different n-dimension
Graph projected back down to 2d even though it is n-d
Should see curve instead of line.
Margin should surroud cluster as a circle
'''
svc_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svc_model.fit(X,y)
plot_svm_boundary(svc_model, X, y)

# Sigmoid example - here it can be seen as a terrible choice for kernal. drastically overfitting
svc_model = SVC(kernel='sigmoid', C=1.0, gamma='scale')
svc_model.fit(X, y)
plot_svm_boundary(svc_model, X, y)

# Polynomial example
svc_model = SVC(kernel='poly', C=1.0, gamma='scale', degree=3)
svc_model.fit(X, y)
plot_svm_boundary(svc_model, X, y)

# Now, set up the model for real! ALWAYS USE GRID SEARCH w/ SVM. Hard to have good intuition about c value, gamma, kernal, etc
base_model = SVC()
c = np.round(a=np.linspace(0.01, 1.1, 10), decimals=2)
param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': c,
    'gamma': ['scale', 'auto']
}
grid_model = GridSearchCV(estimator=base_model, param_grid=param_grid)
grid_model.fit(X_train, y_train)
best_params = grid_model.best_params_
print(best_params)