'''
Adam Forestier
April 23, 2023
Notes:
    Data
        Label
            * target - refers to the prescence of heart disease in the patient. 0 for no prescence, 1 for prescence
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.svm import (
    SVR,
    LinearSVR # NOTE: Faster implementation of SVR, BUT, it can only use a linear kernel
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

# Data investigation
df = pd.read_csv('cement_slump.csv')
df.columns = ['28_Day_CompressiveStrength' if x == 'Compressive Strength (28-day)(Mpa)' else x for x in df.columns]
X = df.drop('28_Day_CompressiveStrength', axis=1)
y = df['28_Day_CompressiveStrength']
plt.figure(dpi=200)
sns.heatmap(df.corr(), annot=True)
plt.show()

# Train, test, split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101, test_size=.3)

# Scale! Important for support vector machines
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
'''
Same parameters as SVC, EXCEPT also epsilon
NOTE: epsilon - How much error you are willing to allow for training data instance
              - epsilon = 0, no error. You want some error or else too high variance/overfitting.
              - .1 by default
'''
base_model = SVR()
c = [.001, .01, .1, .5, 1]
epsilon = [0, 0.01, 0.1, .5, 1, 2]
degree = [1, 2, 3, 4]
param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': c,
    'epsilon': epsilon,
    'gamma': ['scale', 'auto'],
    'degree': degree    
}
grid_model = GridSearchCV(estimator=base_model, param_grid=param_grid)
grid_model.fit(X_train, y_train)
best_params = grid_model.best_params_
print(best_params)

# Create model with best parameters
final_model = SVR(C=1, kernel='linear', gamma='scale', degree=1, epsilon=2)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

# Report metrics
avg = df['28_Day_CompressiveStrength'].mean()
mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred))
print(f'Average 28 Day Compressive Strength: {avg}')
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')