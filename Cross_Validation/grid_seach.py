'''
author: Adam Forestier
date: April 12, 2023
notes:
    - Complex models often have multiple adjustable hyperparameters
    - Grid Search - Way of training and validating a model on every possible combination of hyperparameter options
    - SciKit-Learn has GridSearchCV - class capable of testing a dictionary of multiple hyperparameter options through cross-validation
        * This is absolutely epic! This allows for BOTH cross-validation and grid search to be performed in a generalized way for any model
'''

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns 
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import (mean_squared_error, mean_absolute_error)

# Data gathering
df = pd.read_csv('d://coding//udemy_resources//python for ml//DATA/Advertising.csv')
X = df.drop('sales', axis=1)
y = df['sales']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=101)

# Scale
sclr = StandardScaler()
X_train = sclr.fit_transform(X_train)
X_test = sclr.transform(X_test)

# Create model off grid
base_model = ElasticNet()
grid_parameters  = {
    'alpha': [.001, .01, .1, .25, .5, 1],
    'l1_ratio': [.01, .1, .25, .33, .5, .75, .85, .9, .95, .99, .999, 1]
} 

# Takes in estimator aka model, grid parameters, scoring aka metric, cv aka folds, verbose an integer that determines how much information I want printed to the terminal 
# We then use the object of the GridSearchCV 
# NOTE: THIS IS THE SAUCE! Only problem is if the model is huge and or the computer's hardware is not powerful
grid_model = GridSearchCV(base_model, grid_parameters, scoring='neg_mean_squared_error', cv=10, verbose=1)
grid_model.fit(X_train, y_train)
predictions = grid_model.predict(X_test) # This automatically uses the best estimator automatically!

# Show best alpha and l1_ratio
best_alpha_li = grid_model.best_estimator_
print(best_alpha_li)

# see all results for each parameter
grid_model_results_df = pd.DataFrame(grid_model.cv_results_)
print(grid_model_results_df)

# print(f'MAE: {mae}')
# print(f'RMSE: {rmse}')