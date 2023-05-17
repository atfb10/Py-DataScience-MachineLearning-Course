'''
author: Adam Forestier
date: April 12, 2023
notes:
    - steps: 
        1. pull out final test set
        2. Choose k folds
        3. train and evaluate for each fold
        4. Obtain the mean error
    - cross_val_score function does this all automatically
    _ cross_validate() - shows multiple performance metrics from cross validation on a model and explore how much time fitting and testing took
'''

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns 
from sklearn.model_selection import (train_test_split, cross_val_score, cross_validate)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (Ridge, RidgeCV)
from sklearn.metrics import mean_squared_error

df = pd.read_csv('d://coding//udemy_resources//python for ml//DATA//Advertising.csv')
X = df.drop('sales', axis=1)
y = df['sales']

# Pull out final test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.10, random_state=101)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create model
model = Ridge(alpha=.1)

# Cross validation
# Takes in estimator aka model, X, y, cv aka folds, scoring - what algorithm is used to score (list here: https://scikit-learn.org/stable/modules/model_evaluation.html)
# Get the average from list of scores
scores = cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_squared_error') 
avg_err_score = abs(scores.mean())
# print(avg_err_score)

# Fit model after cross validation and adjustment of alpha
model.fit(X_train, y_train)
final_y_predictions = model.predict(X_test)

# Show root mean squared error on test data
rmse = np.sqrt(mean_squared_error(y_test, final_y_predictions))
# print(rmse)

# --------------------------------------------------------------------------------------------
# Cross validation with cross_validate
X = df.drop('sales', axis=1)
y = df['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=101)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Ridge(alpha=.01)

# Cross validate
# Pass in model, X_train, y_train, list of scoring metrics, folds
# NOTE: cross_validate returns a dictionary with fit_time, score_time and scoring metrics as keys, and their subsequent values as values. Easily pass dicitonary into Pandas DataFram
scores = cross_validate(model, X_train, y_train, scoring=['neg_mean_absolute_error', 'neg_mean_squared_error'], cv=10)
scores_df = pd.DataFrame(scores)
# print(scores_df)

# Just an example of doing it in one line
scores = pd.DataFrame(cross_validate(model, X_train, y_train, scoring=['neg_mean_absolute_error', 'neg_mean_squared_error'], cv=10))
print(scores.mean()) # Show averages for each
# print(scores)


model.fit(X_train, y_train)
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(rmse)
# [print(prediction) for prediction in predictions]