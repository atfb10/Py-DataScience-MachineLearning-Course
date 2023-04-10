'''
Adam Forestier
April 10, 2023
Notes:
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 
from sklearn.preprocessing import (
    PolynomialFeatures,
    StandardScaler
    )
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    LinearRegression,
    Lasso, # LASSO with Cross Validation! It performs LASSO Regression for a variety of alpha values. Should ALWAYS choose this one - cross validation is more accurate than not using it... duh
    LassoCV,
    Ridge,
    RidgeCV, # Ridge with Cross Validation! It performs Ridge Regression for a variety of alpha values. Should ALWAYS choose this one - cross validation is more accurate than not using it... duh
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    SCORERS, # Dictionaries of metrics one can use to determine how well something performs
)
import scipy as sp
from joblib import dump, load

df = pd.read_csv('Advertising.csv') 
X = df.drop('sales', axis=1)
y = df['sales']

poly_converter = PolynomialFeatures(degree=3, include_bias=False)
poly_features = poly_converter.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=.3, random_state=101)

# Feature Scaling!
# NOTE: DO NOT FIT TO WHOLE dataset! Results in data leakage
scaler = StandardScaler()
scaler.fit(X_train)
scaled_X_train = scaler.transform(X_train)
scaled_X_test = scaler.transform(X_test)
print(y_train.shape)

# OG Linear Regression
model = LinearRegression()
model.fit(scaled_X_train, y_train)
test_pred = model.predict(scaled_X_test)
mae = mean_absolute_error(y_test, test_pred)
rmse = np.sqrt(mean_squared_error(y_test, test_pred))
print('Original Linear Regression')
print(f'MAE: {mae}')
print(f'rmse: {rmse}')
print('-------------------------------')

# Ridge Regression
my_alpha = 10
ridge_model = Ridge(alpha=my_alpha)
ridge_model.fit(scaled_X_train, y_train)
test_pred = ridge_model.predict(scaled_X_test)
mae = mean_absolute_error(y_test, test_pred)
rmse = np.sqrt(mean_squared_error(y_test, test_pred))
print('Ridge Regression')
print(f'Assigned Alpha: {my_alpha}') # See which alpha performed the best
print(f'MAE: {mae}')
print(f'rmse: {rmse}')
print('-------------------------------')

# Ridge Cross Validation model
# CV is k-fold cross validation number. if left to none it performs leave one out. huge datasets require selecting a number. scoring is the algorithm used to determine which alpha performed the best!
ridge_cv_model = RidgeCV(alphas=(.001, .025, .05, 0.1, .15, 0.2, 0.25, 1.0, 10.0), cv=None, scoring='neg_mean_absolute_error') 
ridge_cv_model.fit(scaled_X_train, y_train)
test_pred = ridge_cv_model.predict(scaled_X_test)
mae = mean_absolute_error(y_test, test_pred)
rmse = np.sqrt(mean_squared_error(y_test, test_pred))
print('Cross Validated Ridge Regression')
print(f'Best Scoring Alpha: {ridge_cv_model.alpha_}') # See which alpha performed the best
print(f'MAE: {mae}')
print(f'rmse: {rmse}')
print('-------------------------------')

# Lasso Regression
# Lasso Regression
# CV is k-fold cross validation number. if left to none it performs leave one out. huge datasets require selecting a number. 
# eps is ratio of alpha min to alpha max
# n_alphas is the number of alphas checked
# Max iterations is what it says it is... duh, if not high enough I will get a convergence warning as it does not cover all possibilities. This could also be handled by raising eps, aka lowering the search field
lasso_cv_model = LassoCV(eps=0.001, n_alphas=100, cv=None, max_iter=1000000)
lasso_cv_model.fit(scaled_X_train, y_train)
test_pred = lasso_cv_model.predict(scaled_X_test)
mae = mean_absolute_error(y_test, test_pred)
rmse = np.sqrt(mean_squared_error(y_test, test_pred))
print('Cross Validated LASSO Regression')
print(f'Best Scoring Alpha: {lasso_cv_model.alpha_}') # See which alpha performed the best
print(f'MAE: {mae}')
print(f'rmse: {rmse}')
print('-------------------------------')

# Elastic Regression



# See the coefficients
print(f'LASSO Beta Coefficients {lasso_cv_model.coef_}') # Only considering a few coefficients (ones not equal to 0)! A very simple and easy to understand model
print(f'Ridge Beta Coefficients {ridge_cv_model.coef_}') # considers all coefficients, more complex and harder to understand. 

# lasso_cv_model and ridge_cv_model in this instance are very close in accuracy, but lasso is a simpler model. 
# NOTE: IMPORTANT: Context considered - when accuracy is close, always choose the simpler model