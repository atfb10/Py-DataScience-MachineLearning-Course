'''
Adam Forestier
May 17, 2023
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)
from joblib import dump, load

# Read in Data & Split into features & labels
df = pd.read_csv('Advertising.csv')
X = df.drop('sales', axis=1)
y = df['sales']

# 70% train | 15% test | 15% holdout set
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=.3, random_state=101)
X_test, X_holdout, y_test, y_holdout = train_test_split(X_validation, y_validation, test_size=.5, random_state=101)

# Model Creation
model = RandomForestRegressor(n_estimators=102, random_state=101) # Always set random_state
model.fit(X=X_train, y=y_train)
y_pred = model.predict(X=X_test)

# Metrics
avg_sales = df['sales'].mean()
std_sales = df['sales'].std()
rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_test))
mae = mean_absolute_error(y_pred=y_pred, y_true=y_test)
print(f'Average Sales = {avg_sales}')
print(f'Standard Deviation of Sales = {std_sales}')
print(f'Mean Absolute Error = {mae}')
print(f'Root Mean Squared Error = {rmse}')

# FINAL PERFORMANCE METRICS. NOTE - DO NOT TUNE MODEL AFTER THIS. THIS IS HOW THE MODEL SHOULD EXPECT TO PERFORM ON NEW, UNSEEN DATA!
print('-----------------------------\n Final Holdout Performance')
final_pred = model.predict(X=X_holdout)
avg_sales = df['sales'].mean()
std_sales = df['sales'].std()
rmse = np.sqrt(mean_squared_error(y_pred=final_pred, y_true=y_holdout))
mae = mean_absolute_error(y_pred=final_pred, y_true=y_holdout)
print(f'Average Sales = {avg_sales}')
print(f'Standard Deviation of Sales = {std_sales}')
print(f'Mean Absolute Error = {mae}')
print(f'Root Mean Squared Error = {rmse}')

# Make final model. NOTE: DO THIS TO FIT IT ON THE ENTIRE DATASET
final_model = RandomForestRegressor(n_estimators=102, random_state=101)
final_model.fit(X=X, y=y)
dump(final_model, 'final_model.pkl') # Save pickle file
dump(list(X.columns), 'col_names.pkl') # Save column names


# NOTE: Load model and test it on new data!
new_cols =  load('col_names.pkl')
print(new_cols)
loaded_model = load('final_model.pkl')

print('--------------------------------------------')
pred = loaded_model.predict([[230.1, 37.8, 69.2]]) # Should get value around 22.1. This is the tv, radio, and newspaper values for row 1 of the dataframe
print(f'Expected Sales: {pred}') # 21.9 -> Good enough