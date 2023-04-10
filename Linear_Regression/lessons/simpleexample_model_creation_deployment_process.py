'''
Adam Forestier
April 9, 2023
Notes:

'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)
import scipy as sp
from joblib import dump, load

# Question to answer: what is the relationship between each advertising channel (tv, radio, newspaper) and sales?
df = pd.read_csv('Advertising.csv')

# Features
X = df.drop('sales', axis=1)

# y label
y = df['sales']

'''
Seperate train and test
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
print(X_train.head())
print(y_train.head())

# Create model (Called estimator within Sci-Kit Learn)
model = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=None)
model.fit(X_train, y_train)
test_predictions = model.predict(X_test)

# Error metrics
avg_sales = df['sales'].mean() # 14
mean_abs_error = mean_absolute_error(y_test, test_predictions) # 1.2 - error percentage of approx 10% 
root_mean_sq_err = np.sqrt(mean_squared_error(y_test, test_predictions)) # 1.5

# Ensure data is good selection for linear regression model using residual plot. Make sure there is no clear line or curve! It should look random!
test_residuals = y_test - test_predictions
sns.scatterplot(x=y_test, y=test_residuals) # Plot actual data vs residuals
plt.axhline(y=0, color='r', ls='--')
plt.show() 

# final model
final_model = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=None)
final_model.fit(X,y) # If the model is good enough, fit it on the WHOLE dataset

'''
# beta_coefficients
'''
beta_coefficients = final_model.coef_ 

# Model deployment
dump(final_model, 'final_sales_model.joblib')

# Load a saved model
loaded_model = load('final_sales_model.joblib')

'''
# Predict sales on advertising campaign using final model that has been loaded
'''
campaign = [[149, 22, 12]]
predicted_sales = loaded_model.predict(campaign)
print(f'Predicted Sales: ${predicted_sales}')