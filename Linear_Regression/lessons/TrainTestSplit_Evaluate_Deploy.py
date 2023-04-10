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

# Show visually
fig, axes = plt.subplots(nrows=1, ncols=3)
axes[0].plot(df['TV'], df['sales'], 'o')
axes[0].set_ylabel('Sales')
axes[0].set_title('TV Spend')
axes[1].plot(df['radio'], df['sales'], 'o')
axes[1].set_ylabel('Sales')
axes[1].set_title('Radio Spend')
axes[2].plot(df['newspaper'], df['sales'], 'o')
axes[2].set_ylabel('Sales')
axes[2].set_title('Newspaper Spend')
plt.show()

# Features
X = df.drop('sales', axis=1)

# y label
y = df['sales']

'''
Seperate train and test
test size is percentage that should go to test. ensure to set random_state (similar to random seed) to all algorithms tested
random_state shuffles the data randomly so that is not ordered in any way 
y_train index matches up with index of X_train
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
print(X_train.head())
print(y_train.head())

# Create model (Called estimator within Sci-Kit Learn)
model = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=None)
model.fit(X_train, y_train) # .fit() updates model object in place
test_predictions = model.predict(X_test) # model predicts sales based on what it has been fitted on! (array of predicted values)

# Error metrics
avg_sales = df['sales'].mean() # 14

# NOTE: Notes on these error metric types are in "notes.txt"
# NOTE: use mean_absolute_error and root mean square error -> If they are close, great! If not & root mean squared error is much higher, it means the data has outliers
mean_abs_error = mean_absolute_error(y_test, test_predictions) # 1.2 - error percentage of approx 10% 
mean_sq_err = mean_squared_error(y_test, test_predictions) # 2.3
root_mean_sq_err = np.sqrt(mean_sq_err) # 1.5

# Residual  - see patterns and see if data is a good selection for linear regression. NOTE: Most important plot to look at to determine if linear regression is a good choice for data
test_residuals = y_test - test_predictions
sns.scatterplot(x=y_test, y=test_residuals) # Plot actual data vs residuals
plt.axhline(y=0, color='r', ls='--')
plt.show() # Make sure there is no clear line or curve! It should look random!

sns.displot(test_residuals, bins=20, kde=True) # Distrubution plot
plt.show()

# Probability plot
fig, ax = plt.subplots(figsize=(6,8), dpi=100)
sp.stats.probplot(test_residuals, plot=ax)
plt.show() # Red line = normal distribution. If blue dots are close, likely good for linear regression. If far off, 

# final model
final_model = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=None)
final_model.fit(X,y) # If the model is good enough, fit it on the WHOLE dataset

'''
# beta_coefficients = .046 for tv, .18 for radio, -.001 for newspaper. This means that newspaper has no bearing on sales! Radio has the greatest effect and tv has some effect
This means - for every increase of 1 for tv, we would expect a .046 increase in sales. The same applies for radio. For every increase in 1 for newspaper, expect a drop of .001! (close to 0, but negative means a negative relationship)
# IMPORTANT NOTE: When X labels are in different units, you must normalize the data!
'''
beta_coefficients = final_model.coef_ 

# graph predicted points vs actual points
y_hat = final_model.predict(X)
fig, axes = plt.subplots(nrows=1, ncols=3)
axes[0].plot(df['TV'], df['sales'], 'o') # actual
axes[0].plot(df['TV'], y_hat, 'o', color='red') # predicted
axes[0].set_ylabel('Sales')
axes[0].set_title('TV Spend')
axes[1].plot(df['radio'], df['sales'], 'o') # actual
axes[1].plot(df['radio'], y_hat, 'o') # predicted
axes[1].set_ylabel('Sales')
axes[1].set_title('Radio Spend')
axes[2].plot(df['newspaper'], df['sales'], 'o') # actual
axes[2].plot(df['newspaper'], y_hat, 'o') # predicted
axes[2].set_ylabel('Sales')
axes[2].set_title('Newspaper Spend')
plt.show()

# Model deployment
dump(final_model, 'final_sales_model.joblib') # Trained model file saved on computer that you can send to someoene

# Load a saved model
loaded_model = load('linear_final_sales_model.joblib')
print(loaded_model.coef_) # See that is the same model as the one deployed

'''
# Predict sales on advertising campaign using final model that has been loaded
# 149 for tv, 22 on radio, 12 newspaper
'''
campaign = [[149, 22, 12]]
predicted_sales = loaded_model.predict(campaign)
print(f'Predicted Sales: ${predicted_sales}')