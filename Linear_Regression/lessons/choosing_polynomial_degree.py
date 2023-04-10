'''
Adam Forestier
April 9, 2023
Notes:

'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)
import scipy as sp
from joblib import dump, load

df = pd.read_csv('Advertising.csv')
X = df.drop('sales', axis=1)
y = df['sales']

'''
Steps:
1. Create the different order polynomial
2. split poly feawture train/test
3. fit on train
4. store/save the rmse for BOTH train and test
5. plot the results (error vs polynomial order)
'''
train_rmse_errors = {}
test_rmse_errors = {}

for degree in range(1, 6):
    poly_converter = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly_converter.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=.3, random_state=101)
    model = LinearRegression()
    model.fit(X_train, y_train)
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    train_rmse_errors[degree] = np.sqrt(mean_squared_error(y_train, train_predictions))
    test_rmse_errors[degree] = np.sqrt(mean_squared_error(y_test, test_predictions))

train_rmse_errors_values = list(train_rmse_errors.values())
test_rmse_errors_values = list(test_rmse_errors.values())

# Show on console error metrics
# It appears a degree of 4 is the best! Lowest error metric on both train degree and test degree. AWESOME! Overfit above 4 and underfit under 4
# NOTE: SUPER IMPORTANT!!!! However.... Context is key. Is the 4th power or even the 3rd degree complexity worth the risk, since it appears to be performing only marginally better? 
# Definitely not 4th, maybe third. Certainly select 2nd degree or 3rd degree
for i in range(len(test_rmse_errors_values)):
    print(f'Train degree {i + 1}: {train_rmse_errors_values[i]} | Test degree {i + 1}: {test_rmse_errors_values[i]}')

# Show what is printing as a graph. AWESOME!
plt.plot(range(1,6), train_rmse_errors_values[:5], label='Train RMSE')
plt.plot(range(1,6), test_rmse_errors_values[:5], label='Test RMSE')
plt.ylabel('RMSE')
plt.xlabel('Degree of Polynomial')
plt.legend()
plt.show()

'''
- Save model
- NOTE: for polynomial models, you must save the polynomial converter! Not just the model, becaue there are more features than in the original dataset
'''
final_poly_converter = PolynomialFeatures(degree=3, include_bias=False)
fully_converted_x = final_poly_converter.fit_transform(X)
final_model = LinearRegression()
final_model.fit(fully_converted_x, y)

# Save converter & model
dump(final_model, 'final_poly_model.joblib')
dump(final_poly_converter, 'final_converter.joblib')

# Load and use
loaded_converter = load('final_converter.joblib')
loaded_model = load('final_poly_model.joblib')

campaign = [[149, 22, 12]]
converted_campaign = loaded_converter.fit_transform(campaign)
predicted_sales = loaded_model.predict(converted_campaign)
predicted_sales = round(predicted_sales[0], 2)
print(f'Predicted sales: ${predicted_sales}')