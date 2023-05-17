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
- create instance of polynomial features
- feature list: TV, radio, newspaper, TV**2, radio**2, newspaper2 TV * radio, TV * newspaper, radio * newspaper
'''
poly_converter = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
poly_converter.fit(X)
poly_features = poly_converter.transform(X)
poly_features = poly_converter.fit_transform(X) # Do in a single line instead of 2!

# Train test split
X_train, y_train, X_test, y_test = train_test_split(poly_features, y, test_size=.3, random_state=101)

# Create model 
model = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=None)
model.fit(X_train, y_train)
test_predictions = model.predict(X_test)

# Error metrics
sales_avg = df['sales'].mean()
mean_abs_err = mean_absolute_error(y_test, test_predictions) #.49 - better than linear! 
root_mean_sq_error = np.sqrt(mean_squared_error(y_test, test_predictions)) #.66 - better than linear!