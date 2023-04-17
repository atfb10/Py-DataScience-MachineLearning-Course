'''
Adam Forestier
April 15, 2023
Notes:
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 
from sklearn.preprocessing import (
    StandardScaler
    )
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import (
    ElasticNet
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

df = pd.read_csv('AMES_Final_DF.csv')

# TASK: The label we are trying to predict is the SalePrice column. Separate out the data into X features and y labels
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# TASK: Use scikit-learn to split up X and y into a training set and test set. Since we will later be using a Grid Search strategy, set your test proportion to 10%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=101)

# TASK: The dataset features has a variety of scales and units. For optimal regression performance, scale the X features. 
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# TASK: We will use an Elastic Net model. Create an instance of default ElasticNet model with scikit-learn
# TASK: The Elastic Net model has two main parameters, alpha and the L1 ratio. Create a dictionary parameter grid of values for the ElasticNet. Feel free to play around with these values, keep in mind, you may not match up exactly with the solution choices
base_model = ElasticNet()
grid_parameters  = {
    'alpha': [.001, .01, .1, .25, .5, 1],
    'l1_ratio': [.01, .5, .75, .9, .95, .99, .999, 1]
} 

# TASK: Using scikit-learn create a GridSearchCV object and run a grid search for the best parameters for your model based on your scaled training data.
grid_model = GridSearchCV(base_model, grid_parameters, scoring='neg_mean_squared_error', cv=5, verbose=0)
grid_model.fit(X_train, y_train)

# TASK: Display the best combination of parameters for your model
best_hyperparameters = grid_model.best_params_
print(best_hyperparameters)

# TASK: Evaluate your model's performance on the unseen 10% scaled test set
predictions = grid_model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
avg_saleprice = df['SalePrice'].mean()
accuracy = rmse / avg_saleprice
print(mae)
print(rmse)
print(f'Accuracy Percentage = {accuracy}')