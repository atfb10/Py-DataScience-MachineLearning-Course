'''
author: Adam Forestier
date: April 11, 2023
notes:
    - If we want a truly fair and final set of performance metrics, we should get these metrics from a FINAL test set that we do not allow ourselves to adjust 
    - The model is not fit to the final test data and the model hyperparameters are not adjusted based off of the final test data
    - This data is used to truly evaluate the model
'''

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


# Split
df = pd.read_csv('D://coding//udemy_resources//python for ml//DATA//Advertising.csv')
X = df.drop('sales', axis=1)
y = df['sales']

# Save a validation set. Just run another split
X_train, X_other, y_train, y_other = train_test_split(X, y, test_size=.3, random_state=101)
X_eval, X_test, y_eval, y_test = train_test_split(X_other, y_other, test_size=.3, random_state=101)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_eval = scaler.transform(X_eval)

model = Ridge(alpha=.1)
model.fit(X_train, y_train)
predictions = model.predict(X_eval)
rmse = np.sqrt(mean_squared_error(y_eval, predictions))

# No going back and testing hyperparameters after this 
final_predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
print(rmse)