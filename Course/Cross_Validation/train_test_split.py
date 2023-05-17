'''
author: Adam Forestier
date: April 11, 2023
notes
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=101)

# Scaling
s = StandardScaler()
X_train = s.fit_transform(X_train)
X_test = s.transform(X_test)

# Fit and predict
model = Ridge(alpha=.1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluate
root_mean_squared_error = np.sqrt(mean_squared_error(y_test, predictions))
print(f'RMSE: {root_mean_squared_error}')