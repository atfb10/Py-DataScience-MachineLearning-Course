import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

# Goal predict rock density based upon signal in order to know what cutting tool to used

def true_pred_df(y_true: np.ndarray, y_pred: pd.Series) -> pd.DataFrame:
    '''
    takes in predictions and actual values. returns a data frame with those values as columns, as well as a difference column
    '''
    y_true = y_true.tolist()
    y_pred = y_pred.tolist()
    d = {'Density': y_true, 'Predicted Density': y_pred}
    df = pd.DataFrame(d)
    df['Difference'] = df['Density'] - df['Predicted Density']
    return df

def run_model(model_type: str, model, X_train, y_train, X_test, y_test) -> pd.Series:
    '''
    run model will run a model object and provide results
    1. fit model
    2. performance metrics
    3. plot results
    4. Return predictions for use of true_pred_df func
    '''
    # Fit
    model.fit(X_train, y_train)

    # Results
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_test))
    mae = mean_absolute_error(y_pred=y_pred, y_true=y_test)
    print(f'{model_type} Regression MAE: {mae}')
    print(f'{model_type} Regression RMSE: {rmse}')

    # Plot
    signal_range = np.arange(0, 100)
    signal_pred = model.predict(signal_range.reshape(-1, 1))
    sns.scatterplot(x='Signal', y='Density', data=df)
    plt.plot(signal_range, signal_pred, color='black')
    plt.title(f'{model_type}')
    plt.show()
    return y_pred

# Data
df = pd.read_csv('rock_density_xray.csv')
df.columns = ['Signal', 'Density']

# Can see that linear model will perform terribly
# sns.scatterplot(data=df, x='Signal', y='Density', alpha=.5)
# plt.show()

# Split data (Don't need to scale because of only 1 feature)
X = df['Signal'].values.reshape(-1, 1) # Need to resphape if only grapping 1 feature
y = df['Density']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101, test_size=.1)

# Linear model
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred = lin_model.predict(X_test) # NOTE Almost all predictions are 2.2! just taking average essentially!
mae = mean_absolute_error(y_true=y_test, y_pred=y_pred) # this will appear low ... cannot solely rely on these metrics
rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred)) # this will appear low... this will appear low ... cannot solely rely on these metrics

# Let's investigate this visually (NOTE: Can only do this with 1 feature. for more features, must look at the actual predictions!)
signal = np.arange(0, 100)
signal_pred = lin_model.predict(signal.reshape(-1, 1))
sns.scatterplot(x='Signal', y='Density', data=df)
plt.plot(signal, signal_pred, color='black')
plt.show()

# Let's try other models - use run model

# Polynomial
# NOTE: DANG! Creates polynomial model using pipe to skip multiple steps! This is incredible
# NOTE: Tried and 
pipe = make_pipeline(PolynomialFeatures(degree=6), LinearRegression()) # This will perform well as long as 
y_pred = run_model(model_type='Polynomial', model=pipe, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
true_vs_pred_df = true_pred_df(y_true=y_test, y_pred=y_pred)
# print(true_vs_pred_df.head())

# SVM
model = SVR()
param_grid = {
    'C': [0.01, 0.1, 1, 5, 10, 100],
    'gamma': ['auto', 'scale']
}
model = GridSearchCV(estimator=model, param_grid=param_grid)
# High bias, low variance
y_pred = run_model(model_type='Support Vector Machines', model=model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
true_vs_pred_df = true_pred_df(y_true=y_test, y_pred=y_pred)
# print(true_vs_pred_df.head())

# KNN
k_vals = [1, 3, 5, 10]
for k in k_vals:
    model = KNeighborsRegressor(n_neighbors=k)
    # 5 or 10 looks good. Not too much variance, not too much bias
    y_pred = run_model(model_type=f'{k} Neighbors KNN', model=model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    true_vs_pred_df = true_pred_df(y_true=y_test, y_pred=y_pred)
    # print(true_vs_pred_df.head())

# decision tree
model = DecisionTreeRegressor()
# Too noisy/too much variance (only 1 feature)
y_pred = run_model(model_type='Decision Tree', model=model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
true_vs_pred_df = true_pred_df(y_true=y_test, y_pred=y_pred)
# print(true_vs_pred_df.head())

# Random Forest
model = RandomForestRegressor(n_estimators=10) # only 1 feature so somewhat limited...
# Pretty Good!
y_pred = run_model(model_type='Random Forest', model=model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
true_vs_pred_df = true_pred_df(y_true=y_test, y_pred=y_pred)
# print(true_vs_pred_df.head())

# Gradient Boosting
model = GradientBoostingRegressor() # only 1 feature so somewhat limited...
# similar to random forest, slightly less noise/variance!
y_pred = run_model(model_type='Gradient Boosting', model=model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
true_vs_pred_df = true_pred_df(y_true=y_test, y_pred=y_pred)
# print(true_vs_pred_df.head())

# Ada
model = AdaBoostRegressor() # only 1 feature so somewhat limited...
# Nice! slightly less noise/variance to random forests
y_pred = run_model(model_type='Ada Boosting', model=model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
true_vs_pred_df = true_pred_df(y_true=y_test, y_pred=y_pred)
# print(true_vs_pred_df.head())