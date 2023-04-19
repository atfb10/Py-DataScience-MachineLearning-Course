'''
Adam Forestier
April 19, 2023
Notes:
    Data
        Label
            * target - refers to the prescence of heart disease in the patient. 0 for no prescence, 1 for prescence
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, RocCurveDisplay, auc, roc_curve
)

df = pd.read_csv('heart.csv')

# See df
print(df.head())
description = df.describe()
correlation = df.corr()
info = df.info()

# Visualize
sns.set_style(style='darkgrid')
plt.figure(figsize=(6, 6), dpi=150)
sns.countplot(x='target', data=df)
plt.show()
# sns.pairplot(data=df, hue='target')
# plt.show()
sns.heatmap(correlation, annot=True)
plt.show()

# Seperate Features & Lable
X = df.drop('target', axis=1)
y = df['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.10, random_state=101)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Cross Validated Model
base_model = LogisticRegression(max_iter=1000, solver='liblinear')
param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'l1_ratio': [.1, .3, .5, .75, .85, 9, .95, .99, .999, 1], # Will get warning. Only used when using elastic net is penalty being used
    'C': np.logspace(0, 10, 10), # SciKit Learn recommends logrithmic spacing!
}
grid_model = GridSearchCV(estimator=base_model, param_grid=param_grid)
grid_model.fit(X=X_train, y=y_train)

# Predict
predictions = grid_model.predict(X_test)

# Measure Model Performance
best_parameters = grid_model.best_params_
accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions, labels=grid_model.classes_)
p = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_model.classes_)
p.plot()
cp = classification_report(y_test, predictions)
print(classification_report)
fpr, tpr, thresholds = roc_curve(y_test, predictions)
roc_auc = auc(fpr, tpr)
p = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Heart Disease Result Estimator')
p.plot()
plt.show()

# Predict for patient
patient = [[ 54. ,   1. ,   0. , 122. , 286. ,   0. ,   0. , 116. ,   1. , 3.2,   1. ,   2. ,   2. ]]
predicted_patient_heart_health = grid_model.predict_proba(patient)
print(f'Patient Heart Health: {predicted_patient_heart_health}')