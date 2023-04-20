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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline

# Read in
df = pd.read_csv('d://coding//udemy_resources//python for ml//DATA//gene_expression.csv')
X = df.drop('Cancer Present', axis=1)
y = df['Cancer Present']

# Plot
sns.scatterplot(data=df, x='Gene One', y='Gene Two', hue='Cancer Present', alpha=.5, style='Cancer Present')
plt.show()
sns.pairplot(df, hue='Cancer Present')
plt.show()
sns.heatmap(df.corr(), annot=True)
plt.show()

# Split & Scale NOTE: Always scale for KNN!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=101)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Base model
knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
acc = accuracy_score(y_true=y_test, y_pred=y_pred)
cm = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=knn_model.classes_)
p = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn_model.classes_)
p.plot()
plt.show()
report = classification_report(y_true=y_test, y_pred=y_pred)
print(f'Classification Report: \n{report}')
# print('-----------------------')
# print('Elbow Method')

# Elbow Method
test_error_rates = {'K': [], 'Error Rate': []}
for k in range(1, 30):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    test_error = 1 - accuracy_score(y_true=y_test, y_pred=y_pred)
    test_error_rates['K'].append(k)
    test_error_rates['Error Rate'].append(test_error)

# for key, value in test_error_rates.items():
#     print(f'K: {key} - accuracy: {value}')
elbow_df = pd.DataFrame(test_error_rates)
sns.set_style(style='darkgrid')
sns.lineplot(data=elbow_df, x='K', y='Error Rate')
plt.show()

print('-----------------------')
print('Pipeline GridSearch Model')

# PIPELINE --> Gridsearch
scaler = StandardScaler()
knn = KNeighborsClassifier()

# pipeline
operations = [('scaler', scaler), ('knn', knn)]
pl = Pipeline(steps=operations)
k_vals = list(range(1, 20))

'''
NOTE: If parameter grid is going inside a PipeLine object, you parameter name needs to be specified in the following manner
* chose_string_name + two underscores + parameter key name
* model_name + __ + parameter name
* Example: knn_model + __ + n_neighbors
* NOTE: This is what it should look like! knn_model__n_neighbors
'''
param_grid = {
    'knn__n_neighbors':k_vals,
    'knn__metric':['minkowski']
}

# NOTE Amazing! Does the the scaling for me! The Pipeline will auto scale any data I pass to the model
cv_classifier = GridSearchCV(pl, param_grid=param_grid, cv=5, scoring='accuracy')
cv_classifier.fit(X_train, y_train)
y_pred = cv_classifier.predict(X_test)
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
p = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cv_classifier.classes_)
p.plot()
plt.show()
params = cv_classifier.best_estimator_.get_params()
print(params)
cr = classification_report(y_true=y_test, y_pred=y_pred)
print(cr)

print('-----------------------')
print('Cross Validated Model')

# -> My implementation
base_model = KNeighborsClassifier()
param_grid = {
    'n_neighbors': list(range(1, 10))
}
grid_model = GridSearchCV(estimator=base_model, param_grid=param_grid)
# best_params = grid_model.best_params_
# print(f'best parameters: {best_params}')
grid_model.fit(X=X_train, y=y_train)
y_pred = grid_model.predict(X=X_test)
acc = accuracy_score(y_true=y_test, y_pred=y_pred)
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
p = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_model.classes_)
p.plot()
plt.show()
cp = classification_report(y_true=y_test, y_pred=y_pred)
print(f'Classification Report: \n{cp}')
print('-------------------------------------------------')