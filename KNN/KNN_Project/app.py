'''
Adam Forestier
April 20, 2023
Notes:
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

# load the data
df = pd.read_csv('sonar.all-data.csv')
df_corr = df.corr()
# create heatmap of correlation between different frequency responses
sns.set_style(style='darkgrid')
sns.heatmap(df_corr)
plt.show()

# TASK: What are the top 5 correlated frequencies with the target\label?
def label_dummy(label: str) -> int:
    if label == 'R':
        return 0
    return 1
df['Label_Binary'] = np.vectorize(label_dummy)(df['Label'])
df_corr = abs(df.corr()['Label_Binary'].sort_values())
print(df_corr[-6:])

# Split 90% for train, 10% for test
X = df.drop(['Label_Binary', 'Label'], axis=1)
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.10, random_state=101)

# TASK: Create a PipeLine that contains both a StandardScaler and a KNN model
scaler = StandardScaler()
knn = KNeighborsClassifier()
operations = [('scaler', scaler), ('knn', knn)]
pipe = Pipeline(steps=operations)
param_grid = {
    'knn__n_neighbors': list(range(1, 30)),
    'knn__metric': ['minkowski']
}
model = GridSearchCV(pipe, param_grid=param_grid, cv=10, scoring='accuracy')
model.fit(X_train, y_train)

# Task: show best parameters
best_params = model.best_estimator_.get_params()

# (HARD) TASK: Using the .cv_results_ dictionary, see if you can create a plot of the mean test scores per K value
results_df = pd.DataFrame(model.cv_results_)
sns.lineplot(data=results_df, x='param_knn__n_neighbors', y='mean_test_score', marker='o', markersize=5, markerfacecolor='red')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.show()

# TASK: Using the grid classifier object from the previous step, get a final performance classification report and confusion matrix.
y_pred = model.predict(X_test)
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
p = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
p.plot()
plt.show()
cr = classification_report(y_true=y_test, y_pred=y_pred)
print(cr)