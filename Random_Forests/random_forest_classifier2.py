import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay)
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data_banknote_authentication.csv')
authentic_core = df.corr()['Class'].sort_values()
df = df.rename(columns={'Class': 'Counterfeit'})
df['Counterfeit'] = df['Counterfeit'].map({0: 'Real', 1: 'Counterfeit'}) # Easier to read

# plot 
sns.pairplot(df, hue='Counterfeit')
plt.show()
sns.heatmap(df.corr(), annot=True)
plt.show()

# Features, labels, split
X = df.drop('Counterfeit', axis=1)
y = df['Counterfeit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.15, random_state=101)

# Cross validated random forest NOTE: Always do this to get best parameters. commented out to not cook my poor computer
forest_clf = RandomForestClassifier(random_state=101, oob_score=True)
# param_grid = {
#     'n_estimators': [64, 100, 128, 200],
#     'max_features': [2, 3, 4],
#     'max_depth': [2, 3, 4],
#     'bootstrap': [True, False],
#     'criterion': ['entropy', 'gini']
# }
# grid_clf = GridSearchCV(estimator=forest_clf, param_grid=param_grid)
# grid_clf.fit(X=X_train, y=y_train)

# See best model
# best_params = grid_clf.best_params_ # bootstrap, feataures = 3, n_estimators = 200

rfc = RandomForestClassifier(n_estimators=200, oob_score=True, max_features=2)
rfc.fit(X_train, y_train)

# see results
y_pred = rfc.predict(X_test)
bag = rfc.oob_score_
print(f'OOB: {bag}')
print('-------------------------------------------')
cm = confusion_matrix(y_pred=y_pred, y_true=y_test)
p = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rfc.classes_)
p.plot()
plt.show()
cr = classification_report(y_pred=y_pred, y_true=y_test)
print(cr)

# Elbow method. Just showing errors by missclassification here
err = []
miss_classification = []

for i in range(1, 20):
    rfc = RandomForestClassifier(n_estimators=i)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    acc = accuracy_score(y_pred=y_pred, y_true=y_test)
    err.append(1 - acc)
    n_missed = np.sum(y_pred != y_test) # NOTE: Epic! Compares where predictions do not match up to true values
    miss_classification.append(miss_classification)

# error
plt.plot(range(1, 20), err)
plt.show()