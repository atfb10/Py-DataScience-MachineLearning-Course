import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay)
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('penguins_size.csv')

# Handle na's & weird data
df = df.dropna()
id = df.index[df['sex']=='.'].tolist()[0]
df.at[id, 'sex'] = 'FEMALE'
X = pd.get_dummies(df.drop('species', axis=1), drop_first=True)
y = df['species']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=101)

# Model
'''
max_feature options: {'auto', 'sqrt', 'log2'}, int or float. 
default is 'auto'
if int -> uses that as max features
if float -> considers max features as fraction = max_features * n_features
if auto -> sqrt(n_feautures)
if sqrt -> sqrt(n_feautures)
if log2 -> log2(n_features)
if None -> max_features=n_features

random state - allows model to keep random state so results are same each time is run
'''
forest_clf = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=101)

# Fit & Predict
forest_clf.fit(X_train, y_train)
y_pred = forest_clf.predict(X_test)

# result
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
p = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=forest_clf.classes_)
p.plot()
plt.show()
cr = classification_report(y_true=y_test, y_pred=y_pred)
print(cr)

# See most important features
print('feature importance')
imp_df = pd.DataFrame(index=X.columns, data=forest_clf.feature_importances_, columns=['feature importance']).sort_values('feature importance')
print(imp_df)