'''
Adam Forestier
April 23, 2023
Notes:
    Data
        Label
            * quality - refers to the wine being fraudelent or legit 
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.svm import SVC 
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, RocCurveDisplay, auc, roc_curve
)

df = pd.read_csv('wine_fraud.csv')

# TASK: What are the unique variables in the target column we are trying to predict (quality)?
unique_labels = df['quality'].unique()

# TASK: Create a countplot that displays the count per category of Legit vs Fraud. Is the label/target balanced or unbalanced?
sns.countplot(x='quality', data=df)
plt.show()

# TASK: Let's find out if there is a difference between red and white wine when it comes to fraud. Create a countplot that has the wine type on the x axis with the hue separating columns by Fraud vs Legit.
sns.countplot(x='quality', hue='type', data=df)
plt.show()

# TASK: What percentage of red wines are Fraud? What percentage of white wines are fraud?
# TASK: Convert the categorical column "type" from a string or "red" or "white" to dummy variables:
red_wine_fraud_percent = (len(df[(df['quality'] == 'Fraud') & (df['type'] == 'red')]) / len(df[df['type'] == 'red'])) * 100
white_wine_fraud_percent = (len(df[(df['quality'] == 'Fraud') & (df['type'] == 'white')]) / len(df[df['type'] == 'white'])) * 100

# TASK: Calculate the correlation between the various features and the "quality" column. To do this you may need to map the column to 0 and 1 instead of a string.
df['Fraud'] = np.vectorize(lambda quality: 0 if quality == 'Legit' else 1)(df['quality']) # 1 if fraud. 0 if legit
df['Red'] = df['type'].apply(lambda type: 0 if type == 'white' else 1) # 1 if red. 0 if white
fraud_corr = df.corr()['Fraud']
fraud_df = fraud_corr.to_frame(name='Correlation').reset_index()
print(fraud_df)

# TASK: Create a bar plot of the correlation values to Fraudlent wine.
sns.barplot(data=fraud_df, x='index', y='Correlation')
plt.show()

# TASK: Create a clustermap with seaborn to explore the relationships between variables. NOTE: This cooked my weak pc lol. Just gonna use a heatmap so it doesn't explode; code worked great
# sns.clustermap(df.drop(['quality', 'type'], axis=1), lw=.5, annot=True, col_cluster=False) # col_clustering = False makes clustering based only on index
sns.heatmap(df.corr(), annot=True)
plt.show()

# TASK: Separate out the data into X features and y target label ("quality" column)
X = df.drop(['quality', 'type'], axis=1)
y = df['quality']

# TASK: Perform a Train|Test split on the data, with a 10% test size.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=101)

# TASK: Scale the X train and X test data.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# TASK: Create an instance of a Support Vector Machine classifier. NOTE: Imbalanced between Legit and Fraud for y label
# TASK: Use a GridSearchCV to run a grid search for the best C and gamma parameters.
clf = SVC(class_weight='balanced', C=.001, gamma='scale', kernel='linear')
c = [.001, .01, .1, .5, 1]
degree = [1, 2, 3, 4]
param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': c,
    'gamma': ['scale', 'auto'],
    'degree': degree,
}
grid_clf = GridSearchCV(estimator=clf, param_grid=param_grid)
grid_clf.fit(X_train, y_train)
best_params = grid_clf.best_params_# {'C': 0.001, 'degree': 1, 'gamma': 'scale', 'kernel': 'linear'}

# TASK: Display the confusion matrix and classification report for your model.
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
p = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
p.plot()
plt.show()
class_report = classification_report(y_true=y_test, y_pred=y_pred)
print(class_report)

# The results seem wayyyy too good to be true...