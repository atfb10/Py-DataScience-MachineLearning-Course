'''
Adam Forestier
April 16, 2023
Notes:
    - predict species by sepal length, sepal width, petal length, petal width
    - NOTE: IMPORTANT!!!!! SciKit Learn has no problem using strings as categories instead of integers!
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, RocCurveDisplay, auc, roc_curve
)
from mpl_toolkits.mplot3d import Axes3D

# Model Investigation
df = pd.read_csv('iris.csv')
description = df.describe()
print(description)
species_count = df['species'].value_counts()
print(species_count)
sns.setstyle=('darkgrid')
sns.countplot(x='species', data=df)
plt.show()
sns.pairplot(df, hue='species')
plt.show()
sns.heatmap(df.corr(), annot=True)
plt.show()

# pull out feature and labels
X = df.drop('species', axis=1)
y = df['species']

# Train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.15, random_state=101)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# model NOTE: IMPORTANT parameter information in comments!
# can add l1, l2 or elasticnet penalties
# l1 ratio - only if testing elasticnet
# multi_class: auto, ovr, or multinomial
    # ovr - a binary problem is fit for each label. "One vs Rest"
    # multinomial - the loss minimized is the multinomial fit across the entire probability distribution, even when the data is binary. 'auto' selects ovr if data is binary or if solver ='liblinear', and otherwise selects multinomial
# Solver: algorithm to use in optimization problem. NOTE: Use SciKit-Learn documentation to determine which solver to use!
# Max iteration - most amount of iterations
# C - penalty. NOTE: Use SciKit Learn documentation to determine!
model = LogisticRegression(solver='saga', multi_class='ovr', max_iter=5000)
param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'l1_ratio': [.1, .3, .5, .75, .85, 9, .95, .99, .999, 1], # Will get warning. Only used when using elastic net is penalty being used
    'C': np.logspace(0, 10, 10) # SciKit Learn recommends logrithmic spacing!
}

grid_model = GridSearchCV(estimator=model, param_grid=param_grid)
grid_model.fit(X=X_train, y=y_train)
predictions = grid_model.predict(X_test)

# Classification Performance Metrics
best_parameters = grid_model.best_params_ # Use these parameters to create model without having to go all iterations
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_true=y_test, y_pred=predictions)
cm = confusion_matrix(y_test, predictions, labels=grid_model.classes_)
p = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_model.classes_)
plt.title('Iris Species Confusion Matrix')
plt.show()

# Magic function to plot multiclass ROC. Thanks SciKit Learn!
# NOTE:
def plot_multiclass_roc(classifier, X_test, y_test, n_classes, figsize=(5,5)):
    y_score = classifier.decision_function(X_test)

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %i' % (roc_auc[i], i))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()


# Call function to plot
plot_multiclass_roc(classifier=grid_model, X_test=X_test, y_test=y_test, n_classes=3)