'''
Adam Forestier
April 23, 2023
Notes:
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
from sklearn.tree import (DecisionTreeClassifier, plot_tree)

# Read in data
df = pd.read_csv('penguins_size.csv')
species = df['species'].unique()

# Clean data
nulls = df.isnull().sum()
df = df.dropna() # Get rid of all na's because there is few

gentoo_sex_group = df[df['species'] == 'Gentoo'].groupby('sex').describe().transpose() # Figure out which sex "." is
# appears to be either lol. Could drop, but slightly closer to female
# NOTE: Best to use .at for re-assigning single cell! Awesome!
df.at[336, 'sex'] = 'FEMALE' 

sns.pairplot(data=df, hue='species')
plt.show()
sns.catplot(data=df, x='species', y='body_mass_g', kind='box', col='sex')
plt.show()

# NOTE: SO IMPORTANT - in SciKit-Learn - Decision trees CANNOT take in multicategorical data as strings for features! Must make dummies!
# NOTE: CAN keep labels as strings
X = pd.get_dummies(df.drop('species', axis=1), drop_first=True) # Get dummies for ALL Columns except the label!
y = df['species']

# NOTE: DO NOT NEED TO DO FEATURE SCALING FOR DECISION TREES!

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=101)

# default
# NOTE: AWESOME Parameters!
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)

# See what is important!
importance = tree_model.feature_importances_ # NumPy array 
importance_df = pd.DataFrame(index=X.columns, data=importance, columns=['Feature Importance']).sort_values('Feature Importance') # WEIRD feature importance... You have to tell model which features to consider! 
plt.figure(dpi=200)
plot_tree(tree_model, feature_names=X.columns, filled=True) # SO COOL! Visualize the tree!
# plt.show()

# Results
base_pred = tree_model.predict(X_test).tolist()
y_test_list = y_test.tolist()
base_pred_df = pd.DataFrame({'Actual': y_test_list, 'Predicted': base_pred})
report = classification_report(y_true=y_test, y_pred=base_pred)
print('Base Model')
# print(report)
cm = confusion_matrix(y_true=y_test, y_pred=base_pred)
# p = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=tree_model.classes_)
# p.plot()
# plt.show()

def report_tree_model(model: DecisionTreeClassifier, X_train, X_test, y_train, y_test):
    '''
    arguments: unfitted model
    returns: nothing
    description: takes in untrained tree model and reports back all relevant information
    '''

    # Fit
    model.fit(X=X_train, y=y_train)

    # Most important features
    importance = model.feature_importances_
    importance_df = pd.DataFrame(index=X_train.columns, data=importance, columns=['Feature Importance']).sort_values('Feature Importance')
    print(importance_df)
    print('-------------------------------------------------------------')

    # Plot tree
    plot_tree(decision_tree=model, feature_names=X_train.columns, filled=True)
    plt.show()

    # Predict and results
    pred = model.predict(X=X_test)
    cm = confusion_matrix(y_true=y_test, y_pred=pred)
    p = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    p.plot()
    plt.show()
    report = classification_report(y_true=y_test, y_pred=pred)
    print(report)
    print('-------------------------------------------------------------')
    return None

# Default Hyperparameters
print('Decision tree with default hyperparameters')
report_tree_model(model=tree_model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

# Tuned Hyperparameters
max_depth = 3 # Total amount of splits allowed
pruned_tree = DecisionTreeClassifier(max_depth=max_depth)
print(f'Decision Tree with max depth of {max_depth}')
report_tree_model(model=pruned_tree, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

max_leaf_nodes = 3 # Max number of terminal nodes
max_leaf_tree = DecisionTreeClassifier(max_leaf_nodes=3)
print(f'Decision Tree with max leaf nodes of {max_leaf_nodes}')
report_tree_model(model=max_leaf_nodes, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

mathmatical_criterion = 'entropy'
entropy_tree = DecisionTreeClassifier(criterion=mathmatical_criterion) # information gained algorithm. Default is gini impurity
print(f'Decision Tree using criterion {mathmatical_criterion}')
report_tree_model(model=entropy_tree, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

# NOTE: SHOULD ALWAYS CREATE A GRIDSEARCH FOR BEST Hyperparameters