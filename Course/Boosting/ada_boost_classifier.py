import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay)
from sklearn.ensemble import AdaBoostClassifier

# Predict which mushrooms are poisonous
# Create cautionary guidelines for people picking mushrooms

df = pd.read_csv('mushrooms.csv')
# print(df.head()) # All categorical - dummies

# Data exploration
sns.countplot(x='class', data=df)
plt.show()
df_description = df.describe().transpose().reset_index().rename({'index': 'feature'}, axis=1).sort_values('unique')
sns.barplot(x='feature', y='unique', data=df_description)
plt.xticks(rotation=90)
plt.ylabel('Unique Categories')
plt.show()

# Split label and features
X = pd.get_dummies(df.drop('class', axis=1), drop_first=True)
y = df['class']

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=101)

# Model
'''
    Parameters:
    estimator: Any = None, -> Chooses decision tree classifier w/ max depth of 1 by default. NOTE: You can base in ANY base estimator you want, but stumps (default) are typically the best
    *,
    n_estimators: Int = 50, -> Number of estimator
    learning_rate: Float = 1,
    algorithm: Any = "SAMME.R",
    random_state: Int | RandomState | None = None,
    base_estimator: Any = "deprecated"
    '''

# NOTE: Why do this? this will give the single best feature to use for predicting whether or not a mushroom is poisonous... 
# NOTE: HUGE, low n_estimators will report back what were the most important features! AKA, what one should look out for!
single_estimator_clf = AdaBoostClassifier(n_estimators=1) 

single_estimator_clf.fit(X_train, y_train)
y_pred = single_estimator_clf.predict(X_test)

# Metrics
report = classification_report(y_true=y_test, y_pred=y_pred)
print('Single stump model')
print(report) # NOTE: SUPER IMPRESSIVE RESULTS for a single feature! This means that this single feature is highly powerful for predicting if a mushroom is poisonous or not
feature_importance = single_estimator_clf.feature_importances_ # NOTE: This will be all 0's with a single 1; because only one feature is being used in this model - thus it has total importance
best_feature_index = single_estimator_clf.feature_importances_.argmax() # NOTE: This will get the location in the array of the feature 
best_feature = X.columns[best_feature_index] # odor_n - whether or not a mushroom has an odor. THIS IS THE MOST IMPORTANT WAY TO DETERMINE IF A MUSHROOM IS EDIBLE OR NOT

# Plot smell
sns.countplot(x='odor', hue='class', data=df) # Almost all mushrooms w/ no smell are edible!
plt.show()

# More robust model - Have 95 columns
# NOTE: HUGELY IMPORTANT!!!!!!! As you add more and more features in/stumps, the most important feature will change, as opposed to when you only use a single feauture!
err = []

# NOTE: Commented this out because it eats my machine
# for i in range(1, 96):
#     clf = AdaBoostClassifier(n_estimators=i)
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     err.append(1 - (accuracy_score(y_true=y_test, y_pred=y_pred)))

# plt.plot(range(1,96), err)
# plt.show() # Shows error rate levels out around 19 or 20 - this means shoud use 19 estimators

# Best classifier
final_clf = AdaBoostClassifier(n_estimators=19)
final_clf.fit(X=X_train, y=y_train)
y_pred = final_clf.predict(X=X_test)
cr = classification_report(y_true=y_test, y_pred=y_pred)
print('---------------------------------------------------------------------')
print('Best n_estimators model')
print(cr) # 100% accurate! I could cry...

# Show me best features w/ n_estimators
best_feature = X.columns[final_clf.feature_importances_.argmax()] # gill-size_n!

# NOTE: Create dataframe w/ feature importance
feature_importance_df = pd.DataFrame(index=X.columns, data=final_clf.feature_importances_, columns=['Importance']) 
feature_importance_df = feature_importance_df[feature_importance_df['Importance'] > 0].sort_values('Importance') # Lots of importance will be 0 since n_estimators = 19, for plot, I just want to see what it took into consideration 

# Text
print('---------------------------------------------------------------------')
print('Feature Importance')
print(feature_importance_df['Importance'])

# Plot
sns.barplot(x=feature_importance_df.index, y='Importance', data=feature_importance_df)
plt.xticks(rotation=90)
plt.show()