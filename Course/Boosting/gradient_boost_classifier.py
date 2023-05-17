import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay)
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv('mushrooms.csv')

# Data exploration
plt.show()
df_description = df.describe().transpose().reset_index().rename({'index': 'feature'}, axis=1).sort_values('unique')


# Split label and features
X = pd.get_dummies(df.drop('class', axis=1), drop_first=True)
y = df['class']

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=101)

# Model
'''
    loss: Any = "log_loss",
    learning_rate: Float = 0.1, -> shrinks the contribution of each tree. There is a trade off between learning_rate and n_estimators. Lower learning rate is slower, higher is faster
    n_estimators: Int = 100,
    subsample: Float = 1,
    criterion: Any = "friedman_mse",
    min_samples_split: float | int = 2,
    min_samples_leaf: float | int = 1,
    min_weight_fraction_leaf: Float = 0,
    max_depth: int | None = 3,
    min_impurity_decrease: Float = 0,
    init: str | BaseEstimator | None = None,
    random_state: Int | RandomState | None = None,
    max_features: float | Any | int | None = None,
    verbose: Int = 0,
    max_leaf_nodes: Int | None = None,
    warm_start: bool = False,
    validation_fraction: Float = 0.1,
    n_iter_no_change: Int | None = None,
    tol: Float = 0.0001,
    ccp_alpha: float = 0
'''
# NOTE: commented out due to run time. results are in comments below at best_params
# base_clf = GradientBoostingClassifier()
# param_grid = {
#     'n_estimators': [50, 100],
#     'learning_rate': [.05, .1, .15, .2],
#     'max_depth': [3, 4, 5]
# }
# grid_clf = GridSearchCV(estimator=base_clf, param_grid=param_grid)
# grid_clf.fit(X_train, y_train)
# best_params = grid_clf.best_params_ # learning rate: .15, max_depth: 3, n_estimators 100
# print(best_params)


# Final model
final_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=.15, max_depth=3)
final_clf.fit(X=X_train, y=y_train)
y_pred = final_clf.predict(X=X_test)

# Results
cr = classification_report(y_true=y_test, y_pred=y_pred)
print(cr)
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
p = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=final_clf.classes_)
p.plot()
plt.show()
feature_importance = final_clf.feature_importances_
feature_importance_df = pd.DataFrame(index=X.columns, data=feature_importance, columns=['Importance'])
feature_importance_df = feature_importance_df[feature_importance_df['Importance'] > 0.0005].sort_values('Importance')
sns.barplot(x=feature_importance_df.index, y='Importance', data=feature_importance_df)
plt.xticks(rotation=90)
plt.xlabel('Feature')
plt.show() # no odor is the most imporant feature!