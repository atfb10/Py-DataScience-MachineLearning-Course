'''
Adam Forestier
April 15, 2023
Notes:
    Data
        Features
            * age - Age of participant
            * physical_score - Score achieved during physical exam
        Label
            * test_result - 0 if no pass, 1 if pass
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, RocCurveDisplay, auc, roc_curve
)
from mpl_toolkits.mplot3d import Axes3D

# Explaratory data anaylsis
df = pd.read_csv('hearing_test.csv')
df.columns = ['hearing_test_result' if x == 'test_result' else x for x in df.columns]
description = df.describe()
hearing_test_result_corr = df.corr()['hearing_test_result']
hearing_test_result_counts = df['hearing_test_result'].value_counts()
sns.set(style='darkgrid')
sns.countplot(x='hearing_test_result', data=df)
plt.show()
sns.boxplot(x='hearing_test_result', y='age', data=df)
plt.show()
sns.boxplot(x='hearing_test_result', y='physical_score', data=df)
plt.show()

# Always do a pairplot for classification models
sns.pairplot(df, hue='hearing_test_result')
plt.show()

# Always do heatmap of correlation
colormap = sns.color_palette("plasma")
sns.heatmap(df.corr(), annot=True, cmap=colormap)
plt.show()

# Drill down on pairplot
# sns.scatterplot(data=df, y='physical_score', x='age', hue='hearing_test_result', alpha=.5)
# plt.title('Physical Score by Age')
# plt.show()
# sns.displot(x='age', data=df, kde=True, bins=20, hue='hearing_test_result')
# plt.title('Age Distribution')
# plt.show()
# sns.displot(x='physical_score', data=df, bins=10, kde=True, hue='hearing_test_result')
# plt.title('Physical Score Distribution')
# plt.show()

# 3d scatterplot of both age and physical score on z hearing result
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['age'], df['physical_score'], df['hearing_test_result'], c=df['hearing_test_result'])
plt.ylabel('Physical Score')
plt.xlabel('Age')
plt.show()

# Create model
X = df.drop('hearing_test_result', axis=1)
y = df['hearing_test_result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=101)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# C: inverse of regularization float
# penalty: l1 lasso penalty, l2 ridge, elasticnet 
model = LogisticRegression()
model.fit(X_train, y_train)
coefficients = model.coef_ # Coefficient of how strong of predictor. Higher number (or higher negative number) means stronger predictor
predictions = model.predict(X_test) # 1 value. predicts which class it will belong to
log_probability_predictions = model.predict_log_proba(X_test) # 2 values per row. log probability of belonging to 0 class and log probability of belonging to 1 class
probability_predictions = model.predict_proba(X_test) # 2 values per row. probability of belonging to 0 class and probability of belonging to 1 class

# Classification Performance Metrics
accuracy_score = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions, labels=model.classes_)
p = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
p.plot()
plt.title('Hearing Test Confusion Matrix')
plt.show()

# Classification report is awesome! It reports the precision, recall, f1-score for all values for predicted category
# recall - how many elements of this class are you finding for this class
# precision - how many elements of this class are correctly found for that class
# Support - shows how many elements of that class are actually in the class
# macro avg - actual average 
# weighted avg - weighted avg based upon support
# NOTE: If precision, recall, & f-1 score are all close to accuracy, there is no issue of an imbalanced dataset. If accuracy is much higher, then you have an imbalanced dataset
report = classification_report(y_true=y_test, y_pred=predictions) 
print(classification_report)

# NOTE: IMPORTANT. CAN IMPORT Just precision and recall. Those values are based upon only class 1

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, predictions)
roc_auc = auc(fpr, tpr)
p = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Hearing Test Result Estimator')
p.plot()
plt.show()