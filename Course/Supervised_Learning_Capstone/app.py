import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier) 
from sklearn.tree import (DecisionTreeClassifier, plot_tree)
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, RocCurveDisplay, auc, roc_curve)

# Read in data
df = pd.read_csv('Telco-Customer-Churn.csv')

# TASK: Get a quick statistical summary of the numeric columns with .describe() , you should notice that many columns are categorical, meaning you will eventually need to convert them to dummy variables
stats = df.describe()

# TASK: Confirm that there are no NaN cells by displaying NaN values per feature column.
nulls = df.isna().sum()

# TASK:Display the balance of the class labels (Churn) with a Count Plot.
sns.countplot(x='Churn', data=df)
plt.show()

# TASK: Explore the distrbution of TotalCharges between Churn categories with a Box Plot or Violin Plot.
sns.boxplot(x='Churn', y='TotalCharges', data=df)
plt.show()

# TASK: Create a boxplot showing the distribution of TotalCharges per Contract type, also add in a hue coloring based on the Churn class.
sns.boxplot(x='Contract', y='TotalCharges', data=df, hue='Churn')
plt.show()

# TASK: Create a bar plot showing the correlation of the following features to the class label. Keep in mind, for the categorical features, 
# you will need to convert them into dummy variables first, as you can only calculate correlation for numeric features.
corr_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents','PhoneService', 'MultipleLines', 
 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'InternetService',
   'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']
dummy_df = pd.get_dummies(df[corr_columns]).corr()
corr = dummy_df['Churn_Yes'].sort_values().iloc[1:-1]
sns.barplot(x=corr.index, y=corr.values)
plt.xticks(rotation=90)
plt.show()

# TASK: What are the 3 contract types available
contract_types = df['Contract'].unique()

# TASK: Create a histogram displaying the distribution of 'tenure' column, which is the amount of months a customer was or has been on a customer.
sns.histplot(x='tenure', data=df, bins=50)
plt.show()

# TASK: Now use the seaborn documentation as a guide to create histograms separated by two additional features, Churn and Contract.
sns.displot(data=df, x='tenure', bins=50, col='Contract', row='Churn') # Awesome!
plt.show()

# TASK: Display a scatter plot of Total Charges versus Monthly Charges, and color hue by Churn.
sns.scatterplot(y='TotalCharges', x='MonthlyCharges', hue='Churn', alpha=.7, linewidth=.3, palette='Dark2', data=df)
plt.show()

# TASK: Treating each unique tenure group as a cohort, calculate the Churn rate (percentage that had Yes Churn) per cohort. 
# For example, the cohort that has had a tenure of 1 month should have a Churn rate of 61.99%. 
# You should have cohorts 1-72 months with a general trend of the longer the tenure of the cohort, the less of a churn rate. This makes sense as you are less likely to stop service the longer you've had it.
yes_churn = df.groupby(['Churn', 'tenure']).count().transpose()['Yes']
no_churn = df.groupby(['Churn', 'tenure']).count().transpose()['No']
churn_rate = 100 * yes_churn / (no_churn + yes_churn)
churn_rate = churn_rate.transpose()['customerID']
churn_rate = pd.DataFrame(churn_rate)
churn_rate.columns = ['Churn Rate']
sns.lineplot(x=churn_rate.index, y='Churn Rate', data=churn_rate, color='orange')
plt.show()

'''
Broader Cohort Groups
TASK: Based on the tenure column values, create a new column called Tenure Cohort that creates 4 separate categories:

'0-12 Months'
'12-24 Months'
'24-48 Months'
'Over 48 Months'
'''
def cohort(tenure: int) -> str:
    if tenure < 13:
        return '0-12 Months'
    elif tenure < 25:
        return '12-24 months'
    elif tenure < 49:
        return '24-48 months'
    return 'over 48 months'

df['Tenure Cohorts'] = np.vectorize(cohort)(df['tenure'])

# TASK: Create a scatterplot of Total Charges versus Monthly Charts,colored by Tenure Cohort defined in the previous task.
sns.scatterplot(y='TotalCharges', x='MonthlyCharges', hue='Tenure Cohorts', alpha=.7, linewidth=.3, palette='Dark2', data=df)
plt.show()

# TASK: Create a count plot showing the churn count per cohort.
sns.countplot(x='Tenure Cohorts', data=df)
plt.show()

# TASK: Create a grid of Count Plots showing counts per Tenure Cohort, separated out by contract type and colored by the Churn hue.
sns.catplot(data=df, x='Tenure Cohorts', hue='Churn', kind='count', col='Contract')
plt.show()

# TASK: Perform a train test split, holding out 10% of the data for testing. We'll use a random_state of 101 in the solutions notebook/video.
y = df['Churn']
X = df.drop(['Churn', 'customerID'], axis=1)
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=101)

'''
TASK: Decision Tree Perfomance. Complete the following tasks:

Train a single decision tree model (feel free to grid search for optimal hyperparameters).
Evaluate performance metrics from decision tree, including classification report and plotting a confusion matrix.
Calculate feature importances from the decision tree.
OPTIONAL: Plot your tree, note, the tree could be huge depending on your pruning, so it may crash your notebook if you display it with plot_tree.
'''
clf = DecisionTreeClassifier(max_depth=4)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_pred=y_pred, y_true=y_test)
cm = classification_report(y_pred=y_pred, y_true=y_test)
feat_imp_df = pd.DataFrame(data=clf.feature_importances_, index=X.columns, columns=['Feature Importance']).sort_values('Feature Importance')
feat_imp_df = feat_imp_df[feat_imp_df['Feature Importance'] > 0.001]
sns.barplot(x=feat_imp_df.index, y='Feature Importance', data=feat_imp_df)
plt.xticks(rotation=90)
plt.show()
plot_tree(clf)

# TASK: Create a Random Forest model and create a classification report and confusion matrix from its predicted results on the test set.
clf = RandomForestClassifier(max_depth=3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_pred=y_pred, y_true=y_test)
cm = classification_report(y_pred=y_pred, y_true=y_test)
feat_imp_df = pd.DataFrame(data=clf.feature_importances_, index=X.columns, columns=['Feature Importance']).sort_values('Feature Importance')
feat_imp_df = feat_imp_df[feat_imp_df['Feature Importance'] > 0.001]
sns.barplot(x=feat_imp_df.index, y='Feature Importance', data=feat_imp_df)
plt.xticks(rotation=90)
plt.show()

# TASK: Use AdaBoost or Gradient Boosting to create a model and report back the classification report and plot a confusion matrix for its predicted results
clf = AdaBoostClassifier(max_depth=4)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_pred=y_pred, y_true=y_test)
cm = classification_report(y_pred=y_pred, y_true=y_test)
feat_imp_df = pd.DataFrame(data=clf.feature_importances_, index=X.columns, columns=['Feature Importance']).sort_values('Feature Importance')
feat_imp_df = feat_imp_df[feat_imp_df['Feature Importance'] > 0.001]
sns.barplot(x=feat_imp_df.index, y='Feature Importance', data=feat_imp_df)
plt.xticks(rotation=90)
plt.show()