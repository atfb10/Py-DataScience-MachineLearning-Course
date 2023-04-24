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
print(df['quality'].unique())