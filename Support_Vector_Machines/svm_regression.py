'''
Adam Forestier
April 23, 2023
Notes:
    Data
        Label
            * target - refers to the prescence of heart disease in the patient. 0 for no prescence, 1 for prescence
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)