'''
Author: Adam Forestier
Date: March 28, 2023
Notes:
    - Scatterplots show the relationship between two continuous features
    - Continuous Features - numeric variables that can take any number of values between two values
                          - Examples: Age, height, salary, temperature, prices
    - scatterplots lnie up a set of two continuous features and plot them out as coordinates
    - Example: relationship between employees salaries and sales amount
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('dm_office_sales.csv')
print(df.head())

# example 1
plt.figure(figsize=(8,4))
sns.scatterplot(x='salary', y='sales', data=df, hue='level of education', palette='Dark2') # call seaborn plot. Hue can work on continuous or categorical. palette is a parameter that takes in an argument from matplotlib color maps. Will choose if you do not choose
plt.title('Sales by Salary')
plt.savefig('D:\\coding\\udemy\\python_datascience_machine_learning\\Seaboarn\\img\\scatter1.jpg')
plt.show()

# example 2
plt.figure(figsize=(8,4))
sns.scatterplot(x='salary', y='sales', data=df, size='salary', alpha=.5, style='division') # Change size of dot by salary. alpha is transparency. style is different shapes
plt.title('Sales by Salary')
plt.savefig('D:\\coding\\udemy\\python_datascience_machine_learning\\Seaboarn\\img\\scatter2.jpeg')
plt.show()