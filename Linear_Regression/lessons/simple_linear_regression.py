'''
Adam Forestier
April 7, 2023
notes:
    - SLR limited to 1 x feature (y = mx + b)
    - Example below maps a linear relationship between total amount of advertising spend and resulting sales
'''

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Advertising.csv')

df['total_adv_spend'] = df['TV'] + df['radio'] + df['newspaper']
df = df[['TV', 'radio', 'newspaper', 'total_adv_spend', 'sales']]

sns.regplot(data=df, x='total_adv_spend', y='sales') # Plot using ordinary least squares - Seaborn can plot out simple linear regression using OLS
plt.show()

# best fit line 
# y = mx + b
# y = B1x + B0
X = df['total_adv_spend']
y = df['sales']
b1, b0 = np.polyfit(X, y, deg=1) # calculate b1 and b0

# Range of amount of money one could spend on advertising
potential_spend = np.linspace(0, 500, 100)

# predicted sales amount based on advertising expenditure!
predicted_sales = (b1 * potential_spend) + b0 # y = B1x + B0

# Graph the predicted sales line
sns.scatterplot(data=df, x='total_adv_spend', y='sales')
plt.plot(potential_spend, predicted_sales, color='red')
plt.show()


# see predicted sales if investing 200 into advertising
spend = 200
predicted_sales = round((b1 * spend) + b0, 2)
print(f'Predicted Sales: ${predicted_sales}')

# View polynomial 
b3, b2, b1, b0 = np.polyfit(X, y, deg=3)
pot_spend = np.linspace(0, 500, 100)
pred_sales = (b3 * (potential_spend**3)) + (b2 * (potential_spend**2)) + (potential_spend * b1) + b0
sns.scatterplot(data=df, x='total_adv_spend', y='sales')
plt.plot(pot_spend, pred_sales, color='red')
plt.show()