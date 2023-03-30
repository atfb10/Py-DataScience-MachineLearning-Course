import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('application_record.csv')

# Plot 1 
employed = df[df['DAYS_EMPLOYED'] < 0]
employed['DAYS_EMPLOYED'] = employed['DAYS_EMPLOYED'] *-1
employed['DAYS_BIRTH'] = employed['DAYS_BIRTH'] *-1
sns.scatterplot(data=employed, x='DAYS_BIRTH', y='DAYS_EMPLOYED', alpha=.03, lw=0)

# Plot 2
df['YEARS'] = (-1 * df['DAYS_BIRTH']) / 365
sns.displot(data=df, x='YEARS', bins=20, color='red', alpha=.8)

# Plot 3
half = int(len(df) / 2)
lower_income = df.sort_values(by=['AMT_INCOME_TOTAL'])
lower_income = lower_income.iloc[0:half]
sns.boxplot(data=lower_income, x='NAME_FAMILY_STATUS', y='AMT_INCOME_TOTAL', hue='FLAG_OWN_REALTY')

# Plot 4
sns.heatmap(df.drop('FLAG_MOBIL', axis=1).corr(), cmap='viridis')