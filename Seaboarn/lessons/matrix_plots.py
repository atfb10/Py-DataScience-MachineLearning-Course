'''
Author: Adam Forestier
Date: March 29, 2023
Notes:
    - visual equivalent of pivot tables
    - matrix plot displays all the data passed in: it visualizes all numeric columns
    - 2 types:
        * Heatmap - visually displays the distribution of the cell values with a color mapping
                  - heatmap should have all cells be in the same units, so the color mapping makes sense across the data frame
        * clustermap - same visual as heatmap, but first conducts hierarchical clustering to reorganize data into groups
    - I love these
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('country_table.csv')

# heatmap example - perfect heatmap dataframe is labeled index, labeled columns and all numerical data 
df = df.set_index('Countries') # Need labeled index
sns.heatmap(df.drop('Life expectancy', axis=1), lw=.5, annot=True, cmap='viridis') # Displayed data showed all be in the same units life expectancy is number of years. Other columns are a percentage. amnotation shows the actual value.  cmap is the coloring
plt.show()

# clustermap example
sns.clustermap(df.drop('Life expectancy', axis=1), lw=.5, annot=True, col_cluster=False) # col_clustering = False makes clustering based only on index
plt.show()