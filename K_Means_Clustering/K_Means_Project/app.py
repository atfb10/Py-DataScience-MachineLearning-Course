'''
Adam Forestier
May 15, 2023
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('CIA_Country_Facts.csv')

# TASK: Explore the rows and columns of the data as well as the data types of the columns.
info = df.info()
stats = df.describe().transpose()

# TASK: Create a histogram of the Population column.
# TASK: You should notice the histogram is skewed due to a few large countries, reset the X axis to only show countries with less than 0.5 billion people
sns.histplot(data=df[df['Population'] < 500000000], x='Population')
plt.show()

# TASK: Now let's explore GDP and Regions. Create a bar chart showing the mean GDP per Capita per region (recall the black bar represents std).
sns.barplot(data=df, x=df.Region, y='GDP ($ per capita)', estimator=np.mean, errorbar='sd')
plt.xticks(rotation=90)
plt.show()

# TASK: Create a scatterplot showing the relationship between Phones per 1000 people and the GDP per Capita. Color these points by Region.
sns.scatterplot(data=df, x='GDP ($ per capita)', y='Phones (per 1000)', hue='Region')
plt.show()

# TASK: Create a scatterplot showing the relationship between GDP per Capita and Literacy (color the points by Region). What conclusions do you draw from this plot?
sns.scatterplot(data=df, x='GDP ($ per capita)', y='Literacy (%)', hue='Region')
plt.show()
'''
Literacy and GDP are highly correlated
'''

# TASK: Create a Heatmap of the Correlation between columns in the DataFrame.
sns.heatmap(df.corr())
plt.show()

# TASK: Seaborn can auto perform hierarchal clustering through the clustermap() function. Create a clustermap of the correlations between each column with this function.
sns.clustermap(df.corr())
plt.show()

# TASK: Report the number of missing elements per column.
missing_vals = df.isna().sum()
# print(missing_vals)

# TASK: What countries have NaN for Agriculture? What is the main aspect of these countries?
temp = list(df[df['Agriculture'].isna()]['Country'])

'''
all except western sierra and greenland are tiny islands. fill w/ 0
'''


# TASK: You should have noticed most of these countries are tiny islands, with the exception of Greenland and Western Sahara. 
# Go ahead and fill any of these countries missing NaN values with 0, since they are so small or essentially non-existant.
# df = df[df['Country'].isin(temp)].fillna(0)
df[df['Agriculture'].isnull()] = df[df['Agriculture'].isnull()].fillna(0)

# TASK: Now check to see what is still missing by counting number of missing elements again per feature:
missing_vals = df.isna().sum()

# TASK: Notice climate is missing for a few countries, but not the Region! Let's use this to our advantage. Fill in the missing Climate values based on the mean climate value for its region.
df['Climate'] = df['Climate'].fillna(df.groupby('Region')['Climate'].transform('mean')) 

# TASK: It looks like Literacy percentage is missing. Use the same tactic as we did with Climate missing values and fill in any missing Literacy % values with the mean Literacy % of the Region.
df['Literacy (%)'] = df['Literacy (%)'].fillna(df.groupby('Region')['Literacy (%)'].transform('mean')) 

# TASK: Optional: We are now missing values for only a few countries. Go ahead and drop these countries OR feel free to fill in these last few remaining values with any preferred methodology. For simplicity, we will drop these.
df = df.dropna()

# TASK: It is now time to prepare the data for clustering. The Country column is still a unique identifier string, so it won't be useful for clustering, since its unique for each point. Go ahead and drop this Country column.
X = df.drop('Country', axis=1)

# TASK: Now let's create the X array of features, the Region column is still categorical strings, 
# use Pandas to create dummy variables from this column to create a finalzed X matrix of continuous features along with the dummy variables for the Regions.
X = pd.get_dummies(X)

# TASK: Due to some measurements being in terms of percentages and other metrics being total counts (population), we should scale this data first. Use Sklearn to scale the X feature matrics.
sclr = StandardScaler()
scaled_X = sclr.fit_transform(X)

# TASK: Use a for loop to create and fit multiple KMeans models, testing from K=2-30 clusters. Keep track of the Sum of Squared Distances for each K value, then plot this out to create an "elbow" plot of K versus SSD. 
ssd = {}
for i in range(2, 30):
    m = KMeans(n_clusters=i, max_iter=100)
    m.fit(scaled_X)
    ssd[i] = m.inertia_

plt.plot(list(ssd.keys()), list(ssd.values()), 'o--')
plt.show()

pd.Series(list(ssd.values())).diff().plot(kind='bar')
plt.show()

# Example Interpretation: Choosing K=3
# One could say that there is a significant drop off in SSD difference at K=3 (although we can see it continues to drop off past this). What would an analysis look like for K=3? 
# Let's explore which features are important in the decision of 3 clusters!
m = KMeans(n_clusters=3)
labels = m.fit_predict(scaled_X)
X['cluster'] = labels # Create new column that has predicted cluster label

# Now let's see correlation to cluster label
cluster_corr = X.corr()['cluster'].iloc[:-1].sort_values().plot(kind='bar')
plt.show()

# The best way to interpret this model is through visualizing the clusters of countries on a map!
iso_codes_df = pd.read_csv('country_iso_codes.csv')
iso_codes_dict = iso_codes_df.set_index('Country')['ISO Code'].to_dict()
df['ISO Code'] = df['Country'].map(iso_codes_dict)
df['cluster'] = m.labels_
print(df.head())
fig = px.choropleth(df, 
                    locations='ISO Code',
                    color='cluster',
                    hover_name='Country',
                    )
fig.show()