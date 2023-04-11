'''
Adam Forestier
April 10, 2023
Notes:
'''

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def percent_missing(df: pd.DataFrame) -> pd.Series:
    '''
    arguments: dataframe
    returns: pandas Series with sorted list of missing data by column where a column has missing data
    '''
    percent_missing = 100 * df.isnull().sum() / len(df)
    percent_missing = percent_missing[percent_missing > 0].sort_values(ascending=False)
    return percent_missing

# See the feautres
feature_descriptions = ''
with open('d://coding//udemy_resources//python for ml//DATA//Ames_Housing_Feature_Description.txt', 'r') as f:
    feature_descriptions = f.read()

# print(feature_descriptions)

# Read in data and view missing
df = pd.read_csv('Ames_Housing_NoOutliers.csv')
# print(df.info())

# Remove useless unique id
df = df.drop('PID', axis=1)

# See NaNs
# print(df.isnull().sum()) # Sum of missing
nan_percent = percent_missing(df) # percentage of rows with missing data
plt.figure(figsize=(8,6), dpi=150)
sns.set_style(style='darkgrid')
sns.barplot(x=nan_percent.index, y=nan_percent)
plt.title('Percentage of NaN by column')
plt.xlabel('Column Name')
plt.ylabel('Percent Missing')
plt.xticks(rotation=90)
# plt.savefig('D://coding//udemy//python_datascience_machine_learning//Feature_Engineering//NaN_Percent.jpg')
# plt.show() 

# See columns missing between .000000001 - 1 percent
low_nan_percent = nan_percent[nan_percent < 1.01].sort_values()
# print(low_nan_percent)

# Find missing rows
missing_electrical = df[df['Electrical'].isnull()] 
missing_garage_cars = df[df['Garage Cars'].isnull()]

# Drop missing garage cars & missing electrical, because they are nan and each only make up a single row
df = df.dropna(axis=0, subset=['Electrical', 'Garage Cars'])

# Recalculate percent missing
nan_percent = percent_missing(df)
low_nan_percent = nan_percent[nan_percent < 1.01]
# print(low_nan_percent) 

# See missing basement information. Discover they do not have a basement. Update numeric columns to 0 sq feet and string columns to show they do not have a basement
basement_features = ['Bsmt Half Bath', 'Bsmt Full Bath', 'Bsmt Unf SF','BsmtFin SF 1', 'BsmtFin SF 2', 'Total Bsmt SF']
df[basement_features] = df[basement_features].fillna(0) # Super cool! Create list of column names, apply it to the df, for each of those columns, it performs the function call

# Do the same for basement string columns. NaN should be "None"
basement_features = ['Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2']
df[basement_features] = df[basement_features].fillna('None')

# Handle Mason veneer type
veneer = ['Mas Vnr Type', 'Mas Vnr Area']
df[veneer] = df[veneer].fillna('None')

# Now nan_percent is NaN
nan_percent = percent_missing(df)
low_nan_percent = nan_percent[nan_percent < 1.01]
# print(low_nan_percent) 

# replot
sns.barplot(x=nan_percent.index, y=nan_percent)
plt.title('Percentage of NaN by column')
plt.xlabel('Column Name')
plt.ylabel('Percent Missing')
plt.xticks(rotation=90)
# plt.show()
# print(nan_percent)

# Handle homes with no garage
gar_str_columns = ['Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond']
df[gar_str_columns] = df[gar_str_columns].fillna('None')
df['Garage Yr Blt'] = df['Garage Yr Blt'].fillna(-1)  # No garage
nan_percent = percent_missing(df)
# print(nan_percent)

# Drop useless features
df = df.drop(['Pool QC', 'Misc Feature', 'Alley', 'Fence'], axis=1)
nan_percent = percent_missing(df)
# print(nan_percent)

# Set fireplace quality to none if it does not have one
df['Fireplace Qu'] = df['Fireplace Qu'].fillna('None')
nan_percent = percent_missing(df)

# Handle trickiest... lot frontage. Base it on the neighborhood. Do it using tranform() it essentially combines a group by method with an apply method
df['Lot Frontage'] = df.groupby('Neighborhood')['Lot Frontage'].transform(lambda value: value.fillna(value.mean()))
df['Lot Frontage'] = df['Lot Frontage'].fillna(0)

# Save the improved data frame
df.to_csv('D://coding//udemy//python_datascience_machine_learning//Feature_Engineering//Ames_Housing_Outliers_Missing_Handled.csv')