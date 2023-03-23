'''
author: Adam Forestier
dates: March 22, 2023
notes:
    - apply() apply any custom python function of our own to every row in a series. these functions should return 1 value
'''
import numpy as np 
import pandas as pd

df = pd.read_csv('tips.csv')
reset_df = df

# apply() for single argument
def price_class(price):
    p_class = '$$'
    if price < 10:
        p_class = '$'
    elif price > 20:
        p_class = '$$$'
    return p_class

last_four_cc = df['CC Number'].apply(lambda num: str(num)[-4:]) # This will create a Pandas series of the last 4 digits of credit card numbers. The column name is the argument being passed into the lambda
df['price_class'] = df['total_bill'].apply(price_class) # Create a new column in the dataframe using the apply class

# apply() for multiple arguments
def quality_tipper(total_bill, tip):
    tip_percentage = 100 * (tip / total_bill)
    tip_quality = 'stingy'
    if tip_percentage > 20:
        tip_quality = 'generous'
    return tip_quality

df['tip_quality'] = df[['total_bill', 'tip']].apply(lambda df: quality_tipper(df['total_bill'], df['tip']), axis=1) # apply method on multiple columns in a pandas dataframe

# Vectorize - This is EASIER to type AND FASTER! Really no reason not to use this approach
def inflation(total_bill):
    '''
    i totally just made this up for an example
    '''
    return round(total_bill + (total_bill * .173), 2)

df['first_four_cc'] = np.vectorize(lambda x: str(x)[:4])(df['CC Number']) # Single column with lambda
df['Inflation Price'] = np.vectorize(inflation)(df['total_bill']) # Single column with function
df['tip_quality'] = np.vectorize(quality_tipper)(df['total_bill'], df['tip']) # Another way to apply method on multiple columns. 

# Describing
df = reset_df
df.describe() # Statistical information

# Sorting
df.sort_values('tip') # Ascending
df.sort_values('tip', ascending=False) # ascending
df.sort_values(['tip', 'total_bill']) # Multiple columns

# Searching
df['tip'].max() # Largest
df['tip'].min() # smallest
df['total_bill'].idxmax() # Location of largest value
df['price_per_person'].idxmin() # Location of lowest value
biggest_tip = df.iloc[df['tip'].idxmax()] # get the largest tip by location

# Correlations
df.corr() # Shows how correlated columns are with each other. Max value is 1

# counting
df['sex'].value_counts() # Count male and female
df['day'].unique() # Show what unique values exist in column
df['day'].nunique() # Show number unique values hat exist in column

# Replace
df['sex'] = df['sex'].replace('Female', 'f') # Replace single value in column
df['time'] = df['time'].replace(['Breakfast', 'Lunch', 'Dinner'], ['B', 'L', 'D']) # Replace multiple values in column Replace = Better for single item
my_map = {'Male': 'm'}
df['sex'] = df['sex'].map(my_map) # Replace using mapping for single item. Map = bettter for lots of items
df['sex'] = df['sex'].map(my_map) # Replace using mapping for multiple items. Map = bettter for lots of items
my_map = {
    '$': 'cheap',
    '$$': 'moderate',
    '$$$': 'expensive'
}
df['price_class'] = df['price_class'].map(my_map)
# print(df.head())

# Duplicates
df.duplicated() # Show rows that are duplicates
df.drop_duplicates() # Remove duplicates

# Between
df = reset_df
df['total_bill'].between(10, 20, inclusive=True) # Show boolean for if the total bill row is between 10 - 20
specific_df = df[df['total_bill'].between(10, 20, inclusive=True)] # create filtered dataframe with only bills between 10-20 

# n largest and n smallest
largest_spender_df = df.nlargest(10, 'total_bill') # df with 10 largest spenders
df.nsmallest(5, 'tip') # show 5 smallest tips

# Sampling
df.sample(5) # 5 random rows
df.sample(frac=.1) # 10% of random rows