'''
Author: Adam Forestier
Date: March 24, 2023
Description: Review of total section of course
'''

import numpy as np
import pandas as pd

df = pd.read_csv('hotel_booking_data.csv')

# 1 TODO:how many rows are there
row_count = df.shape[0]

# 2a TODO: is there missing data. 2b which column is missing the most data
missing_data_by_column = df.isna().sum()

# 3 TODO:remove company column
df = df.drop('company', axis=1)

# 4 TODO:What the 5 most common country codes in the dataset
df['country'].value_counts()[:5]

# 5 TODO:What is the name of the person who paid the highest average daily rate (ADR)? How much was their ADR?
biggest_spender = df.loc[df['adr'] == df['adr'].max()][['adr', 'name']] # OPTION 1
biggest_spender_again = df.sort_values('adr', ascending=False)[['name', 'adr']].loc[0] # option 2

# 6 TODO:What is the mean adr across all of the hotel stays in the dataset
adr_mean = round(df['adr'].mean(), 2)

# 7 TODO:what is the average number of nights for a stay across the entire dataset? round to 2 decimal places
df['total_nights_stayed'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
total_night_avg = round(df['total_nights_stayed'].mean(), 2)

# 8 TODO: What is the average total cost for a stay? Round to 2 decimal places
df['total_cost'] = df['adr'] * df['total_nights_stayed']
avg_cost_per_stay = round(df['total_cost'].mean(), 2)

# 9 TODO: Names and emails of people who made exactly 5 special requests
five_special_requests = df[df['total_of_special_requests'] == 5][['name', 'email']]

# 10 TODO: Percentage of hotel stays were repeat guests - use is repeated guest column
repeated_guests = df[df['is_repeated_guest'] == 1]
repeat_guest_percent = round(100 * (repeated_guests.shape[0] / df.shape[0]), 2)

# 11 TODO: 5 most common last name in the dataset. Treat title as last name. example "MD"
def last_name(name: str):
    name_split = name.split(' ')
    return name_split[1]

df['last_name'] = np.vectorize(last_name)(df['name'])
most_common_last_name = df['last_name'].value_counts()[:5]

# 12 TODO: What are the names of the people who booked the most number of children and babies for their stay - only consider reservation
df['total_kids'] = df['children'] + df['babies']
df[['name', 'adults', 'total_kids', 'babies', 'children']].sort_values('total_kids', ascending=False).head()

# 13 TODO: top 3 most common area code. first 3 digits
most_common_areacodes = df['area_code'] = df['phone-number'].apply(lambda phone_number: str(phone_number)[0:3]).value_counts()[:3]

# 14 TODO: inclusive total amount of arrivals between 1-15
arrival_count = df[df['arrival_date_day_of_month'].between(1, 15, inclusive=True)].shape[0]

# 15 TODO: Create a table  for counts for each day of the week that people arrived
def convert(day,month,year):
    return f"{day}-{month}-{year}"

df['date'] = np.vectorize(convert)(df['arrival_date_day_of_month'], df['arrival_date_month'], df['arrival_date_year'])
df['date'] = pd.to_datetime(df['date'])
print(df['date'].dt.day_name().value_counts())