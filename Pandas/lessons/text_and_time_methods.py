'''
Author: Adam Forestier
Date: March 23, 2023
Notes:
    - can always write custom functions and apply using .apply(), but many text functions are built in to the str() method in pandas
    - fastest: vectorize. second fastest: .apply() slowest: using the built in str() methods 
    - dt methods are built in date time methods
'''

from datetime import datetime as dt
import numpy as np
import pandas as pd

# Text methods
names = pd.Series(['Adam', 'Hans', 'Ashley', 'Bethany'])

names.str.upper() # Make all names uppercase
# print(names)

tech_finance = ['GOOG,AAPL,AMZN','JPM,BAC,GS']
tickers = pd.Series(tech_finance)
tickers = tickers.str.split(',', expand=True) # Super cool. Split on , but expand = True will turn each element into a value of column. the column is the position in the list after splitting
print(tickers)
print('------------------------')

names = pd.Series(['Adam  ', ' Hans', 'Ashl:ey', 'Bethany  '])
names = names.str.replace(':', '') # Replace
names = names.str.strip() # remove whitespace
names = pd.Series(['Adam  ', ' hans', 'ashl:ey', 'Bethany  '])
names = names.str.replace(':', '').str.strip().str.capitalize() # do it all in one line and capitlize
print(names)
print('------------------------')

# time methods
myyear = 2015
mymonth = 1
myday = 1
myhour = 2
mymin = 30
mysec = 15

mydate = dt(myyear, mymonth, myday)
mydatetime = dt(myyear, mymonth, myday, myhour, mymin, mysec)

myser = pd.Series(['Nov 3, 1990', '2000-01-01', None])

# pd.to_datetime MAGIC!
timeser = pd.to_datetime(myser) # Incredible. It is sooooo flexible when it comees to the type of date format it can ingest and turn into a datetime object
print(timeser)
print('------------------------')
print(f'Timestamp: {timeser[0]}')
euro_date = '10-12-2000'
pd.to_datetime(euro_date, dayfirst=True) # Tell pandas the first part of the string is a date when converting

# handle funky style dating
style_date = '12--Dec--2000'
pd.to_datetime(style_date,format='%d--%b--%Y') # Can pass in any format and Pandas will handle it for you! 

# A holy smokes example
custom_date = '12 of Dec 2000'
print(f'Crazy awesome example: {pd.to_datetime(custom_date)}') # this is unbelievably good!

# Data example
print('\n\n')
sales = pd.read_csv('RetailSales_BeerWineLiquor.csv')
sales['DATE'] = pd.to_datetime(sales['DATE']) # Convert to datetime
print(sales['DATE'][0].year) # Get the year from the first row

# Read in data and turn to datetime immediately
sales = pd.read_csv('RetailSales_BeerWineLiquor.csv', parse_dates=[0]) # Turn to datetime object while reading in the csv! Just specify the column
sales_copy = pd.read_csv('RetailSales_BeerWineLiquor.csv', parse_dates=[0]) # Turn to datetime object while reading in the csv! Just specify the column
print(sales['DATE'])

# resampling - this is like groupby
sales = sales.set_index('DATE')
sales.resample(rule='A').mean() # How did I know to use A? Go to Pandas documentation. Look at time series offset aliases table! Thanks Pandas

# dt method calls example
print(sales_copy['DATE'].dt.month)
print(sales_copy['DATE'].dt.year)