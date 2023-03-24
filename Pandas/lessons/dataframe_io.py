'''
Author: Adam Forestier
Date: March 23, 2023
Notes
'''

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# CSV
df = pd.read_csv('example.csv') # read in csv
df.to_csv('D:\\coding\\udemy\\python_datascience_machine_learning\\Pandas\\lessons\\write_to_folder\\example1.csv', index=False) # write to csv file. index=True is default 

# HTML Table - Exciting!
url = "https://en.wikipedia.org/wiki/World_population"
tables = pd.read_html(url) # This is absolutely incredible! Just pass in the URL, then read_html finds all table tags on that url and creates a table for each
print(len(tables)) # Amount of tables scraped
most_populous_df = tables[1] # Create a dataframe from a specific table. I am in love
most_populous_df = most_populous_df.drop(11, axis=0)
most_populous_df = most_populous_df.drop('#', axis=1)
most_populous_df.columns = ['Country', '2000', '2015', '2030 Estimate']
def remove_txt(country: str):
    country = country.replace('[', '')
    country = country.replace('B', '')
    country = country.replace(']', '')
    return country
most_populous_df['Country'] = np.vectorize(remove_txt)(most_populous_df['Country'])
# print(most_populous_df.head())
most_populous_df.to_html('D:\\coding\\udemy\\python_datascience_machine_learning\\Pandas\\lessons\\write_to_folder\\sample.html')

# Excel
df = pd.read_excel('my_excel_file.xlsx', sheet_name='First_Sheet')
print(df)
workbook = pd.read_excel('my_excel_file.xlsx')
#print(workbook.sheet_names) # show names of sheets in the workbook

# This will return a dictionary where sheet is the key and data is the value
workbook = pd.read_excel('my_excel_file.xlsx', sheet_name=None)
print(workbook['First_Sheet']) # First sheet is a key and the dataframe of 'First_Sheet' is the value

df.to_excel('D:\\coding\\udemy\\python_datascience_machine_learning\\Pandas\\lessons\\write_to_folder\\First_Sheet_Example.xlsx', index=False)

''' 
SQL Databases
step 1: figure out what SQL engine you are connecting to: Postgresql, MySql, SQLite3, etc
step 2: install appropriate driver library
step 3: use sqlalchemy library to connect to the SQL database with the driver
step 4: use the driver connection w/ the pandas read_sql method

Pandas can read in entire tables as a DataFrame or actually parse a SQL query through the connection!
''' 
temp_db = create_engine('sqlite:///:memory:') # create a temporary db in my computer's memory
df = pd.DataFrame(data=np.random.randint(low=0, high=101, size=(4, 4)), columns=['A','B','C','D'])
df.to_sql(name='my_table', con=temp_db, index=False) # here we go -> name is the new table. con is the database connection that I am writing the table to. 
# USE THE "if_exists" parameter to handle if the table already exists
# if_exists options: a) 'fail' - default, cause the script/app to fail. b) replace - this will overwrite the table if it exists. tread carefully! c.) append - put the new data at the end of the table

# Read in whole table 
df = pd.read_sql(name='my_table', con=temp_db)

# Read in data with query
query_df = pd.read_sql('SELECT a,c,d FROM my_table WHERE a>48', con=temp_db)