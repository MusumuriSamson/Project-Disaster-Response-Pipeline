# Import libraries
import pandas as pd
from sqlalchemy import create_engine, text

# Load datasets
messages = pd.read_csv("../data/disaster_messages.csv")
categories = pd.read_csv("../data/disaster_categories.csv")

# Merge the datasets
df = messages.merge(categories, on='id')


# Split the values in the categories column and expand it into multiple columns in a new dataframe
categories = df['categories'].str.split(";", expand=True)

# Rename the columns with the values from the first row
row1 = categories.iloc[0]

# Keep everything except the number at the end
column_names = row1.apply(lambda x: x[:-2]).tolist()

# Rename columns in the dataframe
categories.columns = column_names

# Loop through all the column values and replace with the last number it holds
for col in categories:
    categories[col] = categories[col].str[-1]
    # Convert the values to astype(int)
    categories[col] = categories[col].astype(int)

# Drop the categories column in the dataframe
df = df.drop('categories', axis=1)

# Concat df and categories dataframes
df = pd.concat([df, categories], axis=1)


# Drop duplicates in dataframe
df.drop_duplicates(inplace=True)


# Load the data into a database
engine = create_engine('sqlite:///DisasterResponse.db', echo=False)
df.to_sql('Disaster_data', engine, if_exists='replace', index=False)
