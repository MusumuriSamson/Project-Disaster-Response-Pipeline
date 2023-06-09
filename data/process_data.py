import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    :param messages_filepath: This variable takes in the filepath of the 
    messages.csv to read.
    :param categories_filepath: This variable takes in the filepath of the 
    categories.csv to read.

    :return: The function returns the merged datasets.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories, on='id')

    return df


def clean_data(df):
    """
    :param messages_filepath: This variable takes in the dataframe to clean.


    :return: The function returns the processed dataframe.
    """

    # Split the categories dataframe by ';' and expand into columns.
    categories = df['categories'].str.split(";", expand=True)

    # Rename the columns with the values from the first row
    row1 = categories.iloc[0]

    # Keep everything except the number at the end
    column_names = row1.apply(lambda x: x[:-2]).tolist()

    # Rename columns in the dataframe
    categories.columns = column_names

    # Loop through all the column values and replace with the last number it
    # holds
    for col in categories:
        categories[col] = categories[col].str[-1]
        # Convert the values to astype(int)
        categories[col] = categories[col].astype(int)

    # Concat df and categories dataframes
    df = pd.concat([df, categories], axis=1)

    # Droppping 'child_alone' column because it only has 0's
    df = df.drop('child_alone', axis=1)

    # Drop the categories column in the dataframe
    df = df.drop('categories', axis=1)

    # Since the related column has an addition value; assuming that it is a typo.
    df['related'] = df['related'].apply(lambda x: 1 if x == 2 else x)

    # Drop duplicates in dataframe
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    :param df: This variable takes in the dataframe for which you want to 
    create a table in the database.
    :param database_filename: This variable takes in the filepath to store 
    the database.

    """
    engine = create_engine('sqlite:///' + database_filename, echo=False)

    df.to_sql('Disaster_data', engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
