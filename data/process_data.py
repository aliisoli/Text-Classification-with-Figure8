import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    ''' loads and merges the data from messages and categories together,
    then returns the resulting dataframe'''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, how='inner', on='id')
    return df
                             


def clean_data(df):
    '''cleans the data from categories in a useful format and removes duplicates
    '''
    categories = df['categories'].str.split(';', expand=True)
    row = categories.loc[0,:]
    category_colnames = row.apply(lambda x: x[:-2]).values
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
    
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
        
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    '''saves the clean data in a sql database'''
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df.to_sql('Figure8', engine, index=False, if_exists='replace')
    


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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()