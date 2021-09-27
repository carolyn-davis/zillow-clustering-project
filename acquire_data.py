import numpy as np
import pandas as pd
from env import host, user, password


import os

def get_connection(db, user = user, host = host, password = password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
    
    
def get_all_zillow_data():
    '''
    This function gets the zillow data needed to predict single unit properities.
    '''
    file_name = 'zillow.csv'
    if os.path.isfile(file_name):
        return pd.read_csv(file_name)
    
    else:
        query = '''
        select * 
        from predictions_2017
        left join properties_2017 using(parcelid)
        left join airconditioningtype using(airconditioningtypeid)
        left join architecturalstyletype using(architecturalstyletypeid)
        left join buildingclasstype using(buildingclasstypeid)
        left join heatingorsystemtype using(heatingorsystemtypeid)
        left join propertylandusetype using(propertylandusetypeid)
        left join storytype using(storytypeid)
        left join typeconstructiontype using(typeconstructiontypeid)
        where latitude is not null and longitude is not null
                '''
    df = pd.read_sql(query, get_connection('zillow'))  
    
     #replace white space with nulls
    df = df.replace(r'^\s*$', np.NaN, regex=True)
    
    df.to_csv(file_name, index = False)
    return df



df = get_all_zillow_data()


# =============================================================================
#                 FUNCTIONS FOR SUMMARIZING THE DATA 
# =============================================================================


def overview(df):
    '''
    This function returns the shape and info of the df. It also includes a breakdown of the number of unique values
    in each column to determine which are categorical/discrete, and which are numerical/continuous. Finally, it returns
    a breakdown of the statistics on all numeric columns.
    '''
    print(f'This dataframe has {df.shape[0]} rows and {df.shape[1]} columns.')
    print('----------------------------------')
    print('')
    print(df.info())
    print('----------------------------------')
    print('')
    print('Unique value counts of each column')
    print('')
    print(df.nunique())
    print('----------------------------------')
    print('')
    print('Stats on Numeric Columns')
    print('')
    print(df.describe())
    
    
    

#--------------------------------------------------------------------------
#Function for Summdarizing columns with Nulls Values:

def cols_with_null_rows(df):
    '''
    Takes in a DataFrame and returns a DataFrame that contains summary
    statistics for the count and percent of rows that are missing from
    each column in the DataFrame passed in
    '''

    # sort columns into alphabetical order
    cols = list(df)
    cols.sort()
    # create empty DataFrame to store results
    missing_df = pd.DataFrame()
    # start loop to calculate missing values from each column
    for col in cols:
        rows_missing = df[col].isnull().sum()
        total_rows = df[col].shape[0]
        missing_row_dict = {'':col, 'num_rows_missing':f'{rows_missing:.0f}',
                    'pct_rows_missing':f'{(rows_missing / total_rows):.2%}'}
        missing_df = missing_df.append(missing_row_dict, ignore_index=True)
    # assign columns to index to improve legibility
    missing_df = missing_df.set_index('')

    return missing_df

cols_with_null_rows(df)

#---------------------------------------------------------------------------

#Function for rows with nulls in their designated column


def rows_with_null_cols(df):
    '''
    Takes in a DataFrame and returns a DataFrame that contains summary
    statistics for the count and percent of null values in any column
    within that row
    '''

    # define number of cols missing from each row
    num_cols_missing = df.isnull().sum(axis=1)
    # get total number of cols
    total_cols = df.shape[1]
    # get percent of missing cols for each row
    pct_cols_missing = (num_cols_missing / total_cols)
    # create DataFrame from dictionary of missing values
    missing_df = pd.DataFrame({'num_cols_missing':num_cols_missing,
                    'pct_cols_missing':pct_cols_missing}).reset_index()\
                    .groupby(['num_cols_missing', 'pct_cols_missing']).count()\
                    .rename(columns={'index':'num_rows'}).reset_index()\
                    .set_index('num_cols_missing')

    return missing_df

