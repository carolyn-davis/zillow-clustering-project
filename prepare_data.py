#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 12:29:22 2021

@author: carolyndavis
"""
# =============================================================================
#                             DATA PREPARATIOM
# =============================================================================
import acquire_data as a 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


import math

df = a.get_all_zillow_data()



# =============================================================================
#                 Functions for Removing the Null Values 
# =============================================================================

# def drop_cols_null(df, max_missing_rows_pct=0.25):
#     '''
#     Takes in a DataFrame and a maximum percent for missing values and
#     returns the passed DataFrame after removing any colums missing the
#     defined max percent or more worth of rows
#     '''
    
#     # set threshold for axis=1 and drop cols
#     thresh_col = math.ceil(df.shape[0] * (1 - max_missing_rows_pct))
#     df = df.dropna(axis=1, thresh=thresh_col)

#     return df

# #this function deleted 039 columns that contained null values 

# new_df = drop_cols_null(df)




# def drop_rows_null(df, max_missing_cols_pct=0.25):
#     '''
#     Takes in a DataFrame and a maximum percent for missing values and
#     returns the passed DataFrame after removing any rows missing the
#     defined max percent or more worth of columns
#     '''
    
#     # set threshold for axis=0 and drop rows
#     thresh_row = math.ceil(df.shape[1] * (1 - max_missing_cols_pct))
#     df = df.dropna(axis=0, thresh=thresh_row)
    
#     return df


# newer_df = drop_rows_null(new_df)
# #this function deleted 8 rows that contain null values 


# #This is the combination of the two functions above
# def drop_null_values(df, max_missing_rows_pct=0.25, max_missing_cols_pct=0.25):
#     '''
#     Takes in a DataFrame and maximum percents for missing values in
#     columns and rows and returns the passed DataFrame after first
#     removing any columns missing the defined max percent or more worth
#     of rows then removing rows missing the defined max percent or more
#     worth of columns
#     '''
    
#     # drop columns with null values for passed percent of rows
#     df = drop_cols_null(df, max_missing_rows_pct)
#     # drop rows with null values for passed percent of columns
#     df = drop_rows_null(df, max_missing_cols_pct)
    
#     return df

# newest = drop_null_values(newer_df)



# =============================================================================
#                 Removing the Outliers
# =============================================================================

# def drop_outliers(df, col_list, k=1.5):
#     '''
#     This function takes in a dataframe and removes outliers that are k * the IQR
#     '''
#     # col_list = ['bathroomcnt', 'bedroomcnt', 'calculatedbathnbr',
#     #     'calculatedfinishedsquarefeet', 'fips', 'fullbathcnt', 'latitude',
#     #     'longitude', 'lotsizesquarefeet', 'roomcnt', 'taxamount', 'logerror',
#     #     'LA', 'Orange', 'Ventura', 'age', 'taxrate', 'acres', 'bath_bed_ratio',
#     #     'abs_logerror']
#     for col in col_list:

#         q_25, q_75 = df[col].quantile([0.25, 0.75])
#         q_iqr = q_75 - q_25
#         q_upper = q_75 + (k * q_iqr)
#         q_lower = q_25 - (k * q_iqr)
#         df = df[df[col] > q_lower]
#         df = df[df[col] < q_upper]
        
#     return df 

# col_list = ['bathroomcnt', 'bedroomcnt', 'calculatedbathnbr',
#         'calculatedfinishedsquarefeet', 'fips', 'fullbathcnt', 'latitude',
#         'longitude', 'lotsizesquarefeet', 'roomcnt', 'taxamount', 'logerror',
#         'taxrate', 'acres', 'bath_bed_ratio',
#         'abs_logerror']




# a = drop_outliers(newest, col_list)
# newest.columns


# ['propertylandusetypeid', 
#  'parcelid', 
#  'id', 
#  'logerror',      TARGET
#  'transactiondate',
# 'id.1',
#  'bathroomcnt',
#  'bedroomcnt',
# 'calculatedbathnbr',
#  'calculatedfinishedsquarefeet',
#  'finishedsquarefeet12',
#  'fips',
#  'fullbathcnt', 
#  'latitude',
#  'longitude',
# 'lotsizesquarefeet',
#  'propertycountylandusecode',
#  'rawcensustractandblock', 
#  'regionidcity'
#  , 'regionidcounty',
# 'regionidzip', 
# 'roomcnt', 
# 'yearbuilt', 
# 'structuretaxvaluedollarcnt',
# 'taxvaluedollarcnt', 
# 'assessmentyear', 
# 'landtaxvaluedollarcnt',
# 'taxamount', 
# 'censustractandblock', 
# 'propertylandusedesc'],
 



def drop_columns(df):
    '''
    This function takes in a pandas DataFrame, and a list of columns to drop,
    and returns a DataFrame after dropping the columns
    '''
    # This list needs to be updated for each DataFrame
 #    col_list = ['propertylandusetypeid', 'id', 'finishedsquarefeet12', 'propertycountylandusecode', 'transactiondate', 'propertylandusedesc', 'censustractandblock', 'regionidcity', 'regionidzip', 'yearbuilt', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'assessmentyear',
 # 'landtaxvaluedollarcnt', 'rawcensustractandblock']
    
    df = df.drop(columns=col_list)
    
    return df


col_list = ['propertylandusetypeid', 'id', 'transactiondate', 'id.1', 'calculatedbathnbr',
        'finishedsquarefeet12', 'propertycountylandusecode', 'rawcensustractandblock', 'regionidcity',
        'regionidzip', 'assessmentyear', 'structuretaxvaluedollarcnt', 'censustractandblock',
        'propertylandusedesc']





zillow_df = drop_columns(df)







# =============================================================================
#                     CHECKING OUT THE NULL VALUES 
# =============================================================================

def nulls_by_col(df):
    num_missing = df.isnull().sum()
    print(type(num_missing))
    rows = df.shape[0]
    prcnt_miss = num_missing / rows * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': prcnt_miss})
    return cols_missing

def nulls_by_row(df):
    num_missing = df.isnull().sum(axis=1)
    prcnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': prcnt_miss})\
    .reset_index()\
    .groupby(['num_cols_missing', 'percent_cols_missing']).count()\
    .rename(index=str, columns={'index': 'num_rows'}).reset_index()
    return rows_missing



def get_nulls(df):
    col =  nulls_by_col(df)
    row =  nulls_by_row(df)
    
    return col, row

get_nulls(zillow_df)
#df.info()
# #looks like there might be some nulls
#  or missing values in the acres col, 
#  full_bath_count,
#  lot_sq_footage, year_built,
#  tax_dollar_value,
#  land_tax_value_dollar,
#  tax_amount


#There are many null values in the cols that need to be taken care of:
    # square_footage
    # full_bath_count
    # lot_sq_footage
    # yearbuilt
    # tax_dollar_value
    # landtaxvaluedollarcnt
    # tax_amount
    
    #the new features as well..
                    
# =============================================================================
#                     Handling Missing Values 
# =============================================================================



def drop_cols_null(df, max_missing_rows_pct=0.25):
    '''
    Takes in a DataFrame and a maximum percent for missing values and
    returns the passed DataFrame after removing any colums missing the
    defined max percent or more worth of rows
    '''
    
    # set threshold for axis=1 and drop cols
    thresh_col = math.ceil(df.shape[0] * (1 - max_missing_rows_pct))
    df = df.dropna(axis=1, thresh=thresh_col)

    return df


def drop_rows_null(df, max_missing_cols_pct=0.25):
    '''
    Takes in a DataFrame and a maximum percent for missing values and
    returns the passed DataFrame after removing any rows missing the
    defined max percent or more worth of columns
    '''
    
    # set threshold for axis=0 and drop rows
    thresh_row = math.ceil(df.shape[1] * (1 - max_missing_cols_pct))
    df = df.dropna(axis=0, thresh=thresh_row)
    
    return df


def drop_null_values(df, max_missing_rows_pct=0.25, max_missing_cols_pct=0.25):
    '''
    Takes in a DataFrame and maximum percents for missing values in
    columns and rows and returns the passed DataFrame after first
    removing any columns missing the defined max percent or more worth
    of rows then removing rows missing the defined max percent or more
    worth of columns
    '''
    
    # drop columns with null values for passed percent of rows
    df = drop_cols_null(df, max_missing_rows_pct)
    # drop rows with null values for passed percent of columns
    df = drop_rows_null(df, max_missing_cols_pct)
    
    return df

zillow_df = drop_null_values(zillow_df)
    
    
  # drop all rows with null values for any tax fields
df = df[df.tax_dollar_value.isnull() == False]
df = df[df.land_tax_value_dollar.isnull() == False]
df = df[df.structuretaxvaluedollarcnt.isnull() == False]
df = df[df.tax_amount.isnull() == False]  
    
# =============================================================================
# Renaming Columns and Setting Index to Parcel Id
# =============================================================================
#Setting the index to customer id and renamed the columns for readibility 



def new_index(df):
    '''
    This function takes in the newly acquired zillow dataframe,
    renames the parcelid column as parcel_id,
    and sets this variable as the index.
    '''
    
    df = df.rename(columns={'parcelid': 'parcel_id', 'logerror': 'target', 
                            'bathroomcnt': 'bath_count', 'bedroomcnt': 'bed_count', 
                            'calculatedfinishedsquarefeet': 'square_footage', 'fullbathcnt': 'full_bath_count', 
                            'lotsizesquarefeet': 'lot_sq_footage', 'regionidcounty': 'region_county_id', 
                            'roomcnt': 'room_count', 'yearbuilt': 'year_built', 'taxvaluedollarcnt': 'tax_dollar_value', 
                            'taxamount': 'tax_amount', 'landtaxvaluedollarcnt': 'land_tax_value_dollar'})
    df = df.set_index('parcel_id')
    
    return df

# x.columns

zillow_df = new_index(zillow_df)

# =============================================================================
#                 #Creating new Features: Counties
# =============================================================================


        
def get_counties(df):
    '''
    This function will create dummy variables out of the original fips column. 
    And return a dataframe with all of the original columns except regionidcounty.
    We will keep fips column for data validation after making changes. 
    New columns added will be 'LA', 'Orange', and 'Ventura' which are boolean 
    The fips ids are renamed to be the name of the county each represents. 
    '''
    # create dummy vars of fips id
    county_df = pd.get_dummies(df.fips)
    # rename columns by actual county name
    county_df.columns = ['LA', 'Orange', 'Ventura']
    # concatenate the dataframe with the 3 county columns to the original dataframe
    df = pd.concat([df, county_df], axis = 1)
    # drop regionidcounty and fips columns
    df = df.drop(columns = ['region_county_id'])
    return df

zillow_df = get_counties(zillow_df)


# =============================================================================
# Creating Additional Features: Acres and taxrate
# =============================================================================

def create_features(df):
    '''
    This function is specific to my zillow clustering project. 
    It creates new feature columns to use in clustering exploration
    and modeling.
    '''
    # create age column that uses yearbuilt to show how old the property is
    df['age'] = 2017 - df.year_built
    # create taxrate variable
    df['tax_rate'] = df.tax_amount/df.tax_dollar_value*100
    # create acres variable
    df['acres'] = df.lot_sq_footage/43560
    
    return df

zillow_df = create_features(zillow_df)


# s.info()


# get_nulls(s)

#Still have a lot of nulls: planning to impute missing values for train, validate, test

# =============================================================================
# Dealing with the Values for Latitude and Longitude Columns:
# =============================================================================



# zillow_df['latitude'] = round(zillow_df['latitude'].astype(float), 6)

# zillow_df['latitude'] = round(zillow_df['latitude'].astype(float), 6)
# zillow_df['longitude'] = round(zillow_df['longitude'].astype(float), 6)

#divide by a million to round the values for these columns to the 6th decimal place

zillow_df['latitude'] = (zillow_df['latitude']) / 1000000
zillow_df['longitude'] = (zillow_df['longitude']) / 1000000

# def lat_long(df):
#     zillow_df['latitude'] = (zillow_df['latitude']) / 1000000
#     zillow_df['longitude'] = (zillow_df['longitude']) / 1000000
    
#     return df

# x = lat_long(zillow_df)





# =============================================================================
#                 TRAIN VALIDATE TEST/// THE SPLIT 
# =============================================================================

# def train_validate_test(df, target):
#     """
#     this function takes in a dataframe and splits it into 3 samples,
#     a test, which is 20% of the entire dataframe,
#     a validate, which is 24% of the entire dataframe,
#     and a train, which is 56% of the entire dataframe.
#     It then splits each of the 3 samples into a dataframe with independent variables
#     and a series with the dependent, or target variable.
#     The function returns 3 dataframes and 3 series:
#     X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test.
#     """
#     # split df into test (20%) and train_validate (80%)
#     train_validate, test = train_test_split(df, test_size=0.2, random_state=1221)

#     # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
#     train, validate = train_test_split(train_validate, test_size=0.3, random_state=1221)

#     # split train into X (dataframe, drop target) & y (series, keep target only)
#     X_train = train.drop(columns=[target])
#     y_train = train[target]

#     # split validate into X (dataframe, drop target) & y (series, keep target only)
#     X_validate = validate.drop(columns=[target])
#     y_validate = validate[target]

#     # split test into X (dataframe, drop target) & y (series, keep target only)
#     X_test = test.drop(columns=[target])
#     y_test = test[target]

#     return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test



# train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test = train_validate_test(zillow_df, zillow_df.target)



def train_validate_test(df, target):
    '''
    this function takes in a dataframe and splits it into 3 samples,
    a test, which is 20% of the entire dataframe,
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe.
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable.
    The function returns train, validate, test sets and also another 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test.
    '''
    # split df into test (20%) and train_validate (80%)
    train_validate, test = (train_test_split(df, test_size=.2, random_state=123))
   
    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test



train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test = train_validate_test(zillow_df, target='target')





# =============================================================================
#                             IMPUTING THE NULL VALUES 
# =============================================================================

def impute_null_values(train, validate, test, strategy='mean', col_list=None):
    '''
    Takes in the train, validate, and test DataFrame and imputes either
    all columns the passed column list with the strategy defined in 
    arguments
    strategy='mean' default behavior
    '''

    # if no list is passed, impute all values
    if col_list != None:
        for col in col_list:
            imputer = SimpleImputer(strategy=strategy)
            train[[col]] = imputer.fit_transform(train[[col]])
            validate[[col]] = imputer.transform(validate[[col]])
            test[[col]] = imputer.transform(test[[col]])
    # if col_list is passed, impute only values within
    else:
        for col in list(train):
            imputer = SimpleImputer(strategy=strategy)
            train[[col]] = imputer.fit_transform(train[[col]])
            validate[[col]] = imputer.transform(validate[[col]])
            test[[col]] = imputer.transform(test[[col]])

    return train, validate, test


















# def discrete_vars(df):
#     # get value counts for discrete variables

#     disc_cols = [col for col in df.columns if (df[col].dtype == "object")]

#     for col in disc_cols:
    
#         print(col)
#         print(df[col].value_counts())
#         print()
        
        
# discrete_vars(newest)



# zillow_df = x


# def zillow_engineering(zillow_df):
#     # zillow_df['taxrate'] = round(zillow_df['taxamount']/zillow_df['taxvaluedollarcnt'] * 100 ,2)
#     # zillow_df['transactiondate'] = pd.to_datetime(zillow_df['transactiondate'],dayfirst=True)
#     # zillow_df['transactionmonth'] = zillow_df['transactiondate'].dt.month
#     # zillow_df['log10price'] = np.log10(zillow_df['taxvaluedollarcnt'])
    
#     zillow_df['latitude'] = zillow_df['latitude'].astype(str)
#     zillow_df['longitude'] = zillow_df['longitude'].astype(str)
    
    
#     for i in range(len(zillow_df)):
#         zillow_df['latitude'][i].replace('.','')
#         zillow_df['longitude'][i].replace('.','')
        
#         split1 = zillow_df['latitude'][i][:2]
#         split2 = zillow_df['latitude'][i][2:-2]
#         new = split1 + '.' + split2
#         zillow_df['latitude'][i] = new
        
#         split1 = zillow_df['longitude'][i][:4]
#         split2 = zillow_df['longitude'][i][4:-2]
#         new = split1 + '.' + split2
#         zillow_df['longitude'][i] = new
        
#     zillow_df['latitude'] = round(zillow_df['latitude'].astype(float), 6)
#     zillow_df['longitude'] = round(zillow_df['longitude'].astype(float), 6)
                
#     return zillow_df

# b = zillow_engineering(zillow_df.columns)


# print(type(zillow_df))
# zillow_df.type()
