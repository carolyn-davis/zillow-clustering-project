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

import sklearn.preprocessing

from scipy import stats

from sklearn.metrics import mean_squared_error

from sklearn.metrics import explained_variance_score

from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures

from scipy.stats import pearsonr, spearmanr

# import visualization tools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math



from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans


# import modeling tools
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, explained_variance_score




df = a.get_all_zillow_data()

zillow_df = df 



# =============================================================================
# ESTABLISHING PRoperty Type of Intrerest
# =============================================================================

# create list of single unit propertylandusedesc
single_unit_props = ['Single Family Residential', 'Condominium', 'Mobile Home',
                     'Manufactured, Modular, Prefabricated Homes', 'Townhouse']
# filter for most-likely single unit properties
zillow_df = zillow_df[zillow_df.propertylandusedesc.isin(single_unit_props)]
zillow_df = zillow_df[(zillow_df.bedroomcnt > 0) & (zillow_df.bedroomcnt <= 10)]
zillow_df = zillow_df[(zillow_df.bathroomcnt > 0) & (zillow_df.bathroomcnt <= 10)]




zillow_df.info()
#Acquire complete Takeaways:
    #It was expected that there would be many missing values. It was found many of the observations
    #were lacking more than 75% of their associated data. The missing values will be handled by filling 
    #any whitespace with null values. Decisions will then be made based on the value to target.,
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




zillow_df.info()


# get_nulls(df)
# zillow_df = df
#Replace a whitespace sequence or empty with a NaN value and reassign this manipulation to df:
zillow_df = zillow_df.replace(r'^\s*S', np.nan, regex=True)


#check to see if we actually have true nulls in the dataframe 
zillow_df.info()

#DROP DUPES:
zillow_df = zillow_df.drop_duplicates()


#DROP the Null Values:
    
# zillow_df = zillow_df.dropna(axis=1)

###Insert takeaways on what happened here^^^^



# =============================================================================
# #DRopping Uninteresting Columns:
# =============================================================================
zillow_df.columns
def drop_columns(df, col_list):
    '''
    This function takes in a pandas DataFrame, and a list of columns to drop,
    and returns a DataFrame after dropping the columns
    '''
    # This list needs to be updated for each DataFrame
 #    col_list = ['propertylandusetypeid', 'id', 'finishedsquarefeet12', 'propertycountylandusecode', 'transactiondate', 'propertylandusedesc', 'censustractandblock', 'regionidcity', 'regionidzip', 'yearbuilt', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'assessmentyear',
 # 'landtaxvaluedollarcnt', 'rawcensustractandblock']
    
    df = df.drop(columns=col_list)
    df = df.drop(columns=(df.filter(regex='typeid').columns))
    df = df.drop(columns=(df.filter(regex='regionid').columns))
    return df


col_list = ['id', 'transactiondate',
            'id.1', 'propertycountylandusecode', 'rawcensustractandblock',
            'assessmentyear', 'roomcnt', 'calculatedbathnbr', 'finishedsquarefeet12']

#Dropping id columns, established as unnecessary
#Dropping type_id columns, established as unnecessary
#Dropping all region id columns, using fips for possible region features
#Droppung columns with information, provided in more contextual columns 


zillow_df = drop_columns(zillow_df, col_list)



get_nulls(zillow_df)

#The functions below drop all columns and rows lacking or missing more than 25% 
#of their respective data 

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


get_nulls(zillow_df)
#Tax columns still contain nulls
#Since null count in low for the rows, dropping any tax/quant rows that possess nulls



# drop all rows with null values for any tax fields
zillow_df = zillow_df[zillow_df.taxvaluedollarcnt.isnull() == False]
zillow_df = zillow_df[zillow_df.landtaxvaluedollarcnt.isnull() == False]
zillow_df = zillow_df[zillow_df.structuretaxvaluedollarcnt.isnull() == False]
zillow_df = zillow_df[zillow_df.taxamount.isnull() == False]

get_nulls(zillow_df)

# Nulls left: calculatedfinishedsquarefeet, fullbathcnt, lotsizesquarefeet, yearbuilt, censustractandblock
# Dropping any rows with low count of nulls   
#dropping lotsizesquarefeet, too many nulls, 
zillow_df = zillow_df[zillow_df.fullbathcnt.isnull() == False]
zillow_df = zillow_df[zillow_df.yearbuilt.isnull() == False]
zillow_df = zillow_df[zillow_df.calculatedfinishedsquarefeet.isnull() == False]
zillow_df = zillow_df[zillow_df.censustractandblock.isnull() == False]

zillow_df.info()

#Dropping these columns bc they either have too many nulls or are 
dropped = ['lotsizesquarefeet', 'fullbathcnt']
zillow_df = zillow_df.drop(columns=dropped)

zillow_df.info()
# =============================================================================
# Renaming Columns for Readibility 
# =============================================================================
#Setting the index to customer id and renamed the columns for readibility 
zillow_df.columns


# def new_index(df):
#     '''
#     This function takes in the newly acquired zillow dataframe,
#     renames the parcelid column as parcel_id,
#     and sets this variable as the index.
#     '''
    
#     df = df.rename(columns={'parcelid': 'parcel_id', 'logerror': 'target', 
#                             'bathroomcnt': 'bath_count', 'bedroomcnt': 'bed_count', 
#                             'calculatedfinishedsquarefeet': 'area', 
#                             'censustractandblock': 'census_tract', 
#                             'yearbuilt': 'year_built', 'taxvaluedollarcnt': 'property_value', 
#                             'taxamount': 'tax_amount', 'structuretaxvaluedollarcnt': 'structure_value',
#                             'landtaxvaluedollarcnt': 'land_value'})
#     df = df.set_index('parcel_id')
    
#     return df

# x.columns

# zillow_df = new_index(zillow_df)

zillow_df = zillow_df.rename(columns={'parcelid': 'parcel_id', 'logerror': 'target', 
                            'bathroomcnt': 'bath_count', 'bedroomcnt': 'bed_count', 
                            'calculatedfinishedsquarefeet': 'area', 
                            'censustractandblock': 'census_tract', 
                            'yearbuilt': 'year_built', 'taxvaluedollarcnt': 'property_value', 
                            'taxamount': 'tax_amount', 'structuretaxvaluedollarcnt': 'structure_value',
                            'landtaxvaluedollarcnt': 'land_value'})


zillow_df.info()


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
    # drop  fips columns
    df = df.drop(columns = ['fips'])
    return df

zillow_df = get_counties(zillow_df)



def create_features(df):
    '''
    This function is specific to my zillow clustering project. 
    It creates new feature columns to use in clustering exploration
    and modeling.
    '''
    # create age column that uses yearbuilt to show how old the property is
    # df['age'] = 2017 - df.year_built
    # create taxrate variable
    zillow_df['tax_rate'] = zillow_df.tax_amount/zillow_df.property_value*100
    # create acres variable
    
    zillow_df['latitude'] = (zillow_df['latitude']) / 1000000
    zillow_df['longitude'] = (zillow_df['longitude']) / 1000000
    
    return df


zillow_df = create_features(zillow_df)



zillow_df.info()


# =============================================================================
# ALMOST FINISHED WITH PREP: LOOKING AT  VISUALIZATIONS AND ANY EXTREME OUTLIERS:
# =============================================================================

# visualize distributions
zillow_df.drop(columns=['target', 'parcel_id']).hist(figsize=(30,25))
plt.tight_layout()
plt.show()


# assign columns to remove IQR outliers from
outlier_cols = ['land_value', 'property_value',
                'structure_value', 'tax_amount', 'area']

def shed_iqr_outliers(df, k=1.5, col_list=None):
    '''
    Takes in a DataFrame and optional column list and removes values 
    that are outside of the uppper and lower bounds for all columns or
    those passed within the list
    '''
    
    # if col_list=['list', 'of', 'cols'], apply outlier removal to cols
    # in col_list
    if col_list != None:
        # start loop for each column in col_list
        for col in col_list:
            # find q1 and q3
            q1, q3 = df[col].quantile([.25, .75])
            # calculate IQR
            iqr = q3 - q1
            # set upper and lower bounds
            upper_bound = q3 + k * iqr
            lower_bound = q1 - k * iqr
            # return DataFrame with IQR outliers removed
            df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
    # if col_list=None, apply outlier removal to all cols
    else:
        # start loop for each column in DataFrame
        for col in list(df):
            # find q1 and q3
            q1, q3 = df[col].quantile([.25, .75])
            # calculate IQR
            iqr = q3 - q1
            # set upper and lower bounds
            upper_bound = q3 + k * iqr
            lower_bound = q1 - k * iqr
            # return DataFrame with IQR outliers removed
            df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]

    return df


zillow_df = shed_iqr_outliers(zillow_df, col_list=outlier_cols) 





# visualize distributions after outlier removal
zillow_df.drop(columns=['target', 'parcel_id']).hist(figsize=(30,25))
plt.tight_layout()
plt.show()

# =============================================================================
# PREPARE TAKEAWAYS:
# =============================================================================

#Removal of the outliers improved the distributions extremely well.
#The data now looks normally distributed/expected forits associated column

# zillow_prepared = zillow_df



# =============================================================================
#                               Data Exploration
# =============================================================================
#This function will perform all the steps outlined in data acquisition and preparation,
 # and will output DataFrames containing the X, y split data for train, validate, and test data sets.



# =============================================================================
#                 TRAIN VALIDATE TEST/// THE SPLIT 
# =============================================================================


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

# peep_train = train 
# peep_train['target_bins'] = pd.cut(peep_train.target, [-5, -.2, -.05, .05, .2, 4])


# peep_train.target_bins.value_counts()
# #output \:
# # (-0.05, 0.05]    22769
# # (0.05, 0.2]       4836
# # (-0.2, -0.05]     3945
# # (0.2, 4.0]        1336
# # (-5.0, -0.2]       616
# # Name: target_bins, dtype: int64
# peep_train.columns

# =============================================================================
#                     SCALING THE DATA 
# =============================================================================
# # Utilized a MinMaxScaller for the data to transform each value in the column 
# proprtionately with the desirable range 0 and 1. 
#Additionally we are dealing with different units for values (dollar, sq ft)
train.shape
#(33502, 18)

validate.shape
#(14358, 17)

test.shape
#(11966, 17)




def Min_Max_Scaler(X_train, X_validate, X_test):
    """
    Takes in X_train, X_validate and X_test dfs with numeric values only
    Returns scaler, X_train_scaled, X_validate_scaled, X_test_scaled dfs 
    """
    #Fit the thing
    scaler = sklearn.preprocessing.MinMaxScaler().fit(X_train)
    
    #transform the thing
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), index = X_train.index, columns = X_train.columns)
    X_validate_scaled = pd.DataFrame(scaler.transform(X_validate), index = X_validate.index, columns = X_validate.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index = X_test.index, columns = X_test.columns)
    
    return scaler, X_train_scaled, X_validate_scaled, X_test_scaled


scaler, X_train_scaled, X_validate_scaled, X_test_scaled = Min_Max_Scaler(X_train, X_validate, X_test)


#Made a copy of the y_train series and copied to df
y_train_df = pd.DataFrame(y_train)
# combine target to DataFrame for exploration
train_scaled = pd.concat((X_train_scaled, y_train_df), axis=1)


# =============================================================================
# Initial thoughts and Hypothesis
# =============================================================================
# =============================================================================
# There seems to be a logical correlation to property location with the gleaning that has been done so far. However, a statistical test will help settle whether any correlation is the result of chance. Below you will find the null and alternate hypothesizes for the variables explored.
# 
# A statistical test will be performed and then compared against an established significance level.
# 
# H0: There is no linear correlation between the variables explored and log error.
# 
# Ha: There is a linear correlation between the variables explored and log error.
# alpha = 0.05
# =============================================================================

# Plotting and Statistical Tests
# The variables in the data set will be visualized and compared for analysis of
#  relationship to the target. Any notable observations will be detailed below
#  after investigating.
train_scaled.columns
# create a pairplot for quick glance at variable interaction
sns.pairplot(train_scaled.drop(
    columns=['census_tract', 'LA', 'Ventura', 'Orange', 'latitude', 'longitude',
             'parcel_id', 'year_built'])\
             .sample(n=3000, random_state=19), y_vars=['target'], height=5, aspect=1)

# create heatmap to find any obvious correlations to target

def target_heat(df, target, method='pearson'):
    '''
    Use seaborn to create heatmap with coeffecient annotations to
    visualize correlation between all variables
    '''

    # define variable for corr matrix
    heat_churn = df.corr()[target][:-1]
    # set figure size
    fig, ax = plt.subplots(figsize=(30, 1))
    # define cmap for chosen color palette
    cmap = sns.diverging_palette(h_neg=220, h_pos=13, sep=25, as_cmap=True)
    # plot matrix turned to DataFrame
    sns.heatmap(heat_churn.to_frame().T, cmap=cmap, center=0,
                annot=True, fmt=".1g", cbar=False, square=True)
    #  improve readability of xticks, remove churn ytick
    plt.xticks(ha='right', va='top', rotation=35, rotation_mode='anchor')
    plt.yticks(ticks=[])
    # set title and print graphic
    plt.title(f'Correlation to {target}\n')
    plt.show()




target_heat(train_scaled.drop(
    columns=['census_tract', 'LA', 'Ventura', 'Orange', 'latitude', 'longitude',
             'parcel_id', 'year_built']), 'target')




# =============================================================================
#                     Hypothesis Testing Correlation to Target
# =============================================================================



def corr_test(data, x, y, alpha=0.05, r_type='pearson'):
    '''
    Performs a pearson or spearman correlation test and returns the r
    measurement as well as comparing the return p valued to the pass or
    default significance level, outputs whether to reject or fail to
    reject the null hypothesis
    
    '''
    
    # obtain r, p values
    if r_type == 'pearson':
        r, p = pearsonr(data[x], data[y])
    if r_type == 'spearman':
        r, p = spearmanr(data[x], data[y])
    # print reject/fail statement
    print(f'''{r_type:>10} r = {r:.2g}
+--------------------+''')
    if p < alpha:
        print(f'''
        Due to p-value {p:.2g} being less than our significance level of \
{alpha}, we may reject the null hypothesis 
        that there is not a linear correlation between "{x}" and "{y}."
        ''')
    else:
        print(f'''
        Due to p-value {p:.2g} being greater than our significance level of \
{alpha}, we fail to reject the null hypothesis 
        that there is not a linear correlation between "{x}" and "{y}."
        ''')

# perform statistical tests on strongest correlations according to heatmap
corr_test(train_scaled, 'bed_count', 'target')
corr_test(train_scaled, 'area', 'target')
corr_test(train_scaled, 'bath_count', 'target')
corr_test(train_scaled, 'land_value', 'target')

# =============================================================================
#    pearson r = 0.025     BED  COUNT VS TARGET
# +--------------------+
# 
#         Due to p-value 4e-06 being less than our significance level of 0.05, we may reject the null hypothesis 
#         that there is not a linear correlation between "bed_count" and "target."
#         
#    pearson r = 0.017   SQFOOTAGE VS TARGET
# +--------------------+
# 
#         Due to p-value 0.0014 being less than our significance level of 0.05, we may reject the null hypothesis 
#         that there is not a linear correlation between "area" and "target."
#         
#    pearson r = 0.012   BATH COUNT VS TARGET
# +--------------------+
# 
#         Due to p-value 0.023 being less than our significance level of 0.05, we may reject the null hypothesis 
#         that there is not a linear correlation between "bath_count" and "target."
#         
#    pearson r = -0.025  LAND VALUE VS TARGET
# +--------------------+
# 
#         Due to p-value 6.3e-06 being less than our significance level of 0.05, we may reject the null hypothesis 
#         that there is not a linear correlation between "land_value" and "target."
# 
# =============================================================================

def plot_univariate(data, variable):
    '''
    This function takes the passed DataFrame the requested and plots a
    configured boxenplot and distrubtion for it side-by-side
    '''

    # set figure dimensions
    plt.figure(figsize=(30,8))
    # start subplot 1 for boxenplot
    plt.subplot(1, 2, 1)
    sns.boxenplot(x=variable, data=data)
    plt.axvline(data[variable].median(), color='pink')
    plt.axvline(data[variable].mean(), color='red')
    plt.xlabel('')
    plt.title('Enchanced Box Plot', fontsize=25)
    # start subplot 2 for displot
    plt.subplot(1, 2, 2)
    sns.histplot(data=data, x=variable, element='step', kde=True, color='cyan',
                                line_kws={'linestyle':'dashdot', 'alpha':1})
    plt.axvline(data[variable].median(), color='pink')
    plt.axvline(data[variable].mean(), color='red')
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Distribution', fontsize=20)
    # set layout and show plot
    plt.suptitle(f'{variable} $[n = {data[variable].count():,}]$', fontsize=25)
    plt.tight_layout()
    plt.show()



# look more closely at strongest correlations to log_error
plot_univariate(train_scaled, 'bed_count')
plot_univariate(train_scaled, 'area')
plot_univariate(train_scaled, 'bath_count')
plot_univariate(train_scaled, 'land_value')

# =============================================================================
#         THOUGHTS/OBSERVATIONS/TAKEAWAYS
# =============================================================================
# =============================================================================
# By observing the plot of variables to the target log_error, there is no clear relationship
#  made evident with the individual variables. Looking at the pair plot, all of the variables
#  show a significant range in logerror individually. There is no strong correlation between
#  the target, log error, and the other variables. There is an apparent significantly strong
#  negative correlation between bath_count and land_value with the target. This may be due
#  to a unique interaction of the variables that are having an effect on log_error. Clustering
#  will aid in methods of exploration to define any possible relationships evident between the
#  target and variables presented.
# =============================================================================

# =============================================================================
#                   Cluster Exploration
# =============================================================================
# =============================================================================
# With the use of my initial hypothesis that were previously stated, methods of clustering
#  are utilized to establish any meaningful insight into possible key drivers of the target
#  log_error.
# 
# Iterations will be performed in variable exploration in order to identify clusters that
#  not only identify drivers of the target but perhaps even support future modeling
#  predictions. 
# 
# An analysis of variance test (ANOVA) will aid in cluster comparison. The elbow plot
#  method will be utilized for clustering where k is the point of least change to
#  inertia.
# =============================================================================
def elbow_plot(df, col_list):
    '''
    Takes in a DataFrame and column list to use below method to find
    changes in inertia for increasing k in cluster creation methodology
    '''

    # set figure parameters
    plt.figure(figsize=(30, 15))
    # create series and apply increasing k values to test for inertia
    pd.Series({k: KMeans(k).fit(df[col_list])\
                            .inertia_ for k in range(2, 15)}).plot(marker='*')
    # define plot labels and visual components
    plt.xticks(range(2, 15))
    plt.xlabel('$k$')
    plt.ylabel('Inertia')
    plt.ylim(0,50000)
    plt.title('Changes in Inertia for Increasing $k$')
    plt.show()
    
    
    
# set col_list for cluster formation
col_list_scaled = ['latitude', 'longitude']
# create DataFrame for explored variables
explore_df = train_scaled[col_list_scaled]



# create plot to use elbow method to find best k
elbow_plot(explore_df, col_list_scaled)
#somehwere around k of 5 or 6 is good


# =============================================================================
#                 Exploring Clusters
# =============================================================================

def explore_clusters(df, col_list, k=2):
    '''
    Takes in a DataFrame, column list, and optional integer value for
    k to create clusters for the purpose of exploration, returns a
    DataFrame containing cluster group numbers and cluster centers
    '''

    # create kmeans object
    kmeans = KMeans(n_clusters=k, random_state=19)
    # fit kmeans
    kmeans.fit(df[col_list])
    # store predictions
    cluster_df = pd.DataFrame(kmeans.predict(df[col_list]), index=df.index,
                                                        columns=['cluster'])
    cluster_df = pd.concat((df[col_list], cluster_df), axis=1)
    # store centers
    center_df = cluster_df.groupby('cluster')[col_list].mean()
    
    return cluster_df, center_df, kmeans



# create clusters for exploring log_error relative to location
cluster_df, center_df, kmeans = explore_clusters(explore_df, col_list_scaled, k=5)


#---------------------------------


def plot_clusters(cluster_df, center_df, x_var, y_var):
    '''
    Takes in cluster and centers DataFrame created by explore_clusters
    function and plots the passed x and y variables that make up that
    cluster group with different colors
    '''

    # define cluster_ column for better seaborn interpretation
    cluster_df['cluster_'] = 'cluster_' + cluster_df.cluster.astype(str)
    # set scatterplot and dimensions
    plt.figure(figsize=(28, 14))
    sns.scatterplot(x=x_var, y=y_var, data=cluster_df, hue='cluster_', s=100)
    # plot cluster centers
    center_df.plot.scatter(x=x_var, y=y_var, ax=plt.gca(), s=300, c='k',
                                        edgecolor='w', marker='$\\bar{x}$')
    # set labels and legend, show
    plt.xlabel(f'\n{x_var}\n', fontsize=20)
    plt.ylabel(f'\n{y_var}\n', fontsize=20)
    plt.title('\nClusters and Their Centers\n', fontsize=30)
    plt.legend(bbox_to_anchor=(0.95,0.95), fontsize=20)

    plt.show()

# create plot to view clusters for lat_long_clstr
plot_clusters(cluster_df, center_df, 'latitude', 'longitude')

# set alpha for testing significance
alpha = 0.05
# create DataFrame of samples for ANOVA testing
samples = pd.concat((cluster_df, train.target), axis=1)
# Perform ANOVA one-way test for null hypotesis
F, p = stats.f_oneway(samples[samples.cluster == 0].target,
                      samples[samples.cluster == 1].target,
                      samples[samples.cluster == 2].target,
                      samples[samples.cluster == 3].target,
                      samples[samples.cluster == 4].target)
# print fail or succeed to reject and values returned
if p < alpha:
    state = '✓ May reject'
else:
    state = '𐄂 Fail to reject'
print(f'''
      Stats
+---------------+
| F-value: {F:.2f} |  {state}
| p-value: {p:.2f} |  the null hypothesis.
+---------------+
''')
#Results for ANOVA one-way test: Latitude and longitude

#       Stats
# +---------------+
# | F-value: 2.20 |  𐄂 Fail to reject
# | p-value: 0.07 |  the null hypothesis.


# =============================================================================
# Striking out with the initial hypothesis, below the relationship of count of
#  bedrooms and bathrooms will be clustered with size of lot in acres and explored
#  against the target. The clusters will then be compared using an ANOVA test.
# =============================================================================
# set col_list for cluster formation
col_list_scaled = ['bed_count', 'area']
# create DataFrame for explored variables
explore_df = train_scaled[col_list_scaled]





# create plot to use elbow method to find best k
elbow_plot(explore_df, col_list_scaled)
#k= around 4


# create clusters for exploring log_error relative to location
cluster_df, center_df, kmeans = explore_clusters(explore_df, col_list_scaled, k=5)




# create plot to view clusters for lat_long_clstr
plot_clusters(cluster_df, center_df, 'bed_count', 'area')



# set alpha for testing significance
alpha = 0.05
# create DataFrame of samples for ANOVA testing
samples = pd.concat((cluster_df, train_scaled.target), axis=1)
# Perform ANOVA one-way test for null hypotesis
F, p = stats.f_oneway(samples[samples.cluster == 0].target,
                      samples[samples.cluster == 1].target,
                      samples[samples.cluster == 2].target,
                      samples[samples.cluster == 3].target,
                      samples[samples.cluster == 4].target)
# print fail or succeed to reject and values returned
if p < alpha:
    state = '✓ May reject'
else:
    state = '𐄂 Fail to reject'
print(f'''
      Stats
+---------------+
| F-value: {F:.2f} |  {state}
| p-value: {p:.2f} |  the null hypothesis.
+---------------+
''')


# REsults
#       Stats
# +---------------+
# | F-value: nan |  𐄂 Fail to reject
# | p-value: nan |  the null hypothesis.
# +---------------+



# =============================================================================
# Latitude/ Longitude/ Property Value
# =============================================================================
#Needs Hypothesis
# set col_list for cluster formation
col_list_scaled = ['latitude', 'longitude', 'property_value']
# create DataFrame for explored variables
explore_df = train_scaled[col_list_scaled]





# create plot to use elbow method to find best k
elbow_plot(explore_df, col_list_scaled)
#k= around 5


# create clusters for exploring log_error relative to location
cluster_df, center_df, kmeans = explore_clusters(explore_df, col_list_scaled, k=5)

# cosmetic imports and settings


def plot_three_d_clusters(cluster_df, center_df, x_var, y_var, z_var):
    '''
    Takes in cluster and centers DataFrame created by explore_clusters
    function and creates a three dimesnional plot of the passed x, y,
    and z variables that make up that cluster group with different
    colors
    '''

    # set figure and axes
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111, projection='3d')    
    # set clusters for each cluster passed in arguments
    # set x, y, z for cluster 0
    x0 = cluster_df[cluster_df['cluster'] == 0][x_var]
    y0 = cluster_df[cluster_df['cluster'] == 0][y_var]
    z0 = cluster_df[cluster_df['cluster'] == 0][z_var]
    # set x, y, z for cluster 1
    x1 = cluster_df[cluster_df['cluster'] == 1][x_var]
    y1 = cluster_df[cluster_df['cluster'] == 1][y_var]
    z1 = cluster_df[cluster_df['cluster'] == 1][z_var]
    # set x, y, z for each additional cluster
    if len(center_df) > 2:
        x2 = cluster_df[cluster_df['cluster'] == 2][x_var]
        y2 = cluster_df[cluster_df['cluster'] == 2][y_var]
        z2 = cluster_df[cluster_df['cluster'] == 2][z_var]
    if len(center_df) > 3:
        x3 = cluster_df[cluster_df['cluster'] == 3][x_var]
        y3 = cluster_df[cluster_df['cluster'] == 3][y_var]
        z3 = cluster_df[cluster_df['cluster'] == 3][z_var]
    if len(center_df) > 4:
        x4 = cluster_df[cluster_df['cluster'] == 4][x_var]
        y4 = cluster_df[cluster_df['cluster'] == 4][y_var]
        z4 = cluster_df[cluster_df['cluster'] == 4][z_var]
    if len(center_df) > 5:
        x5 = cluster_df[cluster_df['cluster'] == 5][x_var]
        y5 = cluster_df[cluster_df['cluster'] == 5][y_var]
        z5 = cluster_df[cluster_df['cluster'] == 5][z_var]
        
    # set centers for each cluster passed in arguments
    # set centers for clusters 0, 1
    zero_center = center_df[center_df.index == 0]
    one_center = center_df[center_df.index == 1]
    # set centers for each additional clusters
    if len(center_df) > 2:
        two_center = center_df[center_df.index == 2]
    if len(center_df) > 3:
        three_center = center_df[center_df.index == 3]
    if len(center_df) > 4:
        four_center = center_df[center_df.index == 4]
    if len(center_df) > 5:
        five_center = center_df[center_df.index == 5]
    if len(center_df) > 6:
        six_center = center_df[center_df.index == 6]
        
    # plot clusters and their centers for each cluster passed in arguments
    # plot cluster 0 with center
    ax.scatter(x0, y0, z0, s=100, c='c', edgecolor='k', marker='o',
                                                    label='Cluster 0')
    ax.scatter(zero_center[x_var], zero_center[y_var], zero_center[z_var],
                                    s=300, c='c', marker='$\\bar{x}$')
    # plot cluster 1 with center
    ax.scatter(x1, y1, z1, s=100, c='y', edgecolor='k', marker='o',
                                                    label='Cluster 1')
    ax.scatter(one_center[x_var], one_center[y_var], one_center[z_var],
                                    s=300, c='y', marker='$\\bar{x}$')
    # plot each additional cluster passed in arguments
    if len(center_df) > 2:
        ax.scatter(x2, y2, z2, s=100, c='m', edgecolor='k', marker='o',
                                                    label='Cluster 2')
        ax.scatter(two_center[x_var], two_center[y_var], two_center[z_var],
                                    s=300, c='m', marker='$\\bar{x}$')
    if len(center_df) > 3:
        ax.scatter(x3, y3, z3, s=100, c='k', edgecolor='w', marker='o',
                                                    label='Cluster 3')
        ax.scatter(three_center[x_var],three_center[y_var],three_center[z_var],
                                    s=300, c='k', marker='$\\bar{x}$')
    if len(center_df) > 4:
        ax.scatter(x4, y4, z4, s=100, c='r', edgecolor='k', marker='o',
                                                    label='Cluster 4')
        ax.scatter(four_center[x_var], four_center[y_var], four_center[z_var],
                                    s=300, c='r', marker='$\\bar{x}$')
    if len(center_df) > 5:
        ax.scatter(x5, y5, z5, s=100, c='g', edgecolor='k', marker='o',
                                                    label='Cluster 5')
        ax.scatter(five_center[x_var], five_center[y_var], five_center[z_var],
                                    s=300, c='g', marker='$\\bar{x}$')
    # if len(center_df) > 6:
    #     ax.scatter(x6, y6, z6, s=100, c='b', edgecolor='k', marker='o',
    #                                                 label='Cluster 6')
        ax.scatter(six_center[x_var], six_center[y_var], six_center[z_var],
                                    s=300, c='b', marker='$\\bar{x}$')
        
    # set labels, title, and legend
    ax.set_xlabel(f'\n$x =$ {x_var}', fontsize=15)
    ax.set_ylabel(f'\n$y =$ {y_var}', fontsize=15)
    ax.set_zlabel(f'\n$z =$ {z_var}', fontsize=15)
    plt.title('Clusters and Their Centers', fontsize=30)
    plt.legend(bbox_to_anchor=(0.975,0.975), fontsize=15)

    plt.show()

# create plot to view clusters for lat_long_clstr
plot_three_d_clusters(cluster_df, center_df, 'latitude', 'longitude', 'property_value')



# set alpha for testing significance
alpha = 0.05
# create DataFrame of samples for ANOVA testing
samples = pd.concat((cluster_df, train_scaled.target), axis=1)
# Perform ANOVA one-way test for null hypotesis
F, p = stats.f_oneway(samples[samples.cluster == 0].target,
                      samples[samples.cluster == 1].target,
                      samples[samples.cluster == 2].target,
                      samples[samples.cluster == 3].target,
                      samples[samples.cluster == 4].target)
# print fail or succeed to reject and values returned
if p < alpha:
    state = '✓ May reject'
else:
    state = '𐄂 Fail to reject'
print(f'''
      Stats
+---------------+
| F-value: {F:.2f} |  {state}
| p-value: {p:.2f} |  the null hypothesis.
+---------------+
''')


# OUTPUT: SUCCESS, POSSIBLE FEATURE FOR MODELING
# =============================================================================
#       Stats
# +---------------+
# | F-value: 8.74 |  ✓ May reject
# | p-value: 0.00 |  the null hypothesis.
# +---------------+
# 
# =============================================================================



# add cluster to DataFrame for feature exploration
train_scaled['lat_long_pv_clstr'] = cluster_df.cluster
train_scaled = pd.get_dummies(train_scaled, columns=['lat_long_pv_clstr'], drop_first=True)


# =============================================================================
# BEDCOUNT VS TAX RATE 
# =============================================================================
# set col_list for cluster formation
col_list_scaled = ['bed_count', 'tax_rate']
# create DataFrame for explored variables
explore_df = train_scaled[col_list_scaled]





# create plot to use elbow method to find best k
elbow_plot(explore_df, col_list_scaled)
#k= around 4


# create clusters for exploring log_error relative to location
cluster_df, center_df, kmeans = explore_clusters(explore_df, col_list_scaled, k=5)




# create plot to view clusters for lat_long_clstr
plot_clusters(cluster_df, center_df, 'bed_count', 'tax_rate')



# set alpha for testing significance
alpha = 0.05
# create DataFrame of samples for ANOVA testing
samples = pd.concat((cluster_df, train_scaled.target), axis=1)
# Perform ANOVA one-way test for null hypotesis
F, p = stats.f_oneway(samples[samples.cluster == 0].target,
                      samples[samples.cluster == 1].target,
                      samples[samples.cluster == 2].target,
                      samples[samples.cluster == 3].target,
                      samples[samples.cluster == 4].target)
# print fail or succeed to reject and values returned
if p < alpha:
    state = '✓ May reject'
else:
    state = '𐄂 Fail to reject'
print(f'''
      Stats
+---------------+
| F-value: {F:.2f} |  {state}
| p-value: {p:.2f} |  the null hypothesis.
+---------------+
''')

#OUTPUT: SUCCESS 
#       Stats
# +---------------+
# | F-value: 7.03 |  ✓ May reject
# | p-value: 0.00 |  the null hypothesis.
# +---------------+


# add cluster to DataFrame for feature exploration
train_scaled['bed_taxrate_clstr'] = cluster_df.cluster
train_scaled = pd.get_dummies(train_scaled, columns=['bed_taxrate_clstr'], drop_first=True)




# =============================================================================
# Land Value Vs Census Tract
# =============================================================================
# set col_list for cluster formation
col_list_scaled = ['land_value', 'census_tract']
# create DataFrame for explored variables
explore_df = train_scaled[col_list_scaled]





# create plot to use elbow method to find best k
elbow_plot(explore_df, col_list_scaled)
#k= around 4


# create clusters for exploring log_error relative to location
cluster_df, center_df, kmeans = explore_clusters(explore_df, col_list_scaled, k=5)




# create plot to view clusters for lat_long_clstr
plot_clusters(cluster_df, center_df, 'land_value', 'census_tract')



# set alpha for testing significance
alpha = 0.05
# create DataFrame of samples for ANOVA testing
samples = pd.concat((cluster_df, train_scaled.target), axis=1)
# Perform ANOVA one-way test for null hypotesis
F, p = stats.f_oneway(samples[samples.cluster == 0].target,
                      samples[samples.cluster == 1].target,
                      samples[samples.cluster == 2].target,
                      samples[samples.cluster == 3].target,
                      samples[samples.cluster == 4].target)
# print fail or succeed to reject and values returned
if p < alpha:
    state = '✓ May reject'
else:
    state = '𐄂 Fail to reject'
print(f'''
      Stats
+---------------+
| F-value: {F:.2f} |  {state}
| p-value: {p:.2f} |  the null hypothesis.
+---------------+
''')

#Output: SUCCESS
# =============================================================================
#           Stats
# +---------------+
# | F-value: 10.37 |  ✓ May reject
# | p-value: 0.00 |  the null hypothesis.
# +---------------+
# =============================================================================

# add cluster to DataFrame for feature exploration
train_scaled['land_val_census_clstr'] = cluster_df.cluster
train_scaled = pd.get_dummies(train_scaled, columns=['land_val_census_clstr'], drop_first=True)


# =============================================================================
#                 FEATURE EXPLORATION
# =============================================================================

train_scaled2 = train


scaler2, X_train_scaled2, X_validate_scaled2, X_test_scaled2 = scaler, X_train_scaled, X_validate_scaled, X_test_scaled

def select_rfe(X, y, n=1, model=LinearRegression(normalize=True), rank=False):
    '''
    Takes in the X, y train and an optional n values and model to use with
    RFE to return n (default=1) best variables for predicting the
    target of y, optionally can be used to output ranks of features in
    predictions
    '''

    # assign RFE using LinearRegression and top two features as default
    selector = RFE(estimator=model, n_features_to_select=n)
    # fit selector to training set
    selector.fit(X, y)
    # assign and apply mask to DataFrame for column names
    mask = selector.get_support()
    top_n = X.columns[mask].to_list()
    # check if rank=True
    if rank == True:
        # print DataFrame of rankings
        print(pd.DataFrame(X.columns, selector.ranking_,
                           [f'n={n} RFE Rankings']).sort_index())
    return top_n

select_rfe(train_scaled2, y_train_df.target, n=7, rank=True) #top 7
# =============================================================================
# OUTPUT:
#    n=7 RFE Rankings
# 1            target
# 1        bath_count
# 1         bed_count
# 1          latitude
# 1         longitude
# 1            Orange
# 1                LA
# 2          tax_rate
# 3        tax_amount
# 4              area
# 5   structure_value
# 6        land_value
# 7    property_value
# 8         parcel_id
# 9           Ventura
# 10       year_built
# 11     census_tract
# Out[69]: ['target', 'bath_count', 'bed_count', 'latitude', 'longitude', 'LA', 'Orange']
# 
# =============================================================================



def select_kbest(X, y, k=1, score_func=f_regression):
    '''
    Takes in the X, y train and an optional k values and score_func to use
    SelectKBest to return k (default=1) best variables for predicting the
    target of y
    
    '''

    # assign SelectKBest using f_regression and top two features default
    selector = SelectKBest(score_func=score_func, k=k)
    # fit selector to training set
    selector.fit(X, y)
    # assign and apply mask to DataFrame for column names
    mask = selector.get_support()
    top_k = X.columns[mask].to_list()
    return top_k
# use KBest to find recommended non-cluster features
select_kbest(train_scaled2, y_train_df.target, k=7)

# =============================================================================
# OUTPUT: 
# ['target',
#  'bed_count',
#  'area',
#  'structure_value',
#  'property_value',
#  'land_value',
#  'tax_amount']
# =============================================================================
train_scaled.columns

clust_feats = ['lat_long_pv_clstr_1', 'lat_long_pv_clstr_2',
       'lat_long_pv_clstr_3', 'lat_long_pv_clstr_4', 'bed_taxrate_clstr_1',
       'bed_taxrate_clstr_2', 'bed_taxrate_clstr_3', 'bed_taxrate_clstr_4',
       'land_val_census_clstr_1', 'land_val_census_clstr_2',
       'land_val_census_clstr_3', 'land_val_census_clstr_4']
col_list = ['parcel_id', 'target']
train_scaled2 = drop_columns(train_scaled2, col_list)


# clust_feats = pd.DataFrame(clust_feats)

# all_features = pd.concat([train_scaled2, clust_feats], axis=1)

# use RFE to find recommended features including clusters
select_rfe(train_scaled2, y_train_df.target, n=7, rank=True)


# =============================================================================
# # output
# ['bath_count',
#  'bed_count',
#  'latitude',
#  'longitude',
#  'Orange',
#  'Ventura',
#  'tax_rate']
# =============================================================================
# use KBest to find recommended features including clusters
select_kbest(train_scaled2, y_train_df.target, k=7)



# =============================================================================
# OUPUT:
# =============================================================================
# ['bed_count',
#  'area',
#  'structure_value',
#  'property_value',
#  'land_value',
#  'tax_amount',
#  'tax_rate']




# =============================================================================
#                 MODELING AND EVALUATION
# =============================================================================
# use RFE to find top 9 recommended features for modeling
top_feats = select_rfe(train_scaled2, y_train_df.target, n=9)
print(f'\nThe top recommended features via RFE are:\n{top_feats}.\n')


# output:The top recommended features via RFE are:
# ['bath_count', 'bed_count', 'latitude', 'longitude', 'year_built', 'LA', 'Orange', 'Ventura', 'tax_rate'].




# use RFE to find top 9 recommended features for modeling
top_ks = select_kbest(train_scaled2, y_train_df.target, k=9)
print(f'\nThe top recommended features via SelectKBest are:\n{top_ks}.\n')


# The top recommended features via SelectKBest are:
# ['bath_count', 'bed_count', 'area', 'longitude', 'structure_value', 
#  'property_value', 'land_value', 'tax_amount', 'tax_rate'].




# =============================================================================
#                           Setting Baselines
# =============================================================================
# =============================================================================
# Two baselines will be created as additional comparison metrics. baseline_mean
#  will be created using the mean of log_error of the y_train, and baseline_median
#  will be created using the median log_error of the y_train. These baselines will
#  be used to measure performance against both out-of-sample evaluation on the
#  validate data set and the test data set.
# =============================================================================
# create variable holding mean of log_error and attach to y
baseline_mean = y_train_df.target.mean()
y_train_df['baseline_mean'] = baseline_mean
y_validate['baseline_mean'] = baseline_mean
y_test['baseline_mean'] = baseline_mean


# create variable holding mean of log_error and attach to y
baseline_median = y_train_df.target.median()
y_train_df['baseline_median'] = baseline_median
y_validate['baseline_median'] = baseline_median
y_test['baseline_median'] = baseline_median





y_val_df = pd.DataFrame(y_validate)
y_val_df['baseline_mean'] = baseline_mean
y_val_df['baseline_median'] = baseline_median


y_test_df = pd.DataFrame(y_test)
y_test_df['baseline_mean'] = baseline_mean
y_test_df['baseline_median'] = baseline_median
# =============================================================================
# Train
# ----------
# Create DataFrames to hold mean and median baselines for future comparison on train performance.
# =============================================================================

def get_metrics(true, predicted, display=False):
    '''
    Takes in the true and predicted values and returns the rmse and r^2 for the
    model performance
    '''
    
    rmse = mean_squared_error(true, predicted, squared=False)
    r2 = explained_variance_score(true, predicted)
    if display == True:
        print(f'Model RMSE: {rmse:.2g}')
        print(f'       R^2: {r2:.2g}')
    return rmse, r2


# obtain mean baseline performance for model comparison on in-sample
rmse, r2 = get_metrics(y_train_df.target, y_train_df.baseline_mean)
model_dict = {'model':'baseline_mean', 'RMSE':rmse, 'R^2':r2}
insamp_df = pd.DataFrame([model_dict])



# obtain median baseline performance for model comparison on in-sample
rmse, r2 = get_metrics(y_train_df.target, y_train_df.baseline_median)
model_dict = {'model':'baseline_median', 'RMSE':rmse, 'R^2':r2}
insamp_df = insamp_df.append([model_dict], ignore_index=True)




# =============================================================================
# Validate
# 
# Create DataFrames to hold mean and median baselines for future comparison on validate performance.
# =============================================================================
# obtain mean baseline performance for model comparison on out-of-sample



rmse, r2 = get_metrics(y_val_df.target, y_val_df.baseline_mean)
model_dict = {'model':'baseline_mean', 'RMSE':rmse, 'R^2':r2}
outsamp_df = pd.DataFrame([model_dict])

# obtain median baseline performance for model comparison on out-of-sample
rmse, r2 = get_metrics(y_val_df.target, y_val_df.baseline_median)
model_dict = {'model':'baseline_median', 'RMSE':rmse, 'R^2':r2}
outsamp_df = outsamp_df.append([model_dict], ignore_index=True)




# =============================================================================
# Test
# 
# Create DataFrames to hold mean and median baselines for future comparison on test performance.
# =============================================================================

# obtain mean baseline performance for model comparison on test
rmse, r2 = get_metrics(y_test_df.target, y_test_df.baseline_mean)
model_dict = {'model':'baseline_mean', 'RMSE':rmse, 'R^2':r2}
test_df = pd.DataFrame([model_dict])


# obtain median baseline performance for model comparison on test
rmse, r2 = get_metrics(y_test_df.target, y_test_df.baseline_median)
model_dict = {'model':'baseline_median', 'RMSE':rmse, 'R^2':r2}
test_df = y_test_df.append([model_dict], ignore_index=True)


# =============================================================================
# Ordinary Least Squares
# =============================================================================
top_feats2 = pd.DataFrame(top_feats)
def train_model(X, y, model, model_name):
    '''
    Takes in the X_train and y_train, model object and model name, fit the
    model and returns predictions and a dictionary containg the model RMSE
    and R^2 scores on train
    '''

    # fit model to X_train_scaled
    model.fit(X, y)
    # predict X_train
    predictions = model.predict(X)
    # get rmse and r^2 for model predictions on X
    rmse, r2 = get_metrics(y, predictions)
    performance_dict = {'model':model_name, 'RMSE':rmse, 'R^2':r2}
    
    return predictions, performance_dict


# using top five according to RFE on train

# create model object
lm1 = LinearRegression(normalize=True)
# run train_model function to fit model to train and obtain performance metrics
y_train_df['lm1_predictions'], model_dict = train_model(X_train_scaled2[top_feats], y_train_df.target, lm1, 'lm1')
# append performance metrics to dataframe
insamp_df = insamp_df.append([model_dict], ignore_index=True)



# show current model performance on train
pd.DataFrame([model_dict])
# =============================================================================
#   model      RMSE       R^2
# 0   lm1  0.154998  0.001164
# =============================================================================
def model_testing(X, y, model, model_name):
    '''
    Takes in the X and y for validate or test, model object and model name and
    returns predictions and a dictionary containg the model RMSE and R^2 scores
    on validate or test
    '''
    
    # obtain predictions on X
    predictions = model.predict(X)
    # get for performance and assign them to dictionary
    rmse, r2 = get_metrics(y, predictions)
    performance_dict = {'model':model_name, 'RMSE':rmse, 'R^2':r2}
    
    return predictions, performance_dict


# run model_testing function to evaluate performance using out-of-sample data set
y_val_df['lm1_predictions'], model_dict = model_testing(X_validate_scaled2[top_feats2], y_val_df.target, lm1, 'lm1')
# append performance metrics to dataframe
outsamp_df = outsamp_df.append([model_dict], ignore_index=True)