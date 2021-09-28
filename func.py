#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 11:36:25 2021

@author: carolyndavis
"""
#Imports Utilized
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



#Dropping id columns, established as unnecessary
#Dropping type_id columns, established as unnecessary
#Dropping all region id columns, using fips for possible region features
#Droppung columns with information, provided in more contextual columns 



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



def create_features(zillow_df):
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
    
    return zillow_df




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


       