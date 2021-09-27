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




