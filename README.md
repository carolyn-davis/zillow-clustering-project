## ZILLOW CLUSTERING PROJECT:
---------------------------------
By: Carolyn Davis, Germain Cohort
9/28/2021

------------------------------------------------------------------------------------------------------------------------------


### Project Summary
What is driving the errors in the Zestimates? For this project I worked with the Zillow Dataset provided by 
Codeup SQL database. The tables utilized within the database were properties_2017 and predictions_2017. In this 
project clustering methodologies are utilized in stages of exploration. This is in aim of acheiving meaningful insight from 
methods of clustering.


------------------------------------------------------------------------------------------------------------------------------

### GOALS OF THIS PROJECT:
- Utilized cluster methodologies to identify key drivers of logerror (the target)
- Create modules that retrieve and prepare the data with use of various functions 
- Provide a step by step walkthrough, documnenting thoroughly for project recreation
- Devlop and produce four various models, the goal is to beat the baseline 
- Ensure steps are met to ensure reproduction of project 
------------------------------------------------------------------------------------------------------------------------------

### DELIVERABLES:
- Final report notebook detailing all finding and pertinent methodologies
- Modules providing functions and comments for replication of each stage of the pipeline
- A README.md file detailing the project's specs, project planning, key finds, recreation steps, and conclusions

------------------------------------------------------------------------------------------------------------------------------


### PROJECT PLANNING:
- Viewed the data and composed SQL query to pull in data from the Codeup database
- Ran through every stage of the project and allot time for each phase ahead of time. 
- Start with a minimum of three features with the goal of testing these features on four models.
- Some Questions asked during this phase:
    1.) Are there groupings where features are more likely to drive log error?
    2.) How can we improve Zestimate?
    3.) What key drivers are related to the difference between our Zestimate and the actual sales prices?

- Acquire Stage Planning:
    -- Goal: Leave this stage with acquire.py file and functions to easily collect data in future.
        Summary:
        -Data is collected from the codeup cloud database with an appropriate SQL query.

        -Data is only acquired from the properties and predictions table 2017
    -- Acquire Stage Checklist:
        1.) -Pull in data from predictions and properties 2017 SQL
        2.) -Gander at the data and entertain 10 features
        3.) -Create a function for future acquirement of the Zillow data for future use
        4.) -In function have data generate with renamed columns for easy prep

- Prepare Stage Planning:
    -- Goal:  Leave this stage a wrangle.py file saved to git repo
                Pair along with steps for other users to recreate 
    Summary:
    -Column data types are appropriate for the data they contain
    -Missing values are investigated and handled
    -Outliers are investigated and handled

    -- Prepare Stage Checklist:
        1.) change values columns
        2.) investigate missing values
        3.) investigate any possible outliers
        4.) If outliers, should they be included/excluded?
        5.) Rename the columns for readibility
        6.) Unique value counts for future selection
- Explore Stage Planning:
    -- Goal: Identify the key driver of the target(logerror) by clustering accounts based on their association with feature picked 
        -Additionally define and label those patterns discovered, How? K-Means, PearsonR, Heatmap 
        Summary:
        - Exploration: interaction between independent variables and the target variable is explored using visualization and statistical testingClustering is used to explore the data. A conclusion, supported by statistical testing and visualization, is drawn on whether or not the clusters are helpful/useful. At least 3 combinations of features for clustering should be tried.

        Explore Stage Checklist:
        1.) Initial Hypothesises documented, Is there a relationship between these three features and the target, logerror?
        2.) **Plotting/Visualizations:** Pairplot/ Heatmap
        3.) **Statistical Testing :**  PearsonR/Explore Univariate
        4.) **Visual:** Explore Univariate/Plotunivariate
        5.) Establish **Observations/Takeaways** after each visualization and statistical test
        6.) Make Clusters **only train** on features

        Explore Clusters Stage Checklist:
        1.) **Initial Hypothesises** for the three features explored
        2.) Make Clusters **only train** on features with KMeans
        3.) Set columns for cluster exploration
        4.) Observe clusters alone with target
        5.) Observe clusters in comparison of features with one another
        6.) Observe clusters for all features (the big three)
        7.) Perform Statistical Testing on Clusters: T_test/ChiSq?
        8.) Validate and Test Chosen Clusters
        9.) Predict Logerror Clusters
        10.) Select KBest: Feature Engineering
        11.) Drop all non-numeric columns for modeling
- Feature Engineering Stage Planning:
    -- Goal/ Summary: 
        -Initialize the machine learning algorithm, in this case LinearRegression
            -Initialize the RFE object, and provide the ML algorithm object from step 1
            -Fit the RFE object to our data. Doing this will provide us a list of features (the number we asked for) as well as a ranking of all the features.
            -Assign the list of selected features to a variable.
            -Optional: Get ranking of all variables (1 being most important
        
        Feature Engineering Stage Checklist:
        1.) Perform Recursive Feature Elimination (RFE)
        2.) Establish Best Features from RFE
        3.) Optional perform K-Best


-Modeling Stage Planning:
    -- Goal/Summary:
    **Modeling:** At least 4 different models are created and their performance is compared. One model is the distinct combination of algorithm, hyperparameters, and features.
    -Regression models are the best fit for this project
    -Supervised models : OLS, LASSO + Lars, Polynomial Regressor

    Modeling Stage Checklist:
        1.) Establish the Scaled/Unscaled** data for modeling
        2.) Set** features
        3.) *Establish** the baseline
        4.) **Run** models on Train
        5.) **-Run** models on Validate
        6.) **-Run** models on Test
        7.) **Observe** Results in comparison to the baseline

- Deliver Stage Planning:
    -- Goal/Summary:
        **You are expected to deliver a github repository with the following contents:**
        -A clearly named **final notebook**. This notebook will be what you present and should contain plenty of markdown documentation and cleaned up code.
        -A **README** that explains what the project is, how to reproduce you work, and your notes from project planning.
        -A **Python module** or modules that automate the data acquisition and preparation process. These modules should be imported and used in your final notebook.
    
    Deliver Stage Checklist:
        1.) **github repo** called zillow-clustering-project
        2.) a final jupyter notebook for presentation
        3.) ReadMe.md all about the project and how to recreate it
        4.) acquire.py file for retrieving Zillow data
        5.) wrangle.py file that retrieves your prepared/cleaned data
        6.) data dictionary

------------------------------------------------------------------------------------------------------------------------------
                             |   DATA DICTIONARY |
------------------------------------------------------------------------------------------------------------------------------
    Variable	        Definition	                                                                Data Type
------------------------------------------------------------------------------------------------------------------------------
**bath_count**	: count of full- and half-bathrooms	                                                float64
------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------
**bed_count**	count of bedrooms	                                                                 int64
------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------
**census_tract**	: US census tract codes for non-precise location	                            float64
------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------
**LA** :	boolean for if county is within Los Angeles County	                                    int64
------------------------------------------------------------------------------------------------------------------------------
**land_value**	: value of land in U.S. dollars	                                                    float64
------------------------------------------------------------------------------------------------------------------------------
**lat_long_pv_clstr**	: boolean for five clusters of latitude, longitude, and property_value	    uint8
------------------------------------------------------------------------------------------------------------------------------
**latitude**	: latitudinal geographic coordinate of property	                                    float64
------------------------------------------------------------------------------------------------------------------------------
**log_error/target** 	: difference of log(Zestimate) and log(SalePrice)	                            float64
------------------------------------------------------------------------------------------------------------------------------
**longitude**	: longitudinal geographic coordinate of property	                                    float64
------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------
**lot_square_feet**	: size of lot(land) in square feet	                                            float64
------------------------------------------------------------------------------------------------------------------------------
**orange**	: boolean for if county is within Orange County	                                        int64
------------------------------------------------------------------------------------------------------------------------------
**parcel_id**	: unique identifier of property	                                                    int64
------------------------------------------------------------------------------------------------------------------------------
**property_id** :	unique identifier of property	                                                int64
------------------------------------------------------------------------------------------------------------------------------
**property_value**	: value of property in entirety in U.S. dollars	                                float64
------------------------------------------------------------------------------------------------------------------------------
**room_count**	: count of bedrooms and full- and half-bathrooms	                                float64
------------------------------------------------------------------------------------------------------------------------------
**square_footage**	dimensions of property in square feet	                                        float64
------------------------------------------------------------------------------------------------------------------------------
**structure_value** :	value of structure on property in U.S. dollars	                            float64
------------------------------------------------------------------------------------------------------------------------------
**tax_amount** : 	most recent tax payment from property owner	                                        float64
------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------
**transaction_date** : most recent date of property sale	                                        datetime64
------------------------------------------------------------------------------------------------------------------------------
**year_built**	: year structure was originally built	int64
------------------------------------------------------------------------------------------------------------------------------



------------------------------------------------------------------------------------------------------------------------------
 **PROJECT REPRODUCTION**:
  To recreate and reproduce results of this project, you will need to create a module named env.py. This file will need to contain login credentials for the Codeup database server stored in their respective variables named host, username, and password. You will also need to create the following function within. This is used in all functions that acquire data from the SQL server to create the URL for connecting. db_name needs to be passed as a string that matches exactly with the name of a database on the server.
-------------------------------------------------------------------------
    def get_connection(db_name):
        return f'mysql+pymysql://{username}:{password}@{host}/{db_name}'
-------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------------------\
**MODULES** 
- acquire.py : contains functions used in initial data acquisition leading into the prepare phase
- prepare_data.py : contains notes and scratchpad that was used in initial data acquisition leading into the prepare phase
- explore_.py :  contains functions to visualize the prepared data and estimate the best drivers of property value
- func.py : contains all functions to prepare and explore data in the manner needed for specific project needs