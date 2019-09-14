#Delsey Sbau
#Lesson 4 assignment 

#data set descritpion:
#Using the hepatitis data set, which has 155 instances
# and 20 attributes
# six of those attributes are numerical 
#age,b_rbn,al_pho,sgot,a_bmn
#fourteen others are binary categorical with 1,2 values
#there are atleast one "?" in 15 attributes

#removing columns:
#protime column had 67 missing values, so i removed the whole column  because
#that's a little above 40% missing values. Left all other columns since they were not as bad

#no rows were removed (since it's a small data set and 
#if i removed rows with missing data, then i would not be able to impute with median etc. )

#replace missing values with median:
#attributes that had missing values replaced;
#"steroid", "fatigue", "malaise", "anorexia", "liver_big", "liver_firm", "spleen_palp", "spiders", "ascites", "varices", "b_rbn","al_pho","sgot","a_bmn",

#outliers:
#only removed outliers from AGE,BILIRUBIN,ALK PHOSPHATE,SGOT,ALBUMIN
#values were within limits if they're between 2 standard devaotions off the mean
#no reason to remove from the rest, since they were vakues that were 1 or 2

#histograms: 
#age,b_rbn,al_pho,sgot,a_bmn
#these are numerical columns and there is histograms before removing outliers
#and after removing outliers

#main program that calls all funcitons at end


#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

###############################################################################
#function acquire, which downloads data set and puts in column names
#no argument, returns HPT, which is the data set going to be used for cleaning
def acquire():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data"
    #HPT is the shortened name of Hepatitis data set
    HPT = pd.read_csv(url, header=None)
    #put in column names
    HPT.columns = ["class", "age", "sex", "steroid", "antiviral", "fatigue", "malaise", "anorexia", "liver_big", "liver_firm", "spleen_palp", "spiders", "ascites", "varices", "b_rbn","al_pho","sgot","a_bmn","protime","histology"]
    HPT.dtypes
    return (HPT)
#end acquire funciton 
###############################################################################
###############################################################################
###############################################################################
###############################################################################
        
#function data_remove_cols removes columns with high number of mssing data
#from reading teh data set description, I can see that attribute protime has 67 
#missing vaues denoted by "?"
#argument is HPT, returns the fewer columns HPT
def data_remove_cols(HPT):

    HPT.head()
    #missing values are denoted by "?", so we replace with nan
    HPT = HPT.replace(to_replace="?", value=float("NaN"))
    #print each column sum of nans
    print("\n sum of missing attribute values \n")
    print(HPT.isnull().sum())
    #creating a flag for columns that have less than 67 missing values
    FlagGood = HPT.isnull().sum() < 67
    #remove columns that have more than 67 missing values (which is protime)
    HPT = HPT.loc[:,FlagGood]
    HPT.shape
    return (HPT)
#end data_remove_cols funciton 
###############################################################################
###############################################################################
###############################################################################
###############################################################################
        

#function data_clean_num cleans columns that are numerical
#cleaning includes replacing outliers with mean and putting in median fo rmissing values
#one argument HPT, returns HPT
def data_clean_num (HPT):
    
    #b_rbn:bilirubin: continuous variable
    HPT.loc[:,"b_rbn"].unique()
    #found nan, coerce into numeric
    HPT.loc[:, "b_rbn"] = pd.to_numeric(HPT.loc[:, "b_rbn"], errors='coerce')
    #find values that are nan
    HasNan = np.isnan(HPT.loc[:,"b_rbn"])
    # The replacement value for NaNs is Median
    Median = np.nanmedian(HPT.loc[:,"b_rbn"])
    #replace nans with Median
    HPT.loc[HasNan, "b_rbn"] = Median
    #print name of attribute, display histogram
    print("\n")
    print("b_rbn")
    plt.hist(HPT.loc[:, "b_rbn"])
    plt.show()
    
    #outlier: visually noticed outlier around 8, need to check
    #High Limit of b_rbn is the mean plus 2 standard deviations
    brn_HI = np.mean(HPT.loc[:, "b_rbn"]) + 2*np.std(HPT.loc[:, "b_rbn"])
    #Low Limit of b_rbn is the mean minus 2 standard deviations
    brn_LO = np.mean(HPT.loc[:, "b_rbn"]) - 2*np.std(HPT.loc[:, "b_rbn"])
    
    #create flag that is within limits
    FlagGood = (HPT.loc[:, "b_rbn"]>= brn_LO) & (HPT.loc[:, "b_rbn"] <= brn_HI)
    #create flag that is not within limits
    FlagBad = ~FlagGood
    
    #replace the elements not in limit with mean of values in limit
    HPT.loc[FlagBad, "b_rbn"]= np.mean(HPT.loc[FlagGood, "b_rbn"])   
    #histogram
    plt.hist(HPT.loc[:, "b_rbn"])
    plt.show()
    #standard deviation
    print(np.std(HPT.loc[:, "b_rbn"]))
    
    ##########################################################################    
    ##########################################################################
    
    #al_pho: ALK PHOSPHATE 
    HPT.loc[:,"al_pho"].unique()
    #found nan, coerce into numeric
    HPT.loc[:, "al_pho"] = pd.to_numeric(HPT.loc[:, "al_pho"], errors='coerce')
    HPT.loc[:,"al_pho"].dtypes
    #find values that are nan
    HasNan = np.isnan(HPT.loc[:,"al_pho"])
    # The replacement value for NaNs is Median
    Median = np.nanmedian(HPT.loc[:,"al_pho"])
    # Median imputation of nans
    HPT.loc[HasNan, "al_pho"] = Median
    print("\n")
    
    print("al_pho")
    plt.hist(HPT.loc[:, "al_pho"])
    plt.show()

    #outlier, usual values need to be between 2 standard deviations from mean
    brn_HI = np.mean(HPT.loc[:, "al_pho"]) + 2*np.std(HPT.loc[:, "al_pho"])
    brn_LO = np.mean(HPT.loc[:, "al_pho"]) - 2*np.std(HPT.loc[:, "al_pho"])

    #creeate flag that is within limits    
    FlagGood = (HPT.loc[:, "al_pho"]>= brn_LO) & (HPT.loc[:, "al_pho"] <= brn_HI)
    #create flag that is not within limits
    FlagBad = ~FlagGood
    #replace the elements not in limit with mean of values in limit
    HPT.loc[FlagBad, "al_pho"]= np.mean(HPT.loc[FlagGood, "al_pho"])   
    plt.hist(HPT.loc[:, "al_pho"])
    plt.show()
    print(np.std(HPT.loc[:, "al_pho"]))
    ##########################################################################
    ##########################################################################

    #sgot: SGOT 
    HPT.loc[:,"sgot"].unique()
    #found nan, coerce into numeric
    HPT.loc[:, "sgot"] = pd.to_numeric(HPT.loc[:, "sgot"], errors='coerce')
    HPT.loc[:,"sgot"].dtypes
    
    #repalce nan with median 
    #find values that are nan
    HasNan = np.isnan(HPT.loc[:,"sgot"])
    # The replacement value for NaNs is Median
    Median = np.nanmedian(HPT.loc[:,"sgot"])
    # Median imputation of nans
    HPT.loc[HasNan, "sgot"] = Median
    print("\n")
    
    #print name of attribute, display histogram
    print("sgot")
    plt.hist(HPT.loc[:, "sgot"])    
    plt.show()
    
    #outlier
    brn_HI = np.mean(HPT.loc[:, "sgot"]) + 2*np.std(HPT.loc[:, "sgot"])
    brn_LO = np.mean(HPT.loc[:, "sgot"]) - 2*np.std(HPT.loc[:, "sgot"])
    FlagGood = (HPT.loc[:, "sgot"]>= brn_LO) & (HPT.loc[:, "sgot"] <= brn_HI)
     #create flag that is not within limits
    FlagBad = ~FlagGood
    #replace the elements not in limit with mean of values in limit
    HPT.loc[FlagBad, "sgot"]= np.mean(HPT.loc[FlagGood, "sgot"])   
    plt.hist(HPT.loc[:, "sgot"])
    plt.show()
    print(np.std(HPT.loc[:, "sgot"]))
    ##########################################################################
    ##########################################################################

    #a_bmn: ALBUMIN
    HPT.loc[:,"a_bmn"].unique()
    #found nan, coerce into numeric 
    HPT.loc[:, "a_bmn"] = pd.to_numeric(HPT.loc[:, "a_bmn"], errors='coerce')
    HPT.loc[:,"a_bmn"].dtypes
    #find values that are nan
    HasNan = np.isnan(HPT.loc[:,"a_bmn"])
    # The replacement value for NaNs is Median
    Median = np.nanmedian(HPT.loc[:,"a_bmn"])
    # Median imputation of nans
    HPT.loc[HasNan, "a_bmn"] = Median
    print("\n")

    #print name of attribute, display histogram
    print("a_bmn")
    plt.hist(HPT.loc[:, "a_bmn"])
    plt.show()
    
    #outlier, values need to be within 2 stds from mean
    brn_HI = np.mean(HPT.loc[:, "a_bmn"]) + 2*np.std(HPT.loc[:, "a_bmn"])
    brn_LO = np.mean(HPT.loc[:, "a_bmn"]) - 2*np.std(HPT.loc[:, "a_bmn"])  
    #create flag that are within limits
    FlagGood = (HPT.loc[:, "a_bmn"]>= brn_LO) & (HPT.loc[:, "a_bmn"] <= brn_HI)
    #create flag that is not within limits
    FlagBad = ~FlagGood   
    #replace the elements not in limit with mean of values in limit
    HPT.loc[FlagBad, "a_bmn"]= np.mean(HPT.loc[FlagGood, "a_bmn"])   
    plt.hist(HPT.loc[:, "a_bmn"])
    plt.show()
    print(np.std(HPT.loc[:, "a_bmn"]))
    
    return (HPT)
#end data_clean_num function 
###############################################################################
###############################################################################
###############################################################################
###############################################################################
#function to clean categorical columns 
#one argument HPT, returns HPT
#i will be reaplcing all missing values in these columns with medians (atleast for now)
##so will coerce objeccts into numeric also
def data_clean_cat (HPT):

    #steroid
    HPT.loc[:,"steroid"].unique()
    HPT.loc[:, "steroid"] = pd.to_numeric(HPT.loc[:, "steroid"], errors='coerce')
    HasNan = np.isnan(HPT.loc[:,"steroid"])
    # The replacement value for NaNs is Median
    Median = np.nanmedian(HPT.loc[:,"steroid"])
    # Median imputation of nans
    HPT.loc[HasNan, "steroid"] = Median

    #no outlier since values are supposed to be 1,2
    ##########################################################################    
    ########################################################################## 
    
    #antivirals
    HPT.loc[:,"antiviral"].unique()
    HPT.loc[:, "antiviral"] = pd.to_numeric(HPT.loc[:, "antiviral"], errors='coerce')
    HasNan = np.isnan(HPT.loc[:,"antiviral"])
    # The replacement value for NaNs is Median
    Median = np.nanmedian(HPT.loc[:,"antiviral"])
    # Median imputation of nans
    HPT.loc[HasNan, "antiviral"] = Median
    #no outlier since values are supposed to be 1,2

    ########################################################################## 
   
    #fatigue
    HPT.loc[:,"fatigue"].unique()
    HPT.loc[:, "fatigue"] = pd.to_numeric(HPT.loc[:, "fatigue"], errors='coerce')
    HasNan = np.isnan(HPT.loc[:,"fatigue"])
    # The replacement value for NaNs is Median
    Median = np.nanmedian(HPT.loc[:,"fatigue"])
    # Median imputation of nans
    HPT.loc[HasNan, "fatigue"] = Median

    ########################################################################## 
    ########################################################################## 
   
    #malaise
    HPT.loc[:,"malaise"].unique()
    HPT.loc[:, "malaise"] = pd.to_numeric(HPT.loc[:, "malaise"], errors='coerce')
    HasNan = np.isnan(HPT.loc[:,"malaise"])
    # The replacement value for NaNs is Median
    Median = np.nanmedian(HPT.loc[:,"malaise"])
    # Median imputation of nans
    HPT.loc[HasNan, "malaise"] = Median

    ########################################################################## 

    ########################################################################## 
   
    #anorexia
    HPT.loc[:,"anorexia"].unique()
    HPT.loc[:, "anorexia"] = pd.to_numeric(HPT.loc[:, "anorexia"], errors='coerce')
    HasNan = np.isnan(HPT.loc[:,"anorexia"])
    # The replacement value for NaNs is Median
    Median = np.nanmedian(HPT.loc[:,"anorexia"])
    # Median imputation of nans
    HPT.loc[HasNan, "anorexia"] = Median

    ##########################################################################
    ########################################################################## 
   
    #liver_big
    HPT.loc[:,"liver_big"].unique()
    HPT.loc[:, "liver_big"] = pd.to_numeric(HPT.loc[:, "liver_big"], errors='coerce')
    HasNan = np.isnan(HPT.loc[:,"liver_big"])
    # The replacement value for NaNs is Median
    Median = np.nanmedian(HPT.loc[:,"liver_big"])
    # Median imputation of nans
    HPT.loc[HasNan, "liver_big"] = Median

    ##########################################################################
    ########################################################################## 
   
    #liver_firm
    HPT.loc[:,"liver_firm"].unique()
    HPT.loc[:, "liver_firm"] = pd.to_numeric(HPT.loc[:, "liver_firm"], errors='coerce')
    HasNan = np.isnan(HPT.loc[:,"liver_firm"])
    # The replacement value for NaNs is Median
    Median = np.nanmedian(HPT.loc[:,"liver_firm"])
    # Median imputation of nans
    HPT.loc[HasNan, "liver_firm"] = Median

    ##########################################################################
    
    ########################################################################## 
   
    #spleen_palp
    HPT.loc[:,"spleen_palp"].unique()
    HPT.loc[:, "spleen_palp"] = pd.to_numeric(HPT.loc[:, "spleen_palp"], errors='coerce')
    HasNan = np.isnan(HPT.loc[:,"spleen_palp"])
    # The replacement value for NaNs is Median
    Median = np.nanmedian(HPT.loc[:,"spleen_palp"])
    # Median imputation of nans
    HPT.loc[HasNan, "spleen_palp"] = Median

    ##########################################################################
    ########################################################################## 
   
    #spiders
    HPT.loc[:,"spiders"].unique()
    HPT.loc[:, "spiders"] = pd.to_numeric(HPT.loc[:, "spiders"], errors='coerce')
    HasNan = np.isnan(HPT.loc[:,"spiders"])
    # The replacement value for NaNs is Median
    Median = np.nanmedian(HPT.loc[:,"spiders"])
    # Median imputation of nans
    HPT.loc[HasNan, "spiders"] = Median

    ##########################################################################
    ########################################################################## 
   
    #ascites
    HPT.loc[:,"ascites"].unique()
    HPT.loc[:, "ascites"] = pd.to_numeric(HPT.loc[:, "ascites"], errors='coerce')
    HasNan = np.isnan(HPT.loc[:,"ascites"])
    # The replacement value for NaNs is Median
    Median = np.nanmedian(HPT.loc[:,"ascites"])
    # Median imputation of nans
    HPT.loc[HasNan, "ascites"] = Median

    ##########################################################################
    ########################################################################## 
   
    #varices
    HPT.loc[:,"varices"].unique()
    HPT.loc[:, "varices"] = pd.to_numeric(HPT.loc[:, "varices"], errors='coerce')
    HasNan = np.isnan(HPT.loc[:,"varices"])
    # The replacement value for NaNs is Median
    Median = np.nanmedian(HPT.loc[:,"varices"])
    # Median imputation of nans
    HPT.loc[HasNan, "varices"] = Median

    return (HPT)
#end data_clean_cat fucntion 
###############################################################################
###############################################################################
###############################################################################
###############################################################################
    
#main function calls all other functions and displays scatter matrix
def main():
    
    #function calls
    HPT = acquire()
    HPT_fewer_cols = data_remove_cols(HPT)
    HPT_cln_fewer_cols = data_clean_num(HPT_fewer_cols)
    HPT_cln_fewer_cols = data_clean_cat(HPT_cln_fewer_cols)
    #display scatter matrix
    #class is the attribute definin gdead or not 
    scatter_matrix(HPT_cln_fewer_cols, c=HPT_cln_fewer_cols.loc[:,"class"], figsize=[20,20], s=1000)
    plt.show()
    return (0)
#end main function
    
#call main
main()