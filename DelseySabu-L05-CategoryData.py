#Delsey Sabu
#DelseySabu-L05-CategoryData.py

#working on the adult dataset from UCI repository
#z-normelization is done on all numerical columns, since it does nto get affected by outliers
#equal width binning is done on the "age" column with new column named "binnedage"
#decoding is done on the "payprediction" column, predicting if the individual-
#would make above or below $50,000
#imputation is done as needed with the most value of the column; "workclass","occupation","nativecon"
#consolidation is done on "edu" column
#"edunum" is binned into "binned_edu_num" and broken down into 6 columns
# ("elementary", "some-HS", "HS-grad", "some-college","bachelors", "graduate") for one hot encoding
#later, "binned_edu_num" is removed (obsolete column)
#plots are shown for "edu" before and after consolidation to show the affects of consolidation
#plot is shown for "binned_edu_num" for ease for comparing with the consolidated "edu" column

#3 functions; acquire(), cln_cat(adult),cln_num(adult)
#function calls at end 


#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###############################################################################
#function acquire, which downloads data set and puts in column names
#no argument, returns adult, which is the data set going to be used for cleaning
def acquire():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    #HPT is the shortened name of Hepatitis data set
    adult = pd.read_csv(url, header=None)
    adult.head()
    #put in column names
    adult.columns = ["age", "workclass", "fnlwgt", "edu", "edunum", "marital", "occupation", "relationship", "race", "sex", "capgain", "caploss", "hrswk", "nativecon", "payprediction"]
    adult.dtypes
    adult.shape
    return (adult)
#end acquire funciton 
###############################################################################
###############################################################################
###############################################################################
###############################################################################
#function cln_cat cleans categorical columns by imputing missing values, consolidation and decoding
#takees in the argumnet adult and returns adult
def cln_cat (adult):
   
    #workclass has missing values
    adult.loc[:,"workclass"].value_counts()
    # Impute missing values, denoted by " ?" notice the space
    #there are missing values,I am just going to impute the missing ? with the most value " Private"
    adult.loc[adult.loc[:, "workclass"] == " ?", "workclass"] = " Private"
    
    
    #education : edu
    #edu does not have missing values
    #plotting teh edu column before consolidation with labels
    adult.loc[:,"edu"].value_counts().plot(kind='bar')
    plt.title("Education before consolidation")
    plt.ylabel("Number of people")
    plt.xlabel("Education level")
    plt.show()
    
    # consolidate
    #grouping te grade levels in HS and elementary together
    adult.loc[adult.loc[:, "edu"] == " 12th", "edu"] = " HS-grad"
    adult.loc[adult.loc[:, "edu"] == " 11th", "edu"] = " Some-HS"
    adult.loc[adult.loc[:, "edu"] == " 10th", "edu"] = " Some-HS"
    adult.loc[adult.loc[:, "edu"] == " 9th", "edu"] = " Some-HS"
    adult.loc[adult.loc[:, "edu"] == " 5th-6th", "edu"] = " Some-elementary"
    adult.loc[adult.loc[:, "edu"] == " 1st-4th", "edu"] = " Some-elementary"
    
    #plotting the edu column after consolidation with labels
    #we can see that it is cleaner to look at
    adult.loc[:,"edu"].value_counts().plot(kind='bar')
    plt.title("Education after consolidation")
    plt.ylabel("Number of people")
    plt.xlabel("Education level")
    plt.show()
    
    
    #occupation 
    adult.loc[:,"occupation"].value_counts()
    #there are missing values, we can cross reference to check if the work class was "Never-worked" (note for later)
    #but here, I am just going to impute the missing ? with the most value " Prof-speciality" 
    adult.loc[adult.loc[:, "occupation"] == " ?","occupation" ] = " Prof-speciality"


    #native country
    adult.loc[:,"nativecon"].value_counts()
    #there are missing values, impute the missing ? with the most value "United-States" 
    adult.loc[adult.loc[:, "nativecon"] == " ?","nativecon" ] = " United-States"

    #pay prediction
    #we need to decode it so that <=50K would be "below" and >50K would be "above"
    adult.loc[:,"payprediction"].value_counts()
    adult.loc[adult.loc[:, "payprediction"] == " <=50K","payprediction" ] = " below"
    adult.loc[adult.loc[:, "payprediction"] == " >50K","payprediction" ] = " above"    

    return (adult)
#end cln_cat funciton 
###############################################################################
###############################################################################
###############################################################################
###############################################################################
    
#function cln_num cleans columns that are numerical
#z-normelization, binning, one hot encoding
#one argument adult, returns adult
def cln_num (adult):
    
    #age
    
    # Equal-width Binning would divide the ages better between bins seeing the spread
    plt.hist(adult.loc[:,"age"])
    # Determine the boundaries of the bins
    NumberOfBins = 4
    BinWidth = (max(adult.loc[:,"age"]) - min(adult.loc[:,"age"]))/NumberOfBins
    MinBin1 = float('-inf')
    MaxBin1 = min(adult.loc[:,"age"]) + 1 * BinWidth
    MaxBin2 = min(adult.loc[:,"age"]) + 2 * BinWidth
    MaxBin3 = min(adult.loc[:,"age"]) + 3 * BinWidth
    MaxBin4 = float('inf')
    

    # Create the categorical variable
    # Start with an empty array that is the same size as age
    adult.loc[:,"binnedage"]= np.empty(len(adult.loc[:,"age"]), object) 
    
    
    # The conditions at the boundaries should consider the difference 
    # between less than (<) and less than or equal (<=) 
    # and greater than (>) and greater than or equal (>=)
    adult.loc[(adult.loc[:,"age"] > MinBin1) & (adult.loc[:,"age"] <= MaxBin1),"binnedage"] = "Below 35"
    adult.loc[(adult.loc[:,"age"] > MaxBin1) & (adult.loc[:,"age"] <= MaxBin2),"binnedage"] = "35-53"
    adult.loc[(adult.loc[:,"age"] > MaxBin2) & (adult.loc[:,"age"] <= MaxBin3),"binnedage"] = "54-71"
    adult.loc[(adult.loc[:,"age"] > MaxBin3) & (adult.loc[:,"age"] <= MaxBin4),"binnedage"] = "above 71"
    adult.loc[:,"binnedage"].unique()
    
    #plot binned age with labels 
    adult.loc[:,"binnedage"].value_counts().plot(kind='bar')
    plt.title("Age: Binned with Equal Width Bins")
    plt.ylabel("Number of people")
    plt.xlabel("Age categories")
    plt.show()

    #z-normalize age
    adult.loc[:,"age"] = ((adult.loc[:,"age"]) - np.mean(adult.loc[:,"age"]))/np.std(adult.loc[:,"age"])
    ##############
    
    #z-normalize fnlwght
    adult.loc[:,"fnlwgt"] = ((adult.loc[:,"fnlwgt"]) - np.mean(adult.loc[:,"fnlwgt"]))/np.std(adult.loc[:,"fnlwgt"])
    ##############
       
    #education num: edunum
    
    #bin edunum, then encode binned_edu_num
    # Determine the boundaries of the bins
    #this is a custom bin with boundaries dictated by cross referecing the edu category
    MinBin1 = float('-inf')
    MaxBin1 = 4
    MaxBin2 = 7
    MaxBin3 = 9
    MaxBin4 = 12
    MaxBin5 = 13
    MaxBin6 = float('inf')
    
 
    # Create the categorical variable "binned_edu_num"
    # Start with an empty array that is the same size as edunum
    adult.loc[:,"binned_edu_num"]= np.empty(len(adult.loc[:,"edunum"]), object)   
    
    # The conditions at the boundaries should consider the difference 
    # between less than (<) and less than or equal (<=) 
    # and greater than (>) and greater than or equal (>=)
    adult.loc[(adult.loc[:,"edunum"] > MinBin1) & (adult.loc[:,"edunum"] <= MaxBin1),"binned_edu_num"] = "elementary"
    adult.loc[(adult.loc[:,"edunum"] > MaxBin1) & (adult.loc[:,"edunum"] <= MaxBin2),"binned_edu_num"] = "some-HS"
    adult.loc[(adult.loc[:,"edunum"] > MaxBin2) & (adult.loc[:,"edunum"] <= MaxBin3),"binned_edu_num"] = "HS grad"
    adult.loc[(adult.loc[:,"edunum"] > MaxBin3) & (adult.loc[:,"edunum"] <= MaxBin4),"binned_edu_num"] = "some-college"
    adult.loc[(adult.loc[:,"edunum"] > MaxBin4) & (adult.loc[:,"edunum"] <= MaxBin5),"binned_edu_num"] = "bachelors"
    adult.loc[(adult.loc[:,"edunum"] > MaxBin5) & (adult.loc[:,"edunum"] <= MaxBin6),"binned_edu_num"] = "grad school"
    adult.loc[:,"binned_edu_num"].unique()
    
    #plot "binned_edu_num", compare to teh consolidated "edu"
    adult.loc[:,"binned_edu_num"].value_counts().plot(kind='bar')
    plt.title("Numerical Education after binning")
    plt.ylabel("Number of people")
    plt.xlabel("Education level")
    plt.show()
    
    #one hot encoding on "binned_edu_num"
    # Create 6 new columns, one for each state in "binned_edu_num"
    adult.loc[:, "elementary"] = (adult.loc[:, "binned_edu_num"] == "elementary").astype(int)
    adult.loc[:, "some-HS"] = (adult.loc[:, "binned_edu_num"] == "some-HS").astype(int)
    adult.loc[:, "HS-Grad"] = (adult.loc[:, "binned_edu_num"] == "HS Grad").astype(int)
    adult.loc[:, "some-college"] = (adult.loc[:, "binned_edu_num"] == "some-college").astype(int)
    adult.loc[:, "bachelors"] = (adult.loc[:, "binned_edu_num"] == "bachelors").astype(int)    
    adult.loc[:, "grad-school"] = (adult.loc[:, "binned_edu_num"] == "grad school").astype(int)
    

    # Remove obsolete column, notice axis is 1
    adult = adult.drop("binned_edu_num", axis=1)
    ##############
    
    #z-normalize edunum    
    adult.loc[:,"edunum"] = ((adult.loc[:,"edunum"]) - np.mean(adult.loc[:,"edunum"]))/np.std(adult.loc[:,"edunum"])
    ##############
   
    #capgain
    #there are apparently outliers, which have not been cleaned for this assignment 
    adult.loc[:,"capgain"] = ((adult.loc[:,"capgain"]) - np.mean(adult.loc[:,"capgain"]))/np.std(adult.loc[:,"capgain"])
    ##############
    
    #caploss
    #there are apparently outliers which have not been changed for this assignment 
    adult.loc[:,"caploss"] = ((adult.loc[:,"caploss"]) - np.mean(adult.loc[:,"caploss"]))/np.std(adult.loc[:,"caploss"])
    ##############
     
    #hrswk
    #there are apparently outliers which have not been changed for this assignment 
    adult.loc[:,"hrswk"].max()
    adult.loc[:,"hrswk"].min()
    adult.loc[:,"hrswk"] = ((adult.loc[:,"hrswk"]) - np.mean(adult.loc[:,"hrswk"]))/np.std(adult.loc[:,"hrswk"])
    ##############
     
    return (adult)
#end cln_num funciton 
###############################################################################
###############################################################################
###############################################################################
###############################################################################
    
#function calls
adult = acquire()
adult = cln_cat(adult)
adult = cln_num(adult)

#Delsey Sabu