#Delsey Sabu
#Data Cleaning
#function calls at bottom

import numpy as np

###########################
#function to create arr1
#function has no arguments, returns arr1
def create_arr1():
    #create arr1 and fill with 0-31
    arr1 = np.arange(32)
    #fill first and last element with outliers (-200,200)
    arr1[0] = -200
    arr1[31] = 200
    arr1
    return (arr1)
#end function create_arr1
###########################
    
###########################
#function to create arr2
#function has no arguments, returns arr2
def create_arr2():
    #create arr2 with some ints and non-numeric elements
    arr2 = np.array([2, 1, " ", 1, "!", 1, 5, 3, "?", 1, 4, 3])
    arr2
    return (arr2)
#end function create_arr2
###########################
    
###########################
#function to remove outliers
#function has one argument arr1, returns arr1
def remove_outlier(arr1):
    #High Limit of arr1 is the mean plus 2 standard deviations
    LimitHi = np.mean(arr1) + 2*np.std(arr1)
    #Low limit of arr1 is the mean minus 2 standard deviations
    LimitLo = np.mean(arr1) - 2*np.std(arr1)
    #create a flag that does not include outlier
    #outlier is between LimiLo and LimitHi
    FlagGood = (arr1>= LimitLo) & (arr1 <= LimitHi)
    #remove outliers in arr1
    arr1 = arr1[FlagGood]
    arr1
    return (arr1)
#end function remove_outlier
###########################
    
###########################
#function to replace outliers with mean of non-outliers
#funciton has one argument arr1, returns arr1
def replace_outlier(arr1):
    #High Limit of arr1 is the mean plus 2 standard deviations
    LimitHi = np.mean(arr1) + 2*np.std(arr1)
    #Low limit of arr1 is the mean minus 2 standard deviations
    LimitLo = np.mean(arr1) - 2*np.std(arr1)
    #create flag that is in within limit
    FlagGood = (arr1>= LimitLo) & (arr1 <= LimitHi)
    #create flag that is not within limits
    FlagBad = ~FlagGood
    #replace the elements not in limit with mean of values in limit
    arr1[FlagBad] = np.mean(arr1[FlagGood])
    arr1
    return (arr1)
#end function replace_outlier
###########################
    
###########################
#function to fill in non-numeric missing values with median of arr2
#function has one argument arr2, returns arr2
def fill_median(arr2):
    #create flag for elements which are digit
    FlagGood = [element.isdigit() for element in arr2]
    #ccreate flag for elemnts which are not digits
    FlagBad = [not element.isdigit() for element in arr2]
    #replace non-numeric values with median of numeric values
    #cast the numbers from text (string) to real numeric values to take median
    arr2[FlagBad] = np.median(arr2[FlagGood].astype(int))
    #cast the numbers from text (string) to real numeric values
    return (arr2.astype(float))
#end function fill_median
###########################
    
###########################
#funciton calls
#create array functions : create_arr1,create_arr2
#data cleaning functions : remove_outlier,replace_outlier,fill_median
#prints out the returned inputs of functions for ease of viewing
###########################  
#assign output of create_arr1 to arr1
arr1 = create_arr1()
#assign output of create_arr2 to arr2
arr2 = create_arr2()

print (create_arr1())
print (create_arr2())
print (remove_outlier(arr1))
print (replace_outlier(arr1))
print (fill_median(arr2))

#the data has been cleaned 
#arr1 was first shown with outliers -200, 200
#after remove_outlier function was called-
#we removed the outliers in arr1 and we can only see values 1-30
#after replace_outlier was called-
#we replaced the outliers -200,200 in arr1 with mean of the non-outlier, which was 15
#after fill_median was called-
#we filled in the missing data ("","!","?") in arr2 with median of arr2 (2)