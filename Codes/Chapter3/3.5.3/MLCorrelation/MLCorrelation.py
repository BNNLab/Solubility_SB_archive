'''
This script finds the correlation matrix for the predictions for the fixed train test splits based on machine learning method.
INPUTS:
MLs.csv - a .csv file containing predictions for each machine learning method, the machine learning methods must be named as in the global variable "ML_columns"
OUTPUTS:
ML_corr.csv - a .csv containing the correlation matrix
'''
##section 1: import modules
import pandas as pd
import os

##section 2: define inputs and outputs
dir=os.path.dirname(__file__)#get current directory to join to files
MLs=pd.read_csv(os.path.join(dir,"MLs.csv"))#location of descriptor file
output_corr=os.path.join(dir,"ML_corr.csv")#location of output file for correlation matrix

##section 3: define correlation method
##names of the machine learning columns in MLs.csv
ML_columns=["MLR","PLS","ANN","SVM","RF","ExtraTrees","Bagging","GPR"]
def get_corr(Data,output):
	##get just MLs
	Data=Data[ML_columns]
    ##get correlation matrix
	corr=Data.corr()
    ##convert to R-squared
	corr=corr.pow(2)
	##save output
	corr.to_csv(output)

##section 4: run method and save output
get_corr(MLs,output_corr)